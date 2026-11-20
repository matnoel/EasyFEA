# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Interface module to various solvers available in Python for solving linear systems (A x = b)."""

import sys
import warnings
from enum import Enum
from functools import cache
from types import ModuleType
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sla
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._simu import _Simu
    from ._problem_type import ProblemType

from ..Utilities import Tic, _types

from ..Utilities._mpi import MPI_COMM, MPI_SIZE, Reduce_sum

from ._linear_system import _LinearSystem

# fem
from ..FEM import LagrangeCondition

try:
    import pypardiso

    CAN_USE_PYPARDISO = True
except ModuleNotFoundError:
    CAN_USE_PYPARDISO = False


@cache
def _Get_PETSc() -> ModuleType | None:
    """Returns the initialized `petsc4py.PETSc` module, or None if petsc4py is unusable."""
    try:
        import petsc4py

        # must precede `from petsc4py import PETSc`, which would initialize PETSc with an empty command line
        petsc4py.init(
            [sys.argv[0], "-options_left", "no", *sys.argv[1:]], comm=MPI_COMM
        )

        from petsc4py import PETSc

        return PETSc

    except Exception:
        return None


class ResolType(str, Enum):
    """Resolution type."""

    r1 = "r1"
    r""":math:`\mathrm{x}_\irm = {\Arm_{\irm\irm}}^{-1} \cdot (b_\irm - \Arm_{\irm\crm} * \mathrm{x}_\crm)`,
    where :math:`\irm` and :math:`\crm` are unknown and known degrees of freedom."""
    r2 = "r2"
    """Lagrange multipliers"""
    r3 = "r3"
    """Penality"""

    def __str__(self) -> str:
        return self.name


class SolverType(str, Enum):
    r"""Solver type used to solve the linear system :math:`\Arm \, \mathrm{x} = \brm`"""

    pypardiso = "pypardiso"
    """pypardiso.spsolve"""

    petsc = "petsc"

    scipy = "scipy"
    """scipy.sparse.linalg.spsolve"""
    lsq_linear = "lsq_linear"
    """scipy.optimize.lsq_linear"""
    cg = "cg"
    """scipy.sparse.linalg.cg"""
    bicg = "bicg"
    """scipy.sparse.linalg.bicg"""
    gmres = "gmres"
    """scipy.sparse.linalg.gmres"""
    lgmres = "lgmres"
    """scipy.sparse.linalg.lgmres"""

    def __str__(self):
        return self.name


def _Solve_Axb(
    simu: "_Simu",
    problemType: "ProblemType | tuple[ProblemType, ...]",
    A: sparse.csr_matrix,
    b: _types.FloatArray,
    x0: _types.FloatArray,
    lb: _types.AnyArray | _types.Numbers,
    ub: _types.AnyArray | _types.Numbers,
    system: _LinearSystem | None = None,
) -> _types.FloatArray:
    """Solves the linear system A x = b

    Parameters
    ----------
    simu : Simu
        Simulation
    problemType : ProblemType | tuple[ProblemType, ...]
        Specify the problemType because a simulation can have several physcal models (such as a damage simulation); a tuple for a monolithic problem's fields.
    A : sparse.csr_matrix
        matrix A
    b : _types.FloatArray
        vector b (N,)
    x0 : _types.FloatArray
        initial solution for iterative solvers
    lb : _types.AnyArra | _types.Numbers
        lowerBoundary of the solution
    ub : _types.AnyArra | _types.Numbers
        upperBoundary of the solution
    system : _LinearSystem, optional
        For PETSc, the numbering and kept solver; by default every row, one process, nothing kept.

    Returns
    -------
    _types.FloatArray
        comuted x solution of A x = b, through PETSc only at `system.rows`
    """

    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)

    b = np.asarray(b).ravel()

    # a system made here is new at each solve: its PETSc solver is not kept
    oneOff = system is None
    comm = None if system is None else simu.mesh.comm
    if system is None:
        system = _LinearSystem()
        system.Set_known(np.zeros(A.shape[0], dtype=bool), None)

    if len(simu.Bc_Lagrange) > 0:
        if MPI_SIZE > 1:
            raise NotImplementedError(
                "Simulations using Lagrangian boundary conditions are not yet supported in MPI environments."
            )
        # Lagrange multipliers yield a saddle-point system: iterative KSP methods diverge.
        # Keep PETSc only when configured as a direct solver (preonly + lu/cholesky
        # backed by mumps, superlu_dist, or petsc in serial). Otherwise fall back.
        solver = SolverType.pypardiso if CAN_USE_PYPARDISO else SolverType.scipy
        if simu.solver == SolverType.petsc and _Get_PETSc() is not None:
            kspType, pcType, solverType = simu._Solver_Get_PETSc4Py_Options(problemType)
            # Built-in PETSc LU (solverType="petsc") uses SeqAIJ which requires a
            # non-zero diagonal — saddle-point matrices from Lagrange multipliers violate
            # this. External solvers (mumps, superlu_dist, …) use fill-in reordering
            # (AMD/METIS) and handle zero-diagonal rows correctly.
            if (
                kspType == "preonly"
                and pcType in {"lu", "cholesky"}
                and solverType != "petsc"
            ):
                solver = SolverType.petsc
    else:
        solver = simu.solver
        if isinstance(problemType, tuple) and __Has_zero_diagonal(A, system, comm):
            __Check_saddle_point_solver(simu, problemType)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.spy(A, marker='.')

    if not A.has_canonical_format:
        # Using sla.norm(A) ensures that A has a cononic format.
        # Canonical Format means:
        # - Within each row, indices are sorted by column.
        # - There are no duplicate entries.
        sla.norm(A)

    tic = Tic()

    if CAN_USE_PYPARDISO and solver == SolverType.pypardiso:
        x = pypardiso.spsolve(A, b)

    elif solver == SolverType.petsc and _Get_PETSc() is not None:
        # get petsc4py options
        kspType, pcType, solverType = simu._Solver_Get_PETSc4Py_Options(problemType)

        x, converged = _PETSc(system, A, b, x0, (kspType, pcType, solverType), comm)
        if oneOff:
            _PETSc_Free(system)
        if not converged:
            raise Exception(
                f"petsc did not converge with ksp:{kspType}, pc:{pcType} and solver:{solverType}."
            )

        # add petsc4py options in solver description
        solver = f"{kspType}, {pcType}, {solverType}"

    elif solver == SolverType.scipy:
        x = sla.spsolve(A, b)

    elif solver == SolverType.cg:
        x, output = sla.cg(A, b, x0, maxiter=None)

    elif solver == SolverType.bicg:
        x, output = sla.bicg(A, b, x0, maxiter=None)

    elif solver == SolverType.gmres:
        x, output = sla.gmres(A, b, x0, maxiter=None)

    elif solver == SolverType.lgmres:
        x, output = sla.lgmres(A, b, x0)

    elif solver == SolverType.lsq_linear:
        # constrained minimization
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
        assert len(lb) == len(ub) != 0
        from scipy.optimize import lsq_linear

        x = lsq_linear(
            A,
            b,
            bounds=(lb, ub),
            tol=1e-10,
            method="trf",
            verbose=0,
        )
        x = x["x"]

    else:
        raise NotImplementedError(f"{solver} is not implemented.")

    tic.Tac("Solver", f"Solve {problemType} ({solver})", simu._verbosity)

    # # A x - b = 0
    # res = np.linalg.norm(A.dot(x)-b.toarray().ravel())
    # print(res/np.linalg.norm(b.toarray().ravel()))

    return np.array(x)


def __Has_zero_diagonal(A: sparse.csr_matrix, system: _LinearSystem, comm) -> bool:
    """Whether a row the system's numbering keeps has no diagonal entry, on any rank (a saddle point); only owned rows are complete under MPI."""
    diagonal = A.diagonal()[system.rows]
    return Reduce_sum(int(np.count_nonzero(diagonal == 0)), comm) > 0


def __Check_saddle_point_solver(
    simu: "_Simu", problem: "tuple[ProblemType, ...]"
) -> None:
    """Rejects a solver that cannot factor a saddle point: a direct one only, PETSc's LU through an external package (Mumps or SuperLU_dist under MPI)."""
    solver = simu.solver
    if solver in (SolverType.pypardiso, SolverType.scipy) and MPI_SIZE == 1:
        return
    if solver == SolverType.petsc:
        kspType, pcType, solverType = simu._Solver_Get_PETSc4Py_Options(problem)
        external = (
            solverType in ("mumps", "superlu_dist")
            if MPI_SIZE > 1
            else solverType != "petsc"
        )
        if kspType == "preonly" and pcType == "lu" and external:
            return
        solver = f"petsc ({kspType}, {pcType}, {solverType})"
    names = ", ".join(problem)
    accepted = (
        "petsc with preonly, lu and mumps or superlu_dist"
        if MPI_SIZE > 1
        else "pypardiso, scipy or petsc with preonly, lu and an external package (mumps)"
    )
    raise ValueError(
        f"({names}) is a saddle point (a zero diagonal block), which only a direct solver factors: use {accepted}, not {solver}."
    )


def Solve_simu(
    simu: "_Simu", problemType: "ProblemType | tuple[ProblemType, ...]"
) -> tuple[_types.FloatArray, _types.FloatArray | None | float]:
    """Solving the simulation's problem according to the resolution type."""

    resolution = ResolType.r1
    resolution = ResolType.r2 if len(simu.Bc_Lagrange) > 0 else resolution

    if resolution == ResolType.r1:
        x, rhsNorm = __Solver_1(simu, problemType)
    elif resolution == ResolType.r2:
        # a monolithic problem refuses Lagrange conditions before solving
        assert not isinstance(problemType, tuple)
        x, lagrange = __Solver_2(simu, problemType)
    elif resolution == ResolType.r3:
        x, rhsNorm = __Solver_3(simu, problemType)
    else:
        raise NotImplementedError

    # x is right at the owned dofs only
    simu._Get_linear_system(problemType).Scatter_forward(x, simu.mesh.comm)

    if resolution in [ResolType.r1, ResolType.r3]:
        return x, rhsNorm
    elif resolution == ResolType.r2:
        return x, lagrange
    else:
        raise NotImplementedError


def __Residual_norm(b_owned: _types.FloatArray, comm) -> float:
    """Residual 2-norm over the ranks of `comm`, the root of the summed squares; raises on NaN/Inf"""
    rhsNorm = float(np.sqrt(Reduce_sum(np.linalg.norm(b_owned) ** 2, comm)))
    if not np.isfinite(rhsNorm):
        raise FloatingPointError("The residual has NaN or Inf entries.")
    return rhsNorm


def __Solver_1(
    simu: "_Simu", problemType: "ProblemType | tuple[ProblemType, ...]"
) -> tuple[_types.FloatArray, float | None]:
    # --       --  --  --   --  --
    # | Aii Aic |  | xi |   | bi |
    # | Aci Acc |  | xc | = | bc |
    # --       --  --  --   --  --
    # xi = inv(Aii) * (bi - Aic * xc)

    # Build the matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r1)

    # Recover dofs
    dofsKnown, dofsUnknown = simu.Bc_dofs_known_unknown(problemType)

    x0 = simu._Solver_Get_x0(problemType)
    lb, ub = simu._Solver_Get_lb_ub(problemType)

    if simu.solver == SolverType.petsc:
        # PETSc drops the known rows and columns itself, numbered -1
        known = np.zeros(x.size, dtype=bool)
        known[dofsKnown] = True
        comm = simu.mesh.comm
        system = simu._Get_linear_system(problemType)
        system.Set_known(known, comm)
        # the known values of the ghosts from their owner
        system.Scatter_forward(x, comm)
        residual = b - A @ x
        if simu.isNonLinear:
            rhsNorm = __Residual_norm(residual[system.rows], comm)
        x[system.rows] = _Solve_Axb(
            simu,
            problemType,
            A,
            residual,
            x0,
            lb,
            ub,
            system,
        )

    else:
        tic = Tic()
        # split of the matrix system into known and unknown dofs
        # Solve : Aii * xi = bi - Aic * xc
        Ai = A[dofsUnknown, :].tocsc()
        Aii = Ai[:, dofsUnknown]
        Aic = Ai[:, dofsKnown]
        bi = b[dofsUnknown]
        xc = x[dofsKnown]

        tic.Tac("Solver", f"System-built ({problemType})", simu._verbosity)

        bi -= Aic @ xc
        if simu.isNonLinear:
            rhsNorm = __Residual_norm(bi, simu.mesh.comm)

        x[dofsUnknown] = _Solve_Axb(
            simu,
            problemType,
            Aii,
            bi,
            x0[dofsUnknown],
            lb,
            ub,
        )

    if simu.isNonLinear:
        return x, rhsNorm
    else:
        return x, None


def __Solver_2(simu: "_Simu", problemType: "ProblemType"):
    # Lagrange multiplier method

    size = simu._Get_FunctionSpace(problemType).Nn * simu.Get_dof_n(problemType)

    # Build the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r2)
    alpha = A.data.max()

    tic = Tic()

    # set to lil matrix because its faster
    A = A.tolil()

    dofs_Dirichlet = simu.Bc_dofs_Dirichlet(problemType)
    values_Dirichlet = simu.Bc_values_Dirichlet(problemType)
    # each orphan not under Dirichlet held at 0 by a Lagrange row appended after Ndof
    orphans = np.setdiff1d(simu._Bc_dofs_orphan(problemType), dofs_Dirichlet)
    Ndof = A.shape[0]
    A.resize((Ndof + orphans.size, Ndof + orphans.size))
    b = np.append(b, np.zeros(orphans.size))
    linesOrphan = np.arange(Ndof, Ndof + orphans.size)
    A[linesOrphan, orphans] = alpha
    A[orphans, linesOrphan] = alpha

    list_Bc_Lagrange = simu.Bc_Lagrange

    nLagrange = len(list_Bc_Lagrange)
    nDirichlet = len(dofs_Dirichlet)
    nCol = nLagrange + nDirichlet

    x0 = simu.Get_x0(problemType)
    x0 = np.append(x0, np.zeros(nCol + orphans.size))

    linesDirichlet = np.arange(size, size + nDirichlet)

    # apply lagrange multiplier
    A[linesDirichlet, dofs_Dirichlet] = alpha
    A[dofs_Dirichlet, linesDirichlet] = alpha
    b[linesDirichlet] = values_Dirichlet * alpha

    tic.Tac("Solver", f"Lagrange ({problemType}) Dirichlet", simu._verbosity)

    # For each lagrange condition we will add a coef to the matrix
    if len(list_Bc_Lagrange) > 0:

        def __apply_lagrange(i: int, lagrangeBc: LagrangeCondition):
            dofs = lagrangeBc.dofs
            values = lagrangeBc.dofsValues * alpha
            coefs = lagrangeBc.lagrangeCoefs * alpha

            A[dofs, i] = coefs
            A[i, dofs] = coefs
            b[i] = values[0]

        start = size + nDirichlet
        [
            __apply_lagrange(i, lagrangeBc)
            for i, lagrangeBc in enumerate(list_Bc_Lagrange, start)
        ]

    tic.Tac("Solver", f"Lagrange ({problemType}) Coupling", simu._verbosity)

    x = _Solve_Axb(
        simu,
        problemType,
        A.tocsr(),
        b,
        x0,
        [],
        [],
    )

    # We don't send back reaction forces
    sol = x[:size]
    lagrange = x[size:Ndof]

    return sol, lagrange


def __Solver_3(simu: "_Simu", problemType: "ProblemType | tuple[ProblemType, ...]"):
    # Resolution using the penalty method

    # This method does not give preference to dirichlet conditions over neumann conditions.
    # This means that if a dof is applied in Neumann and in Dirichlet, it will be privileged over the dof applied in Neumann.

    # This method is never used. It is just implemented as an example

    # Builds the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, b = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r3)

    # Solving the penalized matrix system
    x0 = simu._Solver_Get_x0(problemType)
    lb, ub = simu._Solver_Get_lb_ub(problemType)
    if simu.solver == SolverType.petsc:
        # every dof unknown, each rank its owned rows
        comm = simu.mesh.comm
        system = simu._Get_linear_system(problemType)
        system.Set_known(np.zeros(b.size, dtype=bool), comm)
        if simu.isNonLinear:
            rhsNorm = __Residual_norm(b[system.rows], comm)
        x = np.zeros(b.size, dtype=float)
        # A penalised copy, not the system's A: a solver of its own, freed after
        _PETSc_Free(system)
        x[system.rows] = _Solve_Axb(
            simu,
            problemType,
            A,
            b,
            x0,
            lb,
            ub,
            system,
        )
        _PETSc_Free(system)
    else:
        if simu.isNonLinear:
            rhsNorm = __Residual_norm(b, simu.mesh.comm)
        x = _Solve_Axb(
            simu,
            problemType,
            A,
            b,
            x0,
            lb,
            ub,
        )

    if simu.isNonLinear:
        return x, rhsNorm
    else:
        return x, None


def _PETSc(
    system: _LinearSystem,
    A: sparse.csr_matrix,
    b: _types.FloatArray,
    x0: _types.FloatArray,
    options: tuple[str, str, str],
    comm,
) -> tuple[_types.FloatArray, bool]:
    """x at `system.rows` of `A x = b` and whether PETSc converged, through the solver the system keeps. Collective on `comm`."""
    PETSc = _Get_PETSc()
    assert PETSc is not None, "petsc4py is not available"

    relayout, refill = system.Solver_update(comm)
    if relayout:
        # two factorisations never held at once
        _PETSc_Free(system)
        solver = __Create_PETSc(system, PETSc, A, options, comm)
    else:
        solver = system.solver
    ksp, matrix = solver
    if refill:
        matrix.setValuesCOO(A.data)
    system.Set_solver(solver)

    x, rhs = matrix.createVecs()
    rhs.array[:] = b[system.rows]
    if len(x0) > 0:
        x.array[:] = x0[system.rows]

    ksp.solve(rhs, x)

    values = x.array.copy()
    x.destroy()
    rhs.destroy()

    return values, ksp.is_converged


def __Create_PETSc(
    system: _LinearSystem,
    PETSc: ModuleType,
    A: sparse.csr_matrix,
    options: tuple[str, str, str],
    comm,
) -> tuple:
    """KSP and Mat laid out on A's pattern and the system's numbering, unfilled."""
    kspType, pcType, solverType = options
    comm = comm or PETSc.COMM_SELF
    # -1 rows and columns are dropped: ghost rows, known dofs
    rowMap = PETSc.LGMap().create(system.rowIds.astype(PETSc.IntType), comm=comm)
    columnMap = PETSc.LGMap().create(system.ids.astype(PETSc.IntType), comm=comm)
    matrix = PETSc.Mat().create(comm=comm)
    matrix.setType("aij")
    matrix.setSizes(((system.rows.size, None), (system.rows.size, None)))
    matrix.setLGMap(rowMap, columnMap)
    rowMap.destroy()
    columnMap.destroy()
    # copies: PETSc overwrites them
    rowsCoo = np.repeat(np.arange(A.shape[0], dtype=PETSc.IntType), np.diff(A.indptr))
    matrix.setPreallocationCOOLocal(rowsCoo, A.indices.astype(PETSc.IntType))

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(matrix)
    ksp.setType(kspType)
    pc = ksp.getPC()
    pc.setType(pcType)
    pc.setFactorSolverType(solverType)
    petscOptions = PETSc.Options()
    if solverType == "mumps":
        if not petscOptions.hasName("mat_mumps_icntl_20"):
            # centralised right-hand side, see the warning; a no-op on one process
            petscOptions.setValue("mat_mumps_icntl_20", 0)
        elif comm.Get_size() > 1 and petscOptions.getInt("mat_mumps_icntl_20") == 10:
            warnings.warn(
                "mat_mumps_icntl_20 10 (distributed right-hand side) gave wrong solutions with Mumps 5.8.2 and 5.9.1 over several processes: set it to 0.",
                RuntimeWarning,
            )
    ksp.setFromOptions()
    return ksp, matrix


def _PETSc_Free(system: _LinearSystem) -> None:
    """Destroys the KSP and Mat the system kept. Collective on their communicator."""
    solver = system.Take_solver()
    if solver is None:
        return
    ksp, matrix = solver
    # flush any remaining petsc4py reference cycles as suggested here:
    # https://gitlab.com/petsc/petsc/-/work_items/1309
    PETSc = _Get_PETSc()
    assert PETSc is not None
    PETSc.garbage_cleanup(ksp.getComm())
    ksp.destroy()
    matrix.destroy()
