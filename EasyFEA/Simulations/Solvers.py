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
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ._simu import _Simu
    from ._problem_type import ProblemType

from ..Utilities import Tic, _types

from ..Utilities._mpi import MPI_COMM, MPI_SIZE, Reduce_sum

from ._linear_system import _LinearSystem

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
    derived: bool = False,
) -> _types.FloatArray:
    """x of `A x = b`, through PETSc at `system.rows` only (`derived`: A is the system's derived matrix); a one-off system when None."""

    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)

    b = np.asarray(b).ravel()

    # a system made here is new at each solve: its PETSc solver is not kept
    oneOff = system is None
    comm = None if system is None else simu.mesh.comm
    if system is None:
        system = _LinearSystem()
        system.Set_known(np.zeros(A.shape[0], dtype=bool), None)

    solver = simu.solver
    if isinstance(problemType, tuple) and __Has_zero_diagonal(A, system, comm):
        _Check_saddle_point_solver(simu, problemType)

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

        x, converged = _PETSc(
            system,
            A,
            b,
            x0,
            (kspType, pcType, solverType),
            comm,
            derived,
        )
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


def _Check_saddle_point_solver(
    simu: "_Simu", problem: "ProblemType | tuple[ProblemType, ...]"
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
    names = ", ".join(problem) if isinstance(problem, tuple) else problem
    accepted = (
        "petsc with preonly, lu and mumps or superlu_dist"
        if MPI_SIZE > 1
        else "pypardiso, scipy or petsc with preonly, lu and an external package (mumps)"
    )
    raise ValueError(
        f"({names}) is a saddle point (a zero diagonal block or Lagrange multipliers), which only a direct solver factors: use {accepted}, not {solver}."
    )


def _Backend(
    simu: "_Simu",
    problemType: "ProblemType | tuple[ProblemType, ...]",
    system: _LinearSystem,
) -> Callable[
    [sparse.csr_matrix, _types.FloatArray, _types.IntArray, bool], _types.FloatArray
]:
    """`(matrix, rhs, rows, derived) -> x at rows` through the simulation's solver: PETSc on the system's numbering, the others on `rhs[rows]`."""
    x0 = simu._Solver_Get_x0(problemType)
    lb, ub = simu._Solver_Get_lb_ub(problemType)

    def Solve(
        matrix: sparse.csr_matrix,
        rhs: _types.FloatArray,
        rows: _types.IntArray,
        derived: bool,
    ) -> _types.FloatArray:
        # zero initial multipliers
        x0Full = np.zeros(rhs.size)
        x0Full[: x0.size] = x0
        if simu.solver == SolverType.petsc:
            return _Solve_Axb(
                simu,
                problemType,
                matrix,
                rhs,
                x0Full,
                lb,
                ub,
                system,
                derived=derived,
            )
        return _Solve_Axb(
            simu,
            problemType,
            matrix,
            rhs[rows],
            x0Full[rows],
            lb,
            ub,
        )

    return Solve


def _PETSc(
    system: _LinearSystem,
    A: sparse.csr_matrix,
    b: _types.FloatArray,
    x0: _types.FloatArray,
    options: tuple[str, str, str],
    comm,
    derived: bool = False,
) -> tuple[_types.FloatArray, bool]:
    """x at `system.rows` of `A x = b` and whether PETSc converged, through the solver the system keeps for A or its derived matrix. Collective on `comm`."""
    PETSc = _Get_PETSc()
    assert PETSc is not None, "petsc4py is not available"

    relayout, refill = system.Solver_update(comm, derived)
    if relayout:
        # two factorisations never held at once
        _PETSc_Free(system)
        solver = __Create_PETSc(
            system,
            PETSc,
            A,
            options,
            comm,
        )
    else:
        solver = system.solver
    ksp, matrix = solver
    if refill:
        matrix.setValuesCOO(A.data)
    system.Set_solver(solver, derived)

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
