# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Interface module to various solvers available in Python for solving linear systems (A x = b)."""

import sys
from enum import Enum
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize
import scipy.sparse.linalg as sla
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from ._simu import _Simu
    from ..models import ModelType

from ..utilities import Tic, _types

# fem
from ..fem import LagrangeCondition

try:
    import pypardiso

    CAN_USE_PYPARDISO = True
except ModuleNotFoundError:
    CAN_USE_PYPARDISO = False

try:
    import mumps

    # from mumps import DMumpsContext
    CAN_USE_MUMPS = True
except ModuleNotFoundError:
    CAN_USE_MUMPS = False

try:
    import petsc4py
    from petsc4py import PETSc

    CAN_USE_PETSC = True
    PC_DEFAULT = "ilu"

except ModuleNotFoundError:
    CAN_USE_PETSC = False


class AlgoType(str, Enum):
    elliptic = "elliptic"
    """Solve K u = F"""
    parabolic = "parabolic"
    """Solve K u_npa + C v_npa = F_npa"""
    newmark = "newmark"
    """Solve K u_np1 + C v_np1 + M a_np1 = F_np1 \n
    u_np1 = u_n + dt v_n + dt^2/2 ((1 - 2 B) a_n + 2 B a_np1) \n
    v_np1 = v_n + dt ((1 - gamma) a_n + gamma a_np1)
    """
    midpoint = "midpoint"
    """Solve K u_np1/2 + C v_np1/2 + M a_np1/2 = F_np1/2"""
    hht = "hht"
    """Solve K u_np1ma + C v_np1ma + M a_np1ma = F_np1ma"""

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def Get_Hyperbolic_Types() -> list[str]:
        return [AlgoType.newmark, AlgoType.midpoint, AlgoType.hht]


class ResolType(str, Enum):
    r1 = "1"
    """xi = inv(Aii) * (bi - Aic * xc)"""
    r2 = "2"
    """Lagrange multipliers"""
    r3 = "3"
    """Penality"""

    def __str__(self) -> str:
        return self.name


def _Available_Solvers():
    """Available solvers."""

    solvers = ["scipy", "BoundConstrain", "cg", "bicg", "gmres", "lgmres"]

    if CAN_USE_PYPARDISO:
        solvers.insert(0, "pypardiso")
    if CAN_USE_PETSC:
        solvers.insert(1, "petsc")
    if CAN_USE_MUMPS:
        solvers.insert(2, "mumps")

    return solvers


def _Solve_Axb(
    simu: "_Simu",
    problemType: "ModelType",
    A: sparse.csr_matrix,
    b: sparse.csr_matrix,
    x0: _types.FloatArray,
    lb: Union[_types.AnyArray, _types.Numbers],
    ub: Union[_types.AnyArray, _types.Numbers],
) -> _types.FloatArray:
    """Solves the linear system A x = b

    Parameters
    ----------
    simu : Simu
        Simulation
    problemType : ModelType
        Specify the problemType because a simulation can have several physcal models (such as a damage simulation).
    A : sparse.csr_matrix
        matrix A
    b : sparse.csr_matrix
        vector b
    x0 : _types.FloatArray
        initial solution for iterative solvers
    lb : Union[_types.AnyArra, _types.Numbers]
        lowerBoundary of the solution
    ub : Union[_types.AnyArra, _types.Numbers]
        upperBoundary of the solution

    Returns
    -------
    _types.FloatArray
        comuted x solution of A x = b
    """

    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)

    if not isinstance(b, sparse.csr_matrix):
        b = sparse.csr_matrix(b)

    # Choose the solver
    if len(lb) > 0 and len(ub) > 0:
        solver = "BoundConstrain"
    else:
        if len(simu.Bc_Lagrange) > 0:
            # if lagrange multiplier are found we cannot use iterative solvers
            solver = "scipy"
        else:
            solver = simu.solver

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.spy(A, marker='.')

    if not A.has_canonical_format:
        # Using sla.norm(A) ensures that A has a cononic format.
        # Canonical Format means:
        # - Within each row, indices are sorted by column.
        # - There are no duplicate entries.
        sla.norm(A)

    solver = __Get_solver(solver)

    tic = Tic()

    if solver == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solver == "petsc":
        global PC_DEFAULT
        # TODO find the best for damage problem
        kspType = "cg"

        if simu.problemType == "damage":
            if problemType == "damage":
                pcType = "ilu"
            else:
                pcType = "none"
                # ilu decomposition doesn't seem to work for the displacement problem in a damage simulation

        elif simu.problemType == "hyperelastic":
            pcType = "none"

        else:
            pcType = PC_DEFAULT  # 'ilu' by default
            # if mesh.dim = 3, errors may occurs if we use ilu
            # works faster on 2D and 3D

        x, option, converg = _PETSc(A, b, x0, kspType, pcType)

        if not converg:
            print(
                f"\nWarning petsc did not converge with ksp:{kspType} and pc:{pcType} !"
            )
            print(f"Try out with  ksp:{kspType} and pc:none.\n")
            PC_DEFAULT = "none"
            x, option, converg = _PETSc(A, b, x0, kspType, "none")
            assert converg, "petsc didnt converge 2 times. check for kspType and pcType"

        solver += option

    elif solver == "scipy":
        testSymetric = sla.norm(A - A.transpose()) / sla.norm(A)
        A_isSymetric = testSymetric <= 1e-12
        x = _ScipyLinearDirect(A, b, A_isSymetric)

    elif solver == "BoundConstrain":
        x = _BoundConstrain(A, b, lb, ub)

    elif solver == "cg":
        x, output = sla.cg(A, b.toarray(), x0, maxiter=None)

    elif solver == "bicg":
        x, output = sla.bicg(A, b.toarray(), x0, maxiter=None)

    elif solver == "gmres":
        x, output = sla.gmres(A, b.toarray(), x0, maxiter=None)

    elif solver == "lgmres":
        x, output = sla.lgmres(A, b.toarray(), x0, maxiter=None)
        print(output)

    elif solver == "mumps":
        # # TODO dont work yet
        # ctx = DMumpsContext()
        # if ctx.myid == 0:
        #     ctx.set_centralized_sparse(A)
        #     x = b.copy()
        #     ctx.set_rhs(x) # Modified in place
        # ctx.run(job=6) # Analysis + Factorization + Solve
        # ctx.destroy() # Cleanup
        x = mumps.spsolve(A, b)

    tic.Tac("Solver", f"Solve {problemType} ({solver})", simu._verbosity)

    # # A x - b = 0
    # res = np.linalg.norm(A.dot(x)-b.toarray().ravel())
    # print(res/np.linalg.norm(b.toarray().ravel()))

    return np.array(x)


def __Get_solver(solver: str) -> str:
    """Checks whether the selected solver library is available
    If not, returns the solver usable in all cases (scipy)."""
    defaultSolver = "scipy"
    if solver == "pypardiso":
        return solver if CAN_USE_PYPARDISO else defaultSolver
    elif solver == "mumps":
        return solver if CAN_USE_MUMPS else defaultSolver
    elif solver == "petsc":
        return solver if CAN_USE_PETSC else defaultSolver
    else:
        return solver


def Solve_simu(simu: "_Simu", problemType: "ModelType"):
    """Solving the simulation's problem according to the resolution type."""

    resolution = ResolType.r2 if len(simu.Bc_Lagrange) > 0 else ResolType.r1

    if resolution == ResolType.r1:
        return __Solver_1(simu, problemType)
    elif resolution == ResolType.r2:
        return __Solver_2(simu, problemType)[0]
    elif resolution == ResolType.r3:
        return __Solver_3(simu, problemType)
    else:
        raise ValueError("Unknown resolution.")


def __Solver_1(simu: "_Simu", problemType: "ModelType") -> _types.FloatArray:
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

    tic = Tic()
    # split of the matrix system into known and unknown dofs
    # Solve : Aii * xi = bi - Aic * xc
    Ai = A[dofsUnknown, :].tocsc()
    Aii = Ai[:, dofsUnknown].tocsr()
    Aic = Ai[:, dofsKnown].tocsr()
    bi = b[dofsUnknown, 0]
    xc = x[dofsKnown, 0]

    tic.Tac("Solver", f"System-built ({problemType})", simu._verbosity)

    x0 = simu.Get_x0(problemType)
    x0 = x0[dofsUnknown]

    bDirichlet = Aic @ xc

    lb, ub = simu.Get_lb_ub(problemType)

    xi = _Solve_Axb(simu, problemType, Aii, bi - bDirichlet, x0, lb, ub)

    # apply result to global vector
    x = x.toarray().reshape(x.shape[0])
    x[dofsUnknown] = xi

    return x


def __Solver_2(simu: "_Simu", problemType: "ModelType"):
    # Lagrange multiplier method

    size = simu.mesh.Nn * simu.Get_dof_n(problemType)

    # Build the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r2)
    alpha = A.data.max()

    tic = Tic()

    # set to lil matrix because its faster
    A = A.tolil()
    b = b.tolil()

    dofs_Dirichlet = simu.Bc_dofs_Dirichlet(problemType)
    values_Dirichlet = simu.Bc_values_Dirichlet(problemType)

    list_Bc_Lagrange = simu.Bc_Lagrange

    nLagrange = len(list_Bc_Lagrange)
    nDirichlet = len(dofs_Dirichlet)
    nCol = nLagrange + nDirichlet

    x0 = simu.Get_x0(problemType)
    x0 = np.append(x0, np.zeros(nCol))

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

    x = _Solve_Axb(simu, problemType, A.tocsr(), b.tocsr(), x0, [], [])

    # We don't send back reaction forces
    sol = x[:size]
    lagrange = x[size:]

    return sol, lagrange


def __Solver_3(simu: "_Simu", problemType: "ModelType"):
    # Resolution using the penalty method

    # This method does not give preference to dirichlet conditions over neumann conditions.
    # This means that if a dof is applied in Neumann and in Dirichlet, it will be privileged over the dof applied in Neumann.

    # This method is never used. It is just implemented as an example

    # Builds the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r3)

    # Solving the penalized matrix system
    x0 = simu.Get_x0(problemType)
    lb, ub = simu.Get_lb_ub(problemType)
    x = _Solve_Axb(simu, problemType, A, b, x0, lb, ub)

    return x


def _PETSc(
    A: sparse.csr_matrix,
    b: sparse.csr_matrix,
    x0: _types.FloatArray,
    kspType: str = "cg",
    pcType: str = "ilu",
) -> tuple[_types.FloatArray, str, bool]:
    """PETSc insterface to solve the linear system A x = b

    Parameters
    ----------
    A : sparse.csr_matrix
        sparse matrix (N, N)
    b : sparse.csr_matrix
        sparse vector (N, 1)
    x0 : _types.FloatArray
        initial guess (N)
    kspType : str, optional
        PETSc Krylov method, by default 'cg'\n
        "cg", "bicg", "gmres", "bcgs", "groppcg"\n
        https://petsc.org/release/manualpages/KSP/KSPType/
    pcType : str, optional
        preconditioner, by default 'ilu'\n
        "ilu", "none", "bjacobi", 'icc', "lu", "jacobi", "cholesky"\n
        # TODO iluk ?
        more -> https://petsc.org/release/manualpages/PC/PCType/\n
        remark : The ilu preconditioner does not seem to work for systems using HEXA20 elements.

    Returns
    -------
    _types.FloatArray
        x solution to A x = b
    """

    # # TODO make it work with mpi
    # __comm = MPI.COMM_WORLD
    # nprocs = __comm.Get_size()
    # rank   = __comm.Get_rank()

    # TODO add bound constrain
    # https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.SNES.html ?

    __comm = None
    petsc4py.init(sys.argv, comm=__comm)

    dimI = A.shape[0]
    dimJ = A.shape[1]

    matrix = PETSc.Mat()  # type: ignore [attr-defined]
    csr = (A.indptr, A.indices, A.data)
    matrix.createAIJ([dimI, dimJ], comm=__comm, csr=csr)

    vectb = matrix.createVecLeft()

    lines, _, values = sparse.find(b)

    vectb.array[lines] = values

    x = matrix.createVecRight()
    if len(x0) > 0:
        x.array[:] = x0

    ksp = PETSc.KSP().create()  # type: ignore [attr-defined]
    ksp.setOperators(matrix)
    ksp.setType(kspType)

    pc = ksp.getPC()
    pc.setType(pcType)

    # pc.setFactorSolverType("superlu") #"mumps"

    ksp.solve(vectb, x)
    x = x.array

    converg: bool = ksp.is_converged

    # PETSc._finalize()

    option = f", {kspType}, {pcType}"

    return x, option, converg


def _ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool):
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
    # LU decomposition behind https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html

    hideFacto = False  # Hide decomposition
    # permute = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"

    # if A_isSymetric:
    #     permute="MMD_AT_PLUS_A"
    # else:
    #     permute="COLAMD"
    #     # permute="NATURAL"

    permute = "MMD_AT_PLUS_A"

    if hideFacto:
        x = sla.spsolve(A, b, permc_spec=permute)
        # x = sla.spsolve(A, b)

    else:
        # superlu : https://portal.nersc.gov/project/sparse/superlu/
        # Users' Guide : https://portal.nersc.gov/project/sparse/superlu/ug.pdf
        lu = sla.splu(A.tocsc(), permc_spec=permute)
        x = lu.solve(b.toarray()).ravel()

    return x


def _BoundConstrain(
    A,
    b,
    lb: Union[_types.AnyArray, _types.Numbers],
    ub: Union[_types.AnyArray, _types.Numbers],
):
    assert len(lb) == len(ub), "Must be the same size"

    # constrained minimization : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    b = b.toarray().ravel()
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb, ub), tol=tol, method="trf", verbose=0)
    x = x["x"]

    return x
