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
    from ..Models import ModelType

from ..Utilities import Tic, _types

# fem
from ..FEM import LagrangeCondition

try:
    import pypardiso

    CAN_USE_PYPARDISO = True
except ModuleNotFoundError:
    CAN_USE_PYPARDISO = False

try:
    from mpi4py import MPI

    MPI_COMM = MPI.COMM_WORLD
    MPI_SIZE = MPI_COMM.Get_size()
    MPI_RANK = MPI_COMM.Get_rank()

    CAN_USE_MPI = True
except ModuleNotFoundError:
    CAN_USE_MPI = False

    MPI_COMM = None
    MPI_SIZE = 1
    MPI_RANK = 0

try:
    import petsc4py
    from petsc4py import PETSc

    CAN_USE_PETSC = True

except ModuleNotFoundError:
    CAN_USE_PETSC = False


class AlgoType(str, Enum):
    elliptic = "elliptic"
    r"""Solve :math:`\Krm \, \mathrm{u} = \Frm`"""
    parabolic = "parabolic"
    r"""Solve :math:`\Krm \, \mathrm{u}^{n+\alpha} + \Crm \, \vrm^{n+\alpha} = F^{n+\alpha}`"""
    newmark = "newmark"
    r"""Solve :math:`\Krm \, \mathrm{u}^{n+1} + \Crm \, \vrm^{n+1} + \Mrm \, a^{n+1} = F^{n+1}` \n
    :math:`\mathrm{u}^{n+1} = \mathrm{u}_n + \dt \, \vrm_n + \dt^2/2 ((1 - 2 \beta) a_n + 2 \beta a^{n+1})` \n
    :math:`\vrm^{n+1} = \vrm_n + \dt \, ((1 - \gamma) \, a_n + \gamma \, a^{n+1})`
    """
    midpoint = "midpoint"
    r"""Solve :math:`\Krm \, \mathrm{u}^{\frac{n+1}{2}} + \Crm \, \vrm^{\frac{n+1}{2}} + \Mrm \, a^{\frac{n+1}{2}} = F^{\frac{n+1}{2}}`"""
    hht = "hht"
    r"""Solve :math:`\Krm \, \mathrm{u}^{\frac{n+1-\alpha}{2}} + \Crm \, \vrm^{\frac{n+1-\alpha}{2}} + \Mrm \, a^{\frac{n+1-\alpha}{2}} = F^{\frac{n+1-\alpha}{2}}`"""

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def Get_Hyperbolic_Types() -> list[str]:
        return [AlgoType.newmark, AlgoType.midpoint, AlgoType.hht]

    @staticmethod
    def Get_Hyperbolic_and_Parabolic_Types() -> list[str]:
        algoTypes = AlgoType.Get_Hyperbolic_Types()
        algoTypes.append(AlgoType.parabolic)
        return algoTypes


class ResolType(str, Enum):
    """Resolution type."""

    r1 = "r1"
    """:math:`\mathrm{x}_\irm = {\Arm_{\irm\irm}}^{-1} \cdot (b_\irm - \Arm_{\irm\crm} * \mathrm{x}_\crm)`,
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

    if len(simu.Bc_Lagrange) > 0:
        # If the simulation uses Lagrange multipliers, iterative solvers cannot be employed.
        solver = SolverType.pypardiso if CAN_USE_PYPARDISO else SolverType.scipy
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

    tic = Tic()

    if CAN_USE_PYPARDISO and solver == SolverType.pypardiso:
        x = pypardiso.spsolve(A, b.toarray())

    elif CAN_USE_PETSC and solver == SolverType.petsc:

        # get petsc4py options
        kspType, pcType, solverType = simu._Solver_Get_PETSc4Py_Options()

        # get nodes
        mesh = simu.mesh
        if MPI_SIZE > 1:
            _, _, nodes, ghostNodes = mesh.groupElem._Get_partitionned_data()
            nodes = list(set(nodes).union(ghostNodes))
        else:
            nodes = mesh.nodes

        # get used dofs
        usedDofs = simu.Bc_dofs_nodes(
            nodes, simu.Get_unknowns(problemType), problemType
        )

        # from ..Utilities import Display
        # print(f"rank {MPI_RANK}: orphanNodes = {mesh.orphanNodes}")
        # ax = Display.Init_Axes(2)
        # ax.grid()
        # A = A[usedDofs, :].tocsc()[:, usedDofs]
        # ax.spy(A)
        # ax.set_title(f"rank {MPI_RANK}")
        # Display.plt.show()

        x, converged = _PETSc(A, b, x0, kspType, pcType, solverType, usedDofs)
        if not converged:
            raise Exception(
                f"petsc did not converge with ksp:{kspType}, pc:{pcType} and solver:{solverType}."
            )

        # add petsc4py options in solver description
        solver += f", {kspType}, {pcType}"
        if solverType != "petsc":
            solver += f", {solverType}"

    elif solver == SolverType.scipy:
        testSymetric = sla.norm(A - A.transpose()) / sla.norm(A)
        A_isSymetric = testSymetric <= 1e-12
        x = _ScipyLinearDirect(A, b, A_isSymetric)

    elif solver == SolverType.cg:
        x, output = sla.cg(A, b.toarray(), x0, maxiter=None)

    elif solver == SolverType.bicg:
        x, output = sla.bicg(A, b.toarray(), x0, maxiter=None)

    elif solver == SolverType.gmres:
        x, output = sla.gmres(A, b.toarray(), x0, maxiter=None)

    elif solver == "lgmres":
        x, output = sla.lgmres(A, b.toarray(), x0, maxiter=None)

    elif solver == SolverType.lsq_linear:
        # constrained minimization
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
        assert len(lb) == len(ub) != 0
        x = optimize.lsq_linear(
            A, b.toarray().ravel(), bounds=(lb, ub), tol=1e-10, method="trf", verbose=0
        )
        x = x["x"]

    else:
        raise NotImplementedError(f"{solver} is not implemented.")

    tic.Tac("Solver", f"Solve {problemType} ({solver})", simu._verbosity)

    # # A x - b = 0
    # res = np.linalg.norm(A.dot(x)-b.toarray().ravel())
    # print(res/np.linalg.norm(b.toarray().ravel()))

    return np.array(x)


def Solve_simu(simu: "_Simu", problemType: "ModelType"):
    """Solving the simulation's problem according to the resolution type."""

    resolution = ResolType.r1
    if CAN_USE_PETSC and MPI_SIZE > 1:
        resolution = ResolType.r3
        solverType = simu._Solver_Get_PETSc4Py_Options()[2]
        simu._Solver_Set_PETSc4Py_Options("none", "none", solverType)

    resolution = ResolType.r2 if len(simu.Bc_Lagrange) > 0 else resolution

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

    lb, ub = simu.Get_lb_ub(problemType)

    bi -= Aic @ xc
    xi = _Solve_Axb(simu, problemType, Aii, bi, x0, lb, ub)

    # apply result to global vector
    x = x.toarray().reshape(x.shape[0])
    x[dofsUnknown] = xi

    if simu.isNonLinear:
        return x, sla.norm(bi)
    else:
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
    A, b = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r3)

    # Solving the penalized matrix system
    x0 = simu.Get_x0(problemType)
    lb, ub = simu.Get_lb_ub(problemType)
    x = _Solve_Axb(simu, problemType, A, b, x0, lb, ub)

    if simu.isNonLinear:
        return x, sla.norm(b)
    else:
        return x


def _PETSc(
    A: sparse.csr_matrix,
    b: sparse.csr_matrix,
    x0: _types.FloatArray,
    kspType: str = "cg",
    pcType: str = "none",
    solverType: str = "petsc",
    global_dofs: _types.IntArray = None,
) -> tuple[_types.FloatArray, bool]:
    """PETSc insterface to solve the linear system `A x = b`\n
    KSP - Linear System Solvers: https://petsc.org/release/manual/ksp/#\n


    Parameters
    ----------
    A : sparse.csr_matrix
        sparse matrix (N, N)
    b : sparse.csr_matrix
        sparse vector (N, 1)
    x0 : _types.FloatArray
        initial guess (N)
    kspType : str, optional
        PETSc Krylov method, by default "cg"
        e.g. 'cg', 'bicg', 'gmres', 'bcgs', 'groppcg', ...\n
        https://petsc.org/release/manualpages/KSP/KSPType/#ksptype\n
    pcType : str, optional
        PETSc preconditioner, by default "none"
        e.g. 'none', 'ilu', 'bjacobi', 'icc', 'lu', 'jacobi', 'cholesky', ...\n
        https://petsc.org/release/manualpages/PC/PCType/#pctype\n
    solverType : str, optional
        PETSc Linear Solver, by default "petsc"
        e.g. 'petsc', 'mumps', 'superlu', 'superlu_dist', 'umfpack', 'cholesky' ...\n
        https://petsc.org/release/manual/ksp/#using-external-linear-solvers
    global_dofs : _types.IntArray, optional
        global dofs used to acces values in matrices, by default None

    Returns
    -------
    _types.FloatArray
        x solution to A x = b
    """

    # TODO add bound constrain
    # https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.SNES.html ?

    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"

    petsc4py.init(sys.argv, comm=MPI_COMM)

    matrix = PETSc.Mat()  # type: ignore [attr-defined]

    # get size and cols
    Ndof = A.shape[0]

    # init global to local converter
    global_to_local_converter = np.zeros(Ndof, dtype=np.int32)

    if MPI_SIZE > 1:
        # https://petsc.org/release/manual/mat/#matrices
        # create
        matrix.create(comm=MPI_COMM)
        matrix.setType("aij")

        # get sizes
        Ndof_r = global_dofs.size
        # assert Ndof == comm.allreduce(Ndof_r, MPI.SUM)

        # Resize A and get values
        # print(f"rank{MPI_RANK} global_dofs = {global_dofs}")
        assert isinstance(global_dofs, np.ndarray)
        A = A[global_dofs, :].tocsc()[:, global_dofs].tocsr()
        values = A.toarray().ravel()

        # set matrix size
        # https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html#petsc4py.PETSc.Mat.setSizes
        matrix.setSizes([[Ndof_r, Ndof], [Ndof_r, Ndof]])

        # set global to local converter
        global_to_local_converter[global_dofs] = np.arange(global_dofs.size)

        # get local dofs
        local_dofs = global_to_local_converter[global_dofs]

        # set values
        # https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.html#petsc4py.PETSc.Mat.setValues
        matrix.setValues(local_dofs, local_dofs, values, PETSc.InsertMode.INSERT_VALUES)
        matrix.assemble()
    else:
        # set global to local converter
        global_dofs = list(set(A.nonzero()[0]))
        global_to_local_converter = np.arange(Ndof)

        # set values
        csr = (A.indptr, A.indices, A.data)
        matrix.createAIJ(A.shape, comm=MPI_COMM, csr=csr)

    # set b values
    global_rows, _, values = sparse.find(b)
    # print(f"rank{MPI_RANK} b = \n{b.toarray()}")

    # set rhs values
    rhs = matrix.createVecLeft()
    local_rows = global_to_local_converter[global_rows]
    rhs.array[local_rows] = values
    # print(f"rank{MPI_RANK} rhs = {rhs.array}")

    # get local dofs
    local_dofs = global_to_local_converter[global_dofs]

    # set x values
    x = matrix.createVecRight()
    if len(x0) > 0:
        # print(x.array.shape)
        x.array[local_dofs] = x0[global_dofs]
    # print(f"rank{MPI_RANK} x = {x.array}")

    ksp = PETSc.KSP().create(comm=MPI_COMM)  # type: ignore [attr-defined]
    ksp.setOperators(matrix)
    ksp.setType(kspType)

    # set pc type
    pc = ksp.getPC()
    pc.setType(pcType)

    # set solver type
    pc.setFactorSolverType(solverType)

    # solve x
    ksp.solve(rhs, x)
    # print(f"rank{MPI_RANK} x = {x.array}")

    # set dofsValues values
    dofsValues = np.zeros(Ndof, dtype=float)
    dofsValues[global_dofs] = x.array[local_dofs]
    # print(f"rank{MPI_RANK} dofsValues = {dofsValues}")

    return dofsValues, ksp.is_converged


def _ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool):
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
    # LU decomposition behind https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html

    hideFacto = False  # Hide decomposition
    # permute = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"

    if A_isSymetric:
        permute = "MMD_AT_PLUS_A"
    else:
        permute = "COLAMD"
        # permute="NATURAL"

    if hideFacto:
        x = sla.spsolve(A, b, permc_spec=permute)
        # x = sla.spsolve(A, b)

    else:
        # superlu : https://portal.nersc.gov/project/sparse/superlu/
        # Users' Guide : https://portal.nersc.gov/project/sparse/superlu/ug.pdf
        lu = sla.splu(A.tocsc(), permc_spec=permute)
        x = lu.solve(b.toarray()).ravel()

    return x
