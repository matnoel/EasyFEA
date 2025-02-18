# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Interface module to various solvers available in Python for solving linear systems (A x = b)."""

import sys
from enum import Enum
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize
import scipy.sparse.linalg as sla

# utilities
from ..utilities import Tic
# fem
from ..fem import LagrangeCondition

try:
    import pypardiso
    __canUsePypardiso = True
except ModuleNotFoundError:
    __canUsePypardiso = False

try:
    from sksparse.cholmod import cholesky, cholesky_AAt
    __canUseCholesky = True
except ModuleNotFoundError:
    __canUseCholesky = False

try:
    from scikits.umfpack import umfpackSpsolve    
    __canUseUmfpack = True
except (ModuleNotFoundError, ImportError):
    __canUseUmfpack = False

try:
    import mumps
    # from mumps import DMumpsContext
    __canUseMumps = True
except ModuleNotFoundError:
    __canUseMumps = False

try:
    import petsc4py    
    from petsc4py import PETSc    
    from mpi4py import MPI
    
    __canUsePetsc = True
    __pc_default = 'ilu'

except ModuleNotFoundError:
    __canUsePetsc = False

class AlgoType(str, Enum):
    elliptic = "elliptic"
    """Solve K u = F"""
    parabolic = "parabolic"
    """Solve K u + C v = F"""
    hyperbolic = "hyperbolic"
    """Solve K u + C v + M a = F"""

    def __str__(self) -> str:
        return self.name

class ResolType(str, Enum):
    r1 = "1"
    """xi = inv(Aii) * (bi - Aic * xc)"""
    r2 = "2"
    """Lagrange multipliers"""
    r3 = "3"
    """Penalty"""

    def __str__(self) -> str:
        return self.name

def _Available_Solvers():
    """Available solvers."""

    solvers = ["scipy", "BoundConstrain", "cg", "bicg", "gmres", "lgmres"]
    
    if __canUsePypardiso: solvers.insert(0, "pypardiso")
    if __canUsePetsc: solvers.insert(1, "petsc")
    if __canUseMumps: solvers.insert(2, "mumps")
    if __canUseUmfpack: solvers.insert(3, "umfpack")

    return solvers

def __Cast_Simu(simu):
    """casts the simu as a Simulations.Simu"""
    from ._simu import _Simu
    if isinstance(simu, _Simu):
        return simu

def _Solve_Axb(simu, problemType: str,
               A: sparse.csr_matrix, b: sparse.csr_matrix,
               x0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
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
    x0 : np.ndarray
        initial solution for iterative solvers
    lb : np.ndarray
        lowerBoundary of the solution
    ub : np.ndarray
        upperBoundary of the solution

    Returns
    -------
    np.ndarray
        comuted x solution of A x = b
    """

    # checks types
    simu = __Cast_Simu(simu)
    assert isinstance(A, sparse.csr_matrix)
    assert isinstance(b, sparse.csr_matrix)

    # Choose the solver
    if len(lb) > 0 and len(lb) > 0:        
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

    solver = __Check_solverLibrary(solver)

    tic = Tic()

    sla.use_solver(useUmfpack=__canUseUmfpack)
    
    if solver == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solver == "petsc":
        global __pc_default
        # TODO find the best for damage problem
        kspType = 'cg'
        
        if simu.problemType == 'damage':
            if problemType == 'damage':
                pcType = 'ilu'
            else:
                pcType = 'none'
                # ilu decomposition doesn't seem to work for the displacement problem in a damage simulation
                
        else:
            pcType = __pc_default # 'ilu' by default
            # if mesh.dim = 3, errors may occurs if we use ilu
            # works faster on 2D and 3D

        x, option, converg = _PETSc(A, b, x0, kspType, pcType)

        if not converg:
            print(f'\nWarning petsc did not converge with ksp:{kspType} and pc:{pcType} !')
            print(f'Try out with  ksp:{kspType} and pc:none.\n')
            __pc_default = 'none'
            x, option, converg = _PETSc(A, b, x0, kspType, 'none')
            assert converg, 'petsc didnt converge 2 times. check for kspType and pcType'

        solver += option
    
    elif solver == "scipy":
        testSymetric = sla.norm(A-A.transpose())/sla.norm(A)
        A_isSymetric = testSymetric <= 1e-12
        x = _ScipyLinearDirect(A, b, A_isSymetric)
    
    elif solver == "BoundConstrain":
        x = _BoundConstrain(A, b , lb, ub)

    elif solver == "cg":
        x, output = sla.cg(A, b.toarray(), x0, maxiter=None)

    elif solver == "bicg":
        x, output = sla.bicg(A, b.toarray(), x0, maxiter=None)

    elif solver == "gmres":
        x, output = sla.gmres(A, b.toarray(), x0, maxiter=None)

    elif solver == "lgmres":
        x, output = sla.lgmres(A, b.toarray(), x0, maxiter=None)
        print(output)

    elif solver == "umfpack":
        # lu = umfpack.splu(A)
        # x = lu.solve(b).ravel()
        x = umfpackSpsolve(A, b)

    elif solver == "mumps":
        # # TODO dont work yet
        # ctx = DMumpsContext()
        # if ctx.myid == 0:
        #     ctx.set_centralized_sparse(A)
        #     x = b.copy()
        #     ctx.set_rhs(x) # Modified in place
        # ctx.run(job=6) # Analysis + Factorization + Solve
        # ctx.destroy() # Cleanup
        x = mumps.spsolve(A,b)
            
    tic.Tac("Solver",f"Solve {problemType} ({solver})", simu._verbosity)

    # # A x - b = 0
    # residu = np.linalg.norm(A.dot(x)-b.toarray().ravel())
    # print(residu/np.linalg.norm(b.toarray().ravel()))

    return np.array(x)

def __Check_solverLibrary(solver: str) -> str:
    """Checks whether the selected solver library is available
    If not, returns the solver usable in all cases (scipy)."""
    solveurDeBase="scipy"
    if solver == "pypardiso":
        return solver if __canUsePypardiso else solveurDeBase
    elif solver == "umfpack":
        return solver if __canUseUmfpack else solveurDeBase
    elif solver == "mumps":
        return solver if __canUseMumps else solveurDeBase
    elif solver == "petsc":
        return solver if __canUsePetsc else solveurDeBase
    else:
        return solver

def _Solve(simu, problemType: str, resol: ResolType):
    """Solving the problem according to the resolution type"""
    if resol == ResolType.r1:
        return __Solver_1(simu, problemType)
    elif resol == ResolType.r2:
        return __Solver_2(simu, problemType)
    elif resol == ResolType.r3:
        return __Solver_3(simu, problemType)

def __Solver_1(simu, problemType: str) -> np.ndarray:
    # --       --  --  --   --  --
    # | Aii Aic |  | xi |   | bi |    
    # | Aci Acc |  | xc | = | bc | 
    # --       --  --  --   --  --
    # xi = inv(Aii) * (bi - Aic * xc)

    simu = __Cast_Simu(simu)

    # Build the matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r1)

    # Recover dofs
    dofsKnown, dofsUnknown = simu.Bc_dofs_known_unknow(problemType)

    tic = Tic()
    # split of the matrix system into known and unknown dofs
    # Solve : Aii * xi = bi - Aic * xc
    Ai = A[dofsUnknown, :].tocsc()
    Aii = Ai[:, dofsUnknown].tocsr()
    Aic = Ai[:, dofsKnown].tocsr()
    bi = b[dofsUnknown,0]
    xc = x[dofsKnown,0]

    tic.Tac("Solver",f"System-built ({problemType})", simu._verbosity)

    x0 = simu.Get_x0(problemType)
    x0 = x0[dofsUnknown]    

    bDirichlet = Aic @ xc

    lb, ub = simu.Get_lb_ub(problemType)

    xi = _Solve_Axb(simu, problemType, Aii, bi-bDirichlet, x0, lb, ub)

    # apply result to global vector
    x = x.toarray().reshape(x.shape[0])
    x[dofsUnknown] = xi

    return x

def __Solver_2(simu, problemType: str):
    # Lagrange multiplier method

    simu = __Cast_Simu(simu)
    size = simu.mesh.Nn * simu.Get_dof_n(problemType)

    # Build the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r2)
    alpha = A.data.max()

    tic = Tic()

    # set to lil matrix because its faster
    A = A.tolil()
    b = b.tolil()

    dofs_Dirichlet = np.asarray(simu.Bc_dofs_Dirichlet(problemType))
    values_Dirichlet = np.asarray(simu.Bc_values_Dirichlet(problemType))
    
    list_Bc_Lagrange = simu.Bc_Lagrange

    nLagrange = len(list_Bc_Lagrange)
    nDirichlet = len(dofs_Dirichlet)
    nCol = nLagrange + nDirichlet

    x0 = simu.Get_x0(problemType)
    x0 = np.append(x0, np.zeros(nCol))

    linesDirichlet = np.arange(size, size+nDirichlet)
    
    # apply lagrange multiplier
    A[linesDirichlet, dofs_Dirichlet] = alpha
    A[dofs_Dirichlet, linesDirichlet] = alpha
    b[linesDirichlet] = values_Dirichlet * alpha

    tic.Tac("Solver",f"Lagrange ({problemType}) Dirichlet", simu._verbosity)

    # For each lagrange condition we will add a coef to the matrix
    if len(list_Bc_Lagrange) > 0:

        def __apply_lagrange(i: int, lagrangeBc: LagrangeCondition):
            dofs = lagrangeBc.dofs
            values = lagrangeBc.dofsValues * alpha
            coefs = lagrangeBc.lagrangeCoefs * alpha

            A[dofs,i] = coefs
            A[i,dofs] = coefs
            b[i] = values[0]
        
        start = size + nDirichlet
        [__apply_lagrange(i, lagrangeBc) for i, lagrangeBc in enumerate(list_Bc_Lagrange, start)]
    
    tic.Tac("Solver",f"Lagrange ({problemType}) Coupling", simu._verbosity)

    x = _Solve_Axb(simu, problemType, A.tocsr(), b.tocsr(), x0, [], [])

    # We don't send back reaction forces
    sol = x[:size]
    lagrange = x[size:]

    return sol, lagrange

def __Solver_3(simu, problemType: str):
    # Resolution using the penalty method

    # This method does not give preference to dirichlet conditions over neumann conditions.
    # This means that if a dof is applied in Neumann and in Dirichlet, it will be privileged over the dof applied in Neumann.

    # This method is never used. It is just implemented as an example

    simu = __Cast_Simu(simu)

    # Builds the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolType.r3)

    # Solving the penalized matrix system
    x = _Solve_Axb(simu, problemType, A, b, [], [], [])

    return x

def _PETSc(A: sparse.csr_matrix, b: sparse.csr_matrix, x0: np.ndarray, kspType='cg', pcType='ilu') -> np.ndarray:
    """PETSc insterface to solve the linear system A x = b

    Parameters
    ----------
    A : sparse.csr_matrix
        sparse matrix (N, N)
    b : sparse.csr_matrix
        sparse vector (N, 1)
    x0 : np.ndarray
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
    np.ndarray
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

    matrix = PETSc.Mat()
    csr = (A.indptr, A.indices, A.data)
    matrix.createAIJ([dimI, dimJ], comm=__comm, csr=csr)

    vectb = matrix.createVecLeft()

    lines, _, values = sparse.find(b)    

    vectb.array[lines] = values

    x = matrix.createVecRight()
    if len(x0) > 0:
        x.array[:] = x0

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrix)
    ksp.setType(kspType)
    
    pc = ksp.getPC()    
    pc.setType(pcType)

    # pc.setFactorSolverType("superlu") #"mumps"

    ksp.solve(vectb, x)
    x = x.array

    converg = ksp.is_converged

    # PETSc._finalize()

    option = f", {kspType}, {pcType}"

    return x, option, converg
    

def _ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool):
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
    # LU decomposition behind https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html

    hideFacto = False # Hide decomposition
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

def _BoundConstrain(A, b, lb: np.ndarray, ub: np.ndarray):

    assert len(lb) == len(ub), "Must be the same size"
    
    # constrained minimization : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    b = b.toarray().ravel()
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb,ub), tol=tol, method='trf', verbose=0)                    
    x = x['x']

    return x