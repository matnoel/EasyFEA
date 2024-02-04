"""Interface module with various solvers available on python for solving linear systems of type A x = b.
Interface with PETsC, pypardiso and others ..."""

import sys
from enum import Enum
import numpy as np
import scipy.sparse as sparse

# Solveurs
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from TicTac import Tic

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

class ResolutionType(str, Enum):
    r1 = "1"
    """ui = inv(Aii) * (bi - Aic * xc)"""
    r2 = "2"
    """Lagrange multiplier"""
    r3 = "3"
    """Penalty"""

def Solvers():
    """Available solvers."""

    solvers = ["scipy", "BoundConstrain", "cg", "bicg", "gmres", "lgmres"]
    
    if __canUsePypardiso: solvers.insert(0, "pypardiso")
    if __canUsePetsc: solvers.insert(1, "petsc")
    if __canUseMumps: solvers.insert(2, "mumps")
    if __canUseUmfpack: solvers.insert(3, "umfpack")

    return solvers

def __Cast_Simu(simu):
    """cast the simu as a Simulations.Simu"""

    import Simulations

    if isinstance(simu, Simulations._Simu):
        return simu

def _Solve_Axb(simu, problemType: str, A: sparse.csr_matrix, b: sparse.csr_matrix, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Solve A x = b

    Parameters
    ----------
    simu : Simu
        Simulation
    problemType : ModelType
        Specify the problemType because a simulation can have several physcal models (such as a damage simulation).
    A : sparse.csr_matrix
        Matrice A
    b : sparse.csr_matrix
        vecteur b
    x0 : np.ndarray
        solution initiale pour les solveurs itératifs
    lb : np.ndarray
        lowerBoundary de la solution
    ub : np.ndarray
        upperBoundary de la solution    

    Returns
    -------
    np.ndarray
        x : solution of A x = b
    """

    # check types
    from Simulations import _Simu, ModelType
    assert isinstance(simu, _Simu)
    assert isinstance(problemType, ModelType)
    assert isinstance(A, sparse.csr_matrix)
    assert isinstance(b, sparse.csr_matrix)

    # Choose the solver
    if len(lb) > 0 and len(lb) > 0:        
        solveur = "BoundConstrain"
    else:
        if len(simu.Bc_Lagrange) > 0:
            # if lagrange multiplier are found we cannot use iterative solvers
            solveur = "scipy"
        else:
            solveur = simu.solver

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.spy(A, marker='.')
    testSymetric = sla.norm(A-A.transpose())/sla.norm(A)
    A_isSymetric = testSymetric <= 1e-12
    useCholesky = True if simu.problemType == 'displacement' else False

    solveur = __Check_solveurLibrary(solveur)
    
    tic = Tic()

    sla.use_solver(useUmfpack=__canUseUmfpack)
    
    if solveur == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solveur == "petsc":
        global __pc_default
        # TODO find the best for damage problem
        if simu.problemType == 'damage':
            if problemType == 'damage':
                pcType = 'ilu'
            else:                
                # ilu decomposition doesn't seem to work for the displacement problem in a damage simulation
                # pcType = 'none'
                pcType = 'ilu'
        else:
            pcType = __pc_default # 'ilu' by default
            # if mesh.dim = 3, errors may occurs if we use ilu
            # works faster on 2D and 3D
        kspType = 'cg'
        # kspType = 'bicg'

        x, option, converg = _PETSc(A, b, x0, kspType, pcType)

        if not converg:
            print(f'\nWarning petsc did not converge with ksp:{kspType} and pc:{pcType} !')
            print(f'Try out with  ksp:{kspType} and pc:none.\n')
            __pc_default = 'none'
            x, option, converg = _PETSc(A, b, x0, kspType, 'none')
            assert converg, 'petsc didnt converge 2 times. check for kspType and pcType'

        solveur += option
    
    elif solveur == "scipy":
        x = _ScipyLinearDirect(A, b, A_isSymetric)
    
    elif solveur == "BoundConstrain":
        x = _BoundConstrain(A, b , lb, ub)

    elif solveur == "cg":
        x, output = sla.cg(A, b.toarray(), x0, maxiter=None)

    elif solveur == "bicg":
        x, output = sla.bicg(A, b.toarray(), x0, maxiter=None)

    elif solveur == "gmres":
        x, output = sla.gmres(A, b.toarray(), x0, maxiter=None)

    elif solveur == "lgmres":
        x, output = sla.lgmres(A, b.toarray(), x0, maxiter=None)
        print(output)

    elif solveur == "umfpack":
        # lu = umfpack.splu(A)
        # x = lu.solve(b).reshape(-1)
        x = umfpackSpsolve(A, b)

    elif solveur == "mumps":
        # # TODO dont work yet
        # ctx = DMumpsContext()
        # if ctx.myid == 0:
        #     ctx.set_centralized_sparse(A)
        #     x = b.copy()
        #     ctx.set_rhs(x) # Modified in place
        # ctx.run(job=6) # Analysis + Factorization + Solve
        # ctx.destroy() # Cleanup
        x = mumps.spsolve(A,b)
            
    tic.Tac("Solver",f"Solve {problemType} ({solveur})", simu._verbosity)

    # # A x - b = 0
    # residu = np.linalg.norm(A.dot(x)-b.toarray().reshape(-1))
    # print(residu/np.linalg.norm(b.toarray().reshape(-1)))

    return np.array(x)

def __Check_solveurLibrary(solveur: str) -> str:
    """Checks whether the selected solver library is available
    If not, returns the solver usable in all cases (scipy)."""
    solveurDeBase="scipy"
    if solveur == "pypardiso":
        return solveur if __canUsePypardiso else solveurDeBase
    elif solveur == "umfpack":
        return solveur if __canUseUmfpack else solveurDeBase
    elif solveur == "mumps":
        return solveur if __canUseMumps else solveurDeBase
    elif solveur == "petsc":
        return solveur if __canUsePetsc else solveurDeBase
    else:
        return solveur

def Solve(simu, problemType: str, resol: ResolutionType):
    """Solving the problem according to the resolution type"""
    if resol == ResolutionType.r1:
        return __Solver_1(simu, problemType)
    elif resol == ResolutionType.r2:
        return __Solver_2(simu, problemType)
    elif resol == ResolutionType.r3:
        return __Solver_3(simu, problemType)

def __Solver_1(simu, problemType: str) -> np.ndarray:
    # --       --  --  --   --  --
    # | Aii Aic |  | xi |   | bi |    
    # | Aci Acc |  | xc | = | bc | 
    # --       --  --  --   --  --
    # ui = inv(Aii) * (bi - Aic * xc)

    simu = __Cast_Simu(simu)

    # Builds the matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolutionType.r1)

    # Recovers ddls
    dofsKnown, dofsUnknown = simu.Bc_dofs_known_unknow(problemType)

    tic = Tic()
    # Decomposition of the matrix system into known and unknowns
    # Solve : Aii * ui = bi - Aic * xc
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
    # Solving with the Lagrange multiplier method

    simu = __Cast_Simu(simu)
    size = simu.mesh.Nn * simu.Get_dof_n(problemType)

    # Builds the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolutionType.r2)
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

    # Pour chaque condition de lagrange on va rajouter un coef dans la matrice
    if len(list_Bc_Lagrange) > 0:
        from Simulations import LagrangeCondition

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

    # This method does not give preference to dirichlet conditions over neumann conditions. This means that if a dof is applied in Neumann and in Dirichlet, it will be privileged over the dof applied in Neumann.

    # Normally, this method is never used. It is just implemented as an example

    simu = __Cast_Simu(simu)

    # Builds the penalized matrix system
    b = simu._Solver_Apply_Neumann(problemType)
    A, x = simu._Solver_Apply_Dirichlet(problemType, b, ResolutionType.r3)

    # Solving the penalized matrix system
    x = _Solve_Axb(simu, problemType, A, b, [], [], [])

    return x

def _PETSc(A: sparse.csr_matrix, b: sparse.csr_matrix, x0: np.ndarray, kspType='cg', pcType='ilu') -> np.ndarray:
    """PETSc insterface to solve A x = b

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
    
    __comm = None
    petsc4py.init(sys.argv, comm=__comm)

    dimI = A.shape[0]
    dimJ = A.shape[1]

    matrice = PETSc.Mat()
    csr = (A.indptr, A.indices, A.data)    
    matrice.createAIJ([dimI, dimJ], comm=__comm, csr=csr)

    vectb = matrice.createVecLeft()

    lines, _, values = sparse.find(b)    

    vectb.array[lines] = values

    x = matrice.createVecRight()
    if len(x0) > 0:
        x.array[:] = x0

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrice)
    ksp.setType(kspType)
    
    pc = ksp.getPC()    
    pc.setType(pcType)

    # pc.setFactorSolverType("superlu") #"mumps"

    ksp.solve(vectb, x)
    x = x.array

    converg = ksp.converged and not ksp.diverged    

    # PETSc._finalize()

    option = f", {kspType}, {pcType}"

    return x, option, converg
    

def _ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool):
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
    # décomposition Lu derrière https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html

    hideFacto = False # Cache la décomposition
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
        x = lu.solve(b.toarray()).reshape(-1)

    return x

def _BoundConstrain(A, b, lb: np.ndarray, ub: np.ndarray):

    assert len(lb) == len(ub), "Doit être de la même taille"
    
    # minim sous contraintes : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    b = b.toarray().reshape(-1)
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb,ub), tol=tol, method='trf', verbose=0)                    
    x = x['x']

    return x