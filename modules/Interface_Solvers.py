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

# TODO solveur -> Solver to have everything in English

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
    import mumps as mumps
    __canUseMumps = True
except ModuleNotFoundError:
    __canUseMumps = False

try:
    import petsc4py    
    from mpi4py import MPI
    from petsc4py import PETSc
    __canUsePetsc = True
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
    """Usable solvers"""

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
        if len(simu.Bc_dofs_Lagrange(problemType)) > 0:
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
        if simu.problemType == 'damage':
            if problemType == 'damage':
                pcType = 'ilu'
            else:                
                pcType = 'none' # ilu decomposition doesn't seem to work for the displacement problem in a damage simulation
        else:
            if simu.mesh.dim == 3 and simu.mesh.groupElem.order == 1:
                pcType = 'ilu' # fast on displacement problem dont work for HEXA20 or PRISM15
            else:                
                pcType = 'none'
        kspType = 'cg'

        x, option = _PETSc(A, b, x0, kspType, pcType)

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
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r1)

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

    # Builds the penalized matrix system
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r2)
    alpha = A.data.max()

    tic = Tic()

    # set to lil matrix because its faster
    A = A.tolil()
    b = b.tolil()

    ddls_Dirichlet = np.array(simu.Bc_dofs_Dirichlet(problemType))
    values_Dirichlet = np.array(simu.Bc_values_Dirichlet(problemType))
    
    list_Bc_Lagrange = simu.Bc_Lagrange

    nColEnPlusLagrange = len(list_Bc_Lagrange)
    nColEnPlusDirichlet = len(ddls_Dirichlet)
    nColEnPlus = nColEnPlusLagrange + nColEnPlusDirichlet

    decalage = A.shape[0]-nColEnPlus

    x0 = simu.Get_x0(problemType)
    x0 = np.append(x0, np.zeros(nColEnPlus))

    listeLignesDirichlet = np.arange(decalage, decalage+nColEnPlusDirichlet)
    
    # apply lagrange multiplier
    A[listeLignesDirichlet, ddls_Dirichlet] = alpha
    A[ddls_Dirichlet, listeLignesDirichlet] = alpha
    b[listeLignesDirichlet] = values_Dirichlet * alpha

    tic.Tac("Solver",f"Lagrange ({problemType}) Dirichlet", simu._verbosity)

    # Pour chaque condition de lagrange on va rajouter un coef dans la matrice
    if len(list_Bc_Lagrange) > 0:
        from Simulations import LagrangeCondition

        def __apply_lagrange(i: int, lagrangeBc: LagrangeCondition):
            ddls = lagrangeBc.dofs
            valeurs = lagrangeBc.dofsValues
            coefs = lagrangeBc.lagrangeCoefs

            valeurs = np.array(valeurs) * alpha
            coefs = np.array(coefs) * alpha

            A[ddls,-i] = coefs
            A[-i,ddls] = coefs

            b[-i] = valeurs[0]

        [__apply_lagrange(i, lagrangeBc) for i, lagrangeBc in enumerate(list_Bc_Lagrange, 1)]
    
    tic.Tac("Solver",f"Lagrange ({problemType}) Coupling", simu._verbosity)

    x = _Solve_Axb(simu, problemType, A.tocsr(), b.tocsr(), x0, [], [])

    # We don't send back reaction forces
    x = x[range(decalage)]

    return x 

def __Solver_3(simu, problemType: str):
    # Resolution using the penalty method

    simu = __Cast_Simu(simu)

    # Builds the penalized matrix system
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r3)

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
        more -> https://petsc.org/release/manualpages/PC/PCType/\n
        remark : The ilu preconditioner does not seem to work for systems using HEXA20 elements.    

    Returns
    -------
    np.ndarray
        x solution to A x = b
    """

    # comm   = MPI.COMM_WORLD
    comm   = None
    # nprocs = comm.Get_size()
    # rank   = comm.Get_rank()
    # petsc4py.init(sys.argv, comm=MPI.COMM_WORLD)

    dimI = A.shape[0]
    dimJ = A.shape[1]    

    matrice = PETSc.Mat()
    csr = (A.indptr, A.indices, A.data)    
    matrice.createAIJ([dimI, dimJ], comm=comm, csr=csr)

    # Old way
    # lignes, colonnes, valeurs = sparse.find(A)
    # _, count = np.unique(lignes, return_counts = True)
    # nnz = np.array(count, dtype=np.int32)
    # matrice.createAIJ([dimI, dimJ], nnz=nnz, comm=comm, csr=csr)
    # [matrice.setValue(l, c, v) for l, c, v in zip(lignes, colonnes, valeurs)] # ancienne façon pas optimisée avec csr=None
    # matrice.assemble()

    vectb = matrice.createVecLeft()

    lignes, _, valeurs = sparse.find(b)    

    vectb.array[lignes] = valeurs

    x = matrice.createVecRight()
    x.array[:] = x0

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrice)
    ksp.setType(kspType)
    
    pc = ksp.getPC()    
    pc.setType(pcType)

    # pc.setFactorSolverType("superlu") #"mumps"

    ksp.solve(vectb, x)
    x = x.array

    option = f", {kspType}, {pcType}"

    return x, option
    

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