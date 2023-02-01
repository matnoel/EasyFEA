from enum import Enum
import sys
import platform

import numpy as np
import scipy.sparse as sparse

from TicTac import Tic

# Solveurs
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

try:
    import pypardiso
    canUsePypardiso = True
except ModuleNotFoundError:
    canUsePypardiso = False

try:
    from sksparse.cholmod import cholesky, cholesky_AAt
    canUseCholesky = True
except:
    canUseCholesky = False

try:
    import scikits.umfpack as umfpack
    canUseUmfpack = True
except:
    canUseUmfpack = False

try:
    import mumps as mumps
    canUseMumps = True
except:
    canUseMumps = False

try:
    import petsc4py    
    from petsc4py import PETSc
    canUsePetsc = True
except:
    canUsePetsc = False

class AlgoType(str, Enum):
    elliptic = "elliptic"
    """Résolution K u = F"""
    parabolic = "parabolic"
    """Résolution K u + C v = F"""
    hyperbolic = "hyperbolic"
    """Résolution K u + C v + M a = F"""

class ResolutionType(str, Enum):
    r1 = "1"
    """ui = inv(Aii) * (bi - Aic * xc)"""
    r2 = "2"
    """multiplicateur lagrange"""
    r3 = "3"
    """pénalisation"""

class __SolversLibrary(str, Enum):
    pypardiso = "pypardiso"
    umfpack = "umfpack"
    mumps = "mumps"
    petsc = "petsc"

def __Cast_Simu(simu):

    import Simulations

    if isinstance(simu, Simulations.Simu):
        return simu

def _Solve_Axb(simu, problemType: str, A: sparse.csr_matrix, b: sparse.csr_matrix, x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, useCholesky: bool, A_isSymetric: bool, verbosity: bool) -> np.ndarray:
    """Resolution de A x = b

    Parameters
    ----------
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
    useCholesky : bool, optional
        autorise l'utilisation de la décomposition de cholesky, by default False
    A_isSymetric : bool, optional
        A est simetric, by default False

    Returns
    -------
    np.ndarray
        x : solution de A x = b
    """
    
    # Detection du système
    syst = platform.system()

    from Simulations import Simu, ModelType

    assert isinstance(simu, Simu)
    assert isinstance(problemType, ModelType)

    useCholesky = False

    # Choisie le solveur

    if len(lb) > 0 and len(lb) > 0:        
        solveur = "BoundConstrain"            
    else:
        if syst == "Darwin":            
            if simu.problemType == ModelType.beam:
                solveur = "scipy"
            else:
                # solveur = "cg"
                solveur = "petsc"
                # solveur = "scipy"
        elif syst == "Linux":
            solveur = "pypardiso"
            # method = "umfpack" # Plus rapide de ne pas passer par umfpack
            # method = "scipy"
        else:
            solveur = "pypardiso"
            # method = "cg" # minimise le residu sans la contrainte

    solveur = __Check_solveurLibrary(solveur)
    
    tic = Tic()

    if canUseUmfpack:
        sla.use_solver(useUmfpack=True)
    else:
        sla.use_solver(useUmfpack=False)    
    
    if useCholesky and A_isSymetric and canUseCholesky:
        x = __Cholesky(A, b)

    elif solveur == "BoundConstrain":
        x = __DamageBoundConstrain(A, b , lb, ub)

    elif solveur == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solveur == "scipy":
        x = __ScipyLinearDirect(A, b, A_isSymetric)

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
        x = umfpack.spsolve(A, b)

    elif solveur == "mumps" and syst == 'Linux':
        x = mumps.spsolve(A,b)

    elif solveur == "petsc" and syst in ['Linux', "Darwin"]:
        x, option = __PETSc(A, b)

        solveur += option
            
    tic.Tac("Solveur",f"Solve {problemType} ({solveur})", verbosity)

    # # Verification du residu
    # residu = np.linalg.norm(A.dot(x)-b.toarray().reshape(-1))
    # print(residu/np.linalg.norm(b.toarray().reshape(-1)))

    return np.array(x)

def __Check_solveurLibrary(solveur: str) -> str:
    """Verifie si la librairie du solveur selectionné est disponible\n
    Si c'est pas le cas renvoie le solveur utilisable dans tout les cas (scipy)"""
    solveurDeBase="scipy"
    if solveur == __SolversLibrary.pypardiso:
        return solveur if canUsePypardiso else solveurDeBase
    elif solveur == __SolversLibrary.umfpack:
        return solveur if canUseUmfpack else solveurDeBase
    elif solveur == __SolversLibrary.mumps:
        return solveur if canUseMumps else solveurDeBase
    elif solveur == __SolversLibrary.petsc:
        return solveur if canUsePetsc else solveurDeBase
    else:
        return solveur

def _Solveur(simu, problemType: str, resol: ResolutionType):

    if resol == ResolutionType.r1:
        return __Solveur_1(simu, problemType)
    elif resol == ResolutionType.r2:
        return __Solveur_2(simu, problemType)
    elif resol == ResolutionType.r3:
        return __Solveur_3(simu, problemType)

def __Solveur_1(simu, problemType: str) -> np.ndarray:
    # --       --  --  --   --  --
    # | Aii Aic |  | xi |   | bi |    
    # | Aci Acc |  | xc | = | bc | 
    # --       --  --  --   --  --
    # ui = inv(Aii) * (bi - Aic * xc)

    simu = __Cast_Simu(simu)

    # Construit le système matricielle
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r1)

    # Récupère les ddls
    ddl_Connues, ddl_Inconnues = simu.Bc_ddls_connues_inconnues(problemType)

    tic = Tic()
    # Décomposition du système matricielle en connues et inconnues 
    # Résolution de : Aii * ui = bi - Aic * xc
    Ai = A[ddl_Inconnues, :].tocsc()
    Aii = Ai[:, ddl_Inconnues].tocsr()
    Aic = Ai[:, ddl_Connues].tocsr()
    bi = b[ddl_Inconnues,0]
    xc = x[ddl_Connues,0]

    tic.Tac("Solveur",f"Construit système ({problemType})", simu._verbosity)

    x0 = simu.Get_x0(problemType)
    x0 = x0[ddl_Inconnues]    

    bDirichlet = Aic.dot(xc) # Plus rapide
    # bDirichlet = np.einsum('ij,jk->ik', Aic.toarray(), xc.toarray(), optimize='optimal')
    # bDirichlet = sparse.csr_matrix(bDirichlet)

    useCholesky = simu.useCholesky
    A_isSymetric = simu.A_isSymetric

    lb, ub = simu.Get_lb_ub(problemType)

    xi = _Solve_Axb(simu=simu, problemType=problemType, A=Aii, b=bi-bDirichlet, x0=x0, lb=lb, ub=ub, useCholesky=useCholesky, A_isSymetric=A_isSymetric, verbosity=simu._verbosity)

    # Reconstruction de la solution
    x = x.toarray().reshape(x.shape[0])
    x[ddl_Inconnues] = xi

    return x

def __Solveur_2(simu, problemType: str):
    # Résolution par la méthode des coefs de lagrange

    simu = __Cast_Simu(simu)

    # Construit le système matricielle pénalisé
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r2)    

    tic = Tic()

    A = A.tolil()
    b = b.tolil()

    ddls_Dirichlet = np.array(simu.Bc_ddls_Dirichlet(problemType))
    values_Dirichlet = np.array(simu.BC_values_Dirichlet(problemType))
    
    list_Bc_Lagrange = simu.Bc_Lagrange

    nColEnPlusLagrange = len(list_Bc_Lagrange)
    nColEnPlusDirichlet = len(ddls_Dirichlet)
    nColEnPlus = nColEnPlusLagrange + nColEnPlusDirichlet

    decalage = A.shape[0]-nColEnPlus

    x0 = simu.Get_x0(problemType)
    x0 = np.append(x0, np.zeros(nColEnPlus))

    listeLignesDirichlet = np.arange(decalage, decalage+nColEnPlusDirichlet)
    
    A[listeLignesDirichlet, ddls_Dirichlet] = 1
    A[ddls_Dirichlet, listeLignesDirichlet] = 1
    b[listeLignesDirichlet] = values_Dirichlet

    # Pour chaque condition de lagrange on va rajouter un coef dans la matrice

    if len(list_Bc_Lagrange) > 0:
        from Simulations import LagrangeCondition

        def __apply_lagrange(i: int, lagrangeBc: LagrangeCondition):
            ddls = lagrangeBc.ddls
            valeurs = lagrangeBc.valeurs_ddls
            coefs = lagrangeBc.lagrangeCoefs

            A[ddls,-i] = coefs
            A[-i,ddls] = coefs

            b[-i] = valeurs[0]

        [__apply_lagrange(i, lagrangeBc) for i, lagrangeBc in enumerate(list_Bc_Lagrange, 1)]

    tic.Tac("Solveur",f"Lagrange ({problemType})", simu._verbosity)

    x = _Solve_Axb(simu=simu, problemType=problemType, A=A, b=b, x0=x0, lb=[], ub=[], useCholesky=False, A_isSymetric=False, verbosity=simu._verbosity)
    
    # On renvoie pas les forces de réactions
    x = x[range(decalage)]

    return x 

def __Solveur_3(simu, problemType: str):
    # Résolution par la méthode des pénalisations

    simu = __Cast_Simu(simu)

    # Construit le système matricielle pénalisé
    b = simu._Apply_Neumann(problemType)
    A, x = simu._Apply_Dirichlet(problemType, b, ResolutionType.r3)

    # Résolution du système matricielle pénalisé

    x = _Solve_Axb(simu=simu, problemType=problemType, A=A, b=b, x0=[], lb=[], ub=[], useCholesky=False, A_isSymetric=False, verbosity=simu._verbosity)

    return x

def __Cholesky(A, b):
    
    # Décomposition de cholesky 
    
    # exemple matrice 3x3 : https://www.youtube.com/watch?v=r-P3vkKVutU&t=5s 
    # doc : https://scikit-sparse.readthedocs.io/en/latest/cholmod.html#sksparse.cholmod.analyze
    # Installation : https://www.programmersought.com/article/39168698851/                

    factor = cholesky(A.tocsc())
    # factor = cholesky_AAt(A.tocsc())
    
    # x_chol = factor(b.tocsc())
    x_chol = factor.solve_A(b.tocsc())                

    x = x_chol.toarray().reshape(x_chol.shape[0])

    return x

def __PETSc(A: sparse.csr_matrix, b: sparse.csr_matrix):
    # Utilise PETSc

    # petsc4py.init(sys.argv)

    lignes, colonnes, valeurs = sparse.find(A)

    dimI = A.shape[0]
    dimJ = A.shape[1]
    
    # nb = len(np.where(lignes==299)[0])
    uniqueLignes, count = np.unique(lignes, return_counts = True)
    nnz = np.array(count, dtype=np.int32)
    
    matrice = PETSc.Mat()
    matrice.createAIJ([dimI, dimJ], nnz=nnz)

    # matrice = PETSc.MatSeqAIJ()
    # matrice.createSeqAIJ([dimI, dimJ], nnz=nnz)
    # matrice.create(A)
    
    [matrice.setValue(l, c, v) for l, c, v in zip(lignes, colonnes, valeurs)]    

    matrice.assemble()

    # matrice.isSymmetric()    

    vectb = matrice.createVecLeft()

    lignes, _, valeurs = sparse.find(b)    

    vectb.array[lignes] = valeurs

    x = matrice.createVecRight()

    pc = "lu" # "none", "lu"
    kspType = "cg" # "cg", "bicg, "gmres"

    ksp = PETSc.KSP().create()
    ksp.setOperators(matrice)
    ksp.setType(kspType)
    # ksp.getPC().setType('none')
    ksp.getPC().setType(pc)
    ksp.solve(vectb, x)
    
    # print(f'Solving with PETSc {ksp.getType()}') 

    x = x.array

    option = f", {pc}, {kspType}"

    return x, option
    

def __ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool):
    # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
    # décomposition Lu derrière https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html

    hideFacto = False # Cache la décomposition
    # permc_spec = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"
    # if A_isSymetric and not isDamaged:
    #     permute="MMD_AT_PLUS_A"
    # else:
    #     permute="COLAMD"

    permute="MMD_AT_PLUS_A"

    if hideFacto:                   
        x = sla.spsolve(A, b, permc_spec=permute)
        
    else:
        # superlu : https://portal.nersc.gov/project/sparse/superlu/
        # Users' Guide : https://portal.nersc.gov/project/sparse/superlu/ug.pdf
        lu = sla.splu(A.tocsc(), permc_spec=permute)
        x = lu.solve(b.toarray()).reshape(-1)

    return x

def __DamageBoundConstrain(A, b, lb: np.ndarray, ub: np.ndarray):

    assert len(lb) == len(ub), "Doit être de la même taille"
    
    # minim sous contraintes : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    b = b.toarray().reshape(-1)
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb,ub), tol=tol, method='trf', verbose=0)                    
    x = x['x']

    return x




