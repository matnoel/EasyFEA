import sys
import platform

import numpy as np
import scipy.sparse as sparse

import TicTac

# Solveurs
import scipy.optimize as optimize
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
if platform.system() != "Darwin":
    import pypardiso
try:
    from sksparse.cholmod import cholesky, cholesky_AAt
except:
    pass
    # Le module n'est pas utilisable

try:
    import scikits.umfpack as umfpack
except:
    pass
    # Le module n'est pas utilisable

try:
    import mumps as mumps 
except:
    pass
    # Le module n'est pas utilisable

try:
    import petsc4py
    
    from petsc4py import PETSc
except:
    # Le module n'est pas utilisable
    pass


def Solve_Axb(problemType: str, A: sparse.csr_matrix, b: sparse.csr_matrix, x0: None, isDamaged: bool, damage: np.ndarray, useCholesky=False, A_isSymetric=False, verbosity=False) -> np.ndarray:
    """Resolution de A x = b

    Parameters
    ----------
    problemType : str
        type de probleme ["displacement", "damage", "thermal]
    A : sparse.csr_matrix
        Matrice A
    b : sparse.csr_matrix
        vecteur b
    x0 : None
        solution initiale pour les solveurs itératifs
    isDamaged : bool
        le problème est endommagé
    damage : np.ndarray
        vecteur d'endommagement pour le BoundConstrain
    useCholesky : bool, optional
        autorise l'utilisation de la décomposition de cholesky, by default False
    A_isSymetric : bool, optional
        A est simetric, by default False
    verbosity : bool, optional
        Les solveurs peuvent ecrire dans la console, by default False

    Returns
    -------
    np.ndarray
        x : solution de A x = b
    """
    
    # Detection du système
    syst = platform.system()
    
    tic = TicTac.Tic()

    useCholesky = False

    if isDamaged:
        if problemType == "damage" and len(damage) > 0:
            solveur = "BoundConstrain" # minimise le residu sous la contrainte
        else:
            if syst == "Darwin":
                solveur = "scipy_spsolve"

            else:
                solveur = "pypardiso"
                # method = "cg" # minimise le residu sans la contrainte
    else:
        if syst == "Darwin":
            solveur = "scipy_spsolve"
            # solveur = "cg"

        elif syst == "Linux":
            solveur = "pypardiso"
            # method = "umfpack" # Plus rapide de ne pas passer par umfpack
            # method = "scipy_spsolve"

        else:
            solveur = "pypardiso"

    
    if useCholesky and A_isSymetric:
        x = __Cholesky(A, b)

    elif solveur == "BoundConstrain":
        x = __DamageBoundConstrain(A, b , damage)

    elif solveur == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solveur == "scipy_spsolve":                
        # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
        x = __ScipyLinearDirect(A, b, A_isSymetric, isDamaged)

    elif solveur == "cg":
        __ActiveUmfpack()
        x, output = sla.cg(A, b.toarray(), x0, maxiter=None)

    elif solveur == "bicg":
        __ActiveUmfpack()
        x, output = sla.bicg(A, b.toarray(), x0, maxiter=None)

    elif solveur == "gmres":
        __ActiveUmfpack()
        x, output = sla.gmres(A, b.toarray(), x0, maxiter=None)

    elif solveur == "lgmres":
        __ActiveUmfpack()
        x, output = sla.lgmres(A, b.toarray(), x0, maxiter=None)
        print(output)

    elif solveur == "umfpack":
        # lu = umfpack.splu(A)
        # x = lu.solve(b).reshape(-1)
        import scikits.umfpack as umfpack
        
        x = umfpack.spsolve(A, b)

    elif solveur == "mumps" and syst == 'Linux':
        x = mumps.spsolve(A,b)

    elif solveur == "petsc" and syst in ['Linux', "Darwin"]:
        x = __PETSc(A, b)
            
    tic.Tac(f"Solve {problemType} ({solveur})","Solve Ax=b", verbosity)

    # # Verification du residu
    # residu = np.linalg.norm(A.dot(x)-b.toarray().reshape(-1))
    # print(residu/np.linalg.norm(b.toarray().reshape(-1)))

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

    petsc4py.init(sys.argv)
    A_petsc = PETSc.Mat().createAIJ([A.shape[0], A.shape[1]])
    A.setup()
    
    
    bb = A.createVecLeft()

    x = A.createVecRight()

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType('bcgs')
    ksp.setConvergenceHistory()
    ksp.getPC().setType('none')
    ksp.solve(b, x)

    ksp.setFromOptions()
    print('Solving with:'), ksp.getType()

    # Solve!
    ksp.solve(b, x)

def __ScipyLinearDirect(A: sparse.csr_matrix, b: sparse.csr_matrix, A_isSymetric: bool, isDamaged: bool):
    # décomposition Lu derrière https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html
    
    __ActiveUmfpack()

    hideFacto = False # Cache la décomposition
    # permc_spec = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"
    if A_isSymetric and not isDamaged:
        permute="MMD_AT_PLUS_A"
    else:
        permute="COLAMD"

    if hideFacto:                   
        x = sla.spsolve(A, b, permc_spec=permute)
        
    else:
        # superlu : https://portal.nersc.gov/project/sparse/superlu/
        # Users' Guide : https://portal.nersc.gov/project/sparse/superlu/ug.pdf
        lu = sla.splu(A.tocsc(), permc_spec=permute)
        x = lu.solve(b.toarray()).reshape(-1)

    return x



def __DamageBoundConstrain(A, b, damage: np.ndarray):
    
    __ActiveUmfpack()
    
    # minim sous contraintes : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
    lb = damage
    lb[np.where(lb>=1)] = 1-np.finfo(float).eps
    ub = np.ones(lb.shape)
    b = b.toarray().reshape(-1)
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb,ub), tol=tol, method='trf', verbose=0)                    
    x = x['x']

    return x

def __ActiveUmfpack():
    if platform.system() == "Linux":
        sla.use_solver(useUmfpack=True)
    else:
        sla.use_solver(useUmfpack=False)


