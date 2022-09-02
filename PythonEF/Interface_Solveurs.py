import numpy as np
import scipy.sparse as sparse
import platform
import TicTac

# Solveurs
from scipy.optimize import lsq_linear
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
    import scikits.umfpack as um
except:
    pass
    # Le module n'est pas utilisable

try:
    from mumps import spsolve
except:
    pass
    # Le module n'est pas utilisable


def Solve_Axb(problemType: str, A: sparse.csr_matrix, b: sparse.csr_matrix,
isDamaged: bool, damage=[],
useCholesky=False, A_isSymetric=False, verbosity=False):
    """Résolution de Ax=b"""
    
    # Detection du système
    syst = platform.system()
    
    tic = TicTac.Tic()

    if problemType == "damage" and len(damage) > 0:
        x = __DamageBoundConstrain(A, b , damage)
        

    if syst == "Darwin":
        method = 2
    else:
        method = 1

    useCholesky = False
    if useCholesky and A_isSymetric:
        x = __Cholesky(A, b)

    elif method == 1:
        x = __Pypardiso_spsolve(A, b)

    elif method == 2:                
        # linear solver scipy : https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems                    
        
        x = __ScipyLinearDirect(A, b, A_isSymetric, isDamaged)


    elif method == 3 and syst == 'Linux':
        # Utilise umfpack
        
        # lu = um.splu(A)
        # x = lu.solve(b).reshape(-1)
        
        x = um.spsolve(A, b)
        

    elif method == 4 and syst == 'Linux':
        # Utilise umfpack depuis scipy
        sla.use_solver(useUmfpack=True)
        x = sla.spsolve(A, b)

    elif method == 5 and syst == 'Linux':
        
        x = spsolve(A,b)
        pass

    elif method == 6 and syst in ['Linux', "Darwin"]:
        x = __PETSc(A, b)
        
            
    tic.Tac(f"Solve {problemType}","Solve Ax=b", verbosity)

    return x

def __Pypardiso_spsolve(A, b):
    b = b.toarray()
    x = pypardiso.spsolve(A, b)

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

def __PETSc(A, b):

    # Utilise PETSc
    # Pour l'instant problème à cause de "Invalid MIT-MAGIC-COOKIE-1 key"
    from petsc4py import PETSc
    ksp = PETSc.KSP().create()
    A = PETSc.Mat(A)
    ksp.setOperators(A)
    
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
    # minim sous contraintes : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
    lb = damage
    lb[np.where(lb>=1)] = 1-np.finfo(float).eps
    ub = np.ones(lb.shape)
    b = b.toarray().reshape(-1)
    # x = lsq_linear(A,b,bounds=(lb,ub), verbose=0,tol=1e-6)                    
    x = lsq_linear(A,b,bounds=(lb,ub), tol=1e-10)                    
    x= x['x']

    return x

