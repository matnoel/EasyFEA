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
if platform.system() != "Darwin":
    import pypardiso

try:
    from sksparse.cholmod import cholesky, cholesky_AAt
except:
    # Le module n'est pas utilisable
    pass

try:
    import scikits.umfpack as umfpack
except:
    # Le module n'est pas utilisable
    pass

try:
    import mumps as mumps 
except:
    # Le module n'est pas utilisable
    pass

try:
    import petsc4py
    
    from petsc4py import PETSc
except:
    # Le module n'est pas utilisable
    pass

class AlgoType(str, Enum):
    elliptic = "elliptic"
    parabolic = "parabolic"
    hyperbolic = "hyperbolic"

class ResolutionType(str, Enum):
    r1 = "1"
    """ui = inv(Aii) * (bi - Aic * xc)"""
    r2 = "2"
    """multiplicateur lagrange"""
    r3 = "3"
    """pénalisation"""

def __Cast_Simu(simu):

    import Simulations

    if isinstance(simu, Simulations._Simu):
        return simu

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
    b = simu.Apply_Neumann(problemType)
    A, x = simu.Apply_Dirichlet(problemType, b, ResolutionType.r1)

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

    tic.Tac("Construit Ax=b",f"Construit système ({problemType})", simu._verbosity)

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
    b = simu.Apply_Neumann(problemType)
    A, x = simu.Apply_Dirichlet(problemType, b, ResolutionType.r2)

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

    tic.Tac("Construit Ax=b",f"Lagrange ({problemType})", simu._verbosity)

    x = _Solve_Axb(simu=simu, problemType=problemType, A=A, b=b, x0=[], lb=[], ub=[], useCholesky=False, A_isSymetric=False, verbosity=simu._verbosity)
    
    # On renvoie pas les forces de réactions
    x = x[range(decalage)]

    return x 

def __Solveur_3(simu, problemType: str):
    # Résolution par la méthode des pénalisations

    simu = __Cast_Simu(simu)

    # Construit le système matricielle pénalisé
    b = simu.Apply_Neumann(problemType)
    A, x = simu.Apply_Dirichlet(problemType, b, ResolutionType.r3)

    # Résolution du système matricielle pénalisé

    x = _Solve_Axb(simu=simu, problemType=problemType, A=A, b=b, x0=[], lb=[], ub=[], useCholesky=False, A_isSymetric=False, verbosity=simu._verbosity)

    return x

def Apply_Neumann(self, problemType: str):
    """Applique les conditions de Neumann et construit b de Ax=b"""
    
    tic = Tic()

    self = __Cast_Simu(self)
    algo = self.algo
    
    ddls = self.Bc_ddls_Neumann(problemType)
    valeurs_ddls = self.Bc_values_Neumann(problemType)

    taille = self.mesh.Nn
    if problemType == "displacement":
        taille *= self.dim
    elif problemType == "beam":
        taille *= self.materiau.beamModel.nbddl_n

    # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
    dimSupl = len(self.Bc_Lagrange)
    if dimSupl > 0:
        dimSupl += len(self.Bc_ddls_Dirichlet(problemType))
        
    b = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille+dimSupl,1))
    
    if problemType == "damage" and algo == AlgoType.elliptic:
        b = b + self.Fd

    elif problemType == "displacement" and algo == AlgoType.elliptic:
        b = b + self.Fu

    elif problemType == "beam" and algo == AlgoType.elliptic:
        b = b + self.Fbeam

    elif problemType == "displacement" and algo == AlgoType.hyperbolic:
        b = b + self.Fu

        u_n = self.displacement
        v_n = self.speed

        Cu = self.Get_Rayleigh_Damping()
        
        if len(self.results) == 0 and (b.max() != 0 or b.min() != 0):

            ddl_Connues, ddl_Inconnues = __Construit_ddl_connues_inconnues(self, problemType)

            bb = b - self.Ku.dot(sparse.csr_matrix(u_n.reshape(-1, 1)))
            
            bb -= Cu.dot(sparse.csr_matrix(v_n.reshape(-1, 1)))

            bbi = bb[ddl_Inconnues]
            Aii = self.Mu[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()

            ai_n = _Solve_Axb(problemType=problemType, A=Aii, b=bbi, x0=None, isDamaged=False, damage=[], useCholesky=False, A_isSymetric=True, verbosity=self._verbosity)

            self.accel[ddl_Inconnues] = ai_n
        
        a_n = self.accel

        dt = self.dt
        gamma = self.gamma
        betha = self.betha

        uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1-2*betha) * a_n
        vTild_np1 = v_n + (1-gamma) * dt * a_n

        # Formulation en accel
        b -= self.Ku.dot(uTild_np1.reshape(-1,1))
        b -= Cu.dot(vTild_np1.reshape(-1,1))
        b = sparse.csr_matrix(b)


    elif problemType == "thermal" and algo == AlgoType.elliptic:
        b = b + self.Ft.copy()

    elif problemType == "thermal" and algo == AlgoType.parabolic:
        b = b + self.Ft.copy()

        thermal = self.thermal
        thermalDot =  self.thermalDot

        alpha = self.alpha
        dt = self.dt

        thermalDotTild_np1 = thermal + (1-alpha) * dt * thermalDot
        thermalDotTild_np1 = sparse.csr_matrix(thermalDotTild_np1.reshape(-1, 1))

        # Resolution de la température
        b = b + self.Mt.dot(thermalDotTild_np1/(alpha*dt))
            
        # # Résolution de la dérivée temporelle de la température
        # b = b - simu.Kt.dot(thermalDotTild_np1)
            

    else:
        raise "Configuration inconnue"

    tic.Tac("Construit Ax=b",f"Neumann ({problemType})", self._verbosity)

    return b

def Apply_Dirichlet(self, problemType: str, b: sparse.csr_matrix, resolution: ResolutionType):
    

    tic = Tic()

    self = __Cast_Simu(self)
    algo = self.algo

    ddls = self.Bc_ddls_Dirichlet(problemType)
    valeurs_ddls = self.BC_values_Dirichlet(problemType)

    taille = self.mesh.Nn

    if problemType == "damage" and algo == AlgoType.elliptic:
        A = self.Kd

    elif problemType == "displacement" and algo == AlgoType.elliptic:
        taille *= self.dim
        A = self.Ku

    elif problemType == "beam" and algo == AlgoType.elliptic:
        taille *= self.materiau.beamModel.nbddl_n
        A = self.Kbeam

    elif problemType == "displacement" and algo == AlgoType.hyperbolic:
        taille *= self.dim

        dt = self.dt
        gamma = self.gamma
        betha = self.betha
        
        Cu = self.Get_Rayleigh_Damping()

        # Forumlation en accel
        A = self.Mu + (self.Ku * betha * dt**2)
        A += (gamma * dt * Cu)

        a_n = self.accel
        valeurs_ddls = a_n[ddls]
                
            
    elif problemType == "thermal" and algo == AlgoType.elliptic:
        A = self.Kt

    elif problemType == "thermal" and algo == AlgoType.parabolic:        
        
        alpha = self.alpha
        dt = self.dt

        # Resolution de la température
        A = self.Kt + self.Mt/(alpha * dt)
            
        # # Résolution de la dérivée temporelle de la température
        # A = simu.Kt.copy() * alpha * dt + simu.Mt.copy()

    else:
        raise "Configuration inconnue"


    if resolution in [1,2]:
        
        # ici on renvoie la solution avec les ddls connues
        x = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1), dtype=np.float64)

        # l,c ,v = sparse.find(x)

        tic.Tac("Construit Ax=b",f"Dirichlet ({problemType})", self._verbosity)

        return A, x

    elif resolution == 3:
        # Pénalisation

        A = A.tolil()
        b = b.tolil()            
        
        # Pénalisation A
        A[ddls] = 0.0
        A[ddls, ddls] = 1

        # Pénalisation b
        b[ddls] = valeurs_ddls

        tic.Tac("Construit Ax=b",f"Dirichlet ({problemType})", self._verbosity)

        # ici on renvoie A pénalisé
        return A.tocsr(), b.tocsr()

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

    from Simulations import _Simu, ModelType

    assert isinstance(simu, _Simu)
    assert isinstance(problemType, ModelType)

    useCholesky = False

    # Choisie le solveur

    if len(lb) > 0 and len(lb) > 0:        
        solveur = "BoundConstrain"            
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
            # method = "cg" # minimise le residu sans la contrainte
    
    tic = Tic()

    if platform.system() == "Linux":
        sla.use_solver(useUmfpack=True)
    else:
        sla.use_solver(useUmfpack=False)
    
    if useCholesky and A_isSymetric:
        x = __Cholesky(A, b)

    elif solveur == "BoundConstrain":
        x = __DamageBoundConstrain(A, b , lb, ub)

    elif solveur == "pypardiso":
        x = pypardiso.spsolve(A, b.toarray())

    elif solveur == "scipy_spsolve":
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
        x = __PETSc(A, b)
            
    tic.Tac(f"Solve {problemType} ({solveur})","Solve Ax=b", verbosity)

    # # Verification du residu
    # residu = np.linalg.norm(A.dot(x)-b.toarray().reshape(-1))
    # print(residu/np.linalg.norm(b.toarray().reshape(-1)))

    return np.array(x)

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




