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

def __Cast_Simu(simu):

    import Simulations

    if isinstance(simu, Simulations.Simu_Displacement):
        return simu
    elif isinstance(simu, Simulations.Simu_Damage):
        return simu
    elif isinstance(simu, Simulations.Simu_Beam):
        return simu
    elif isinstance(simu, Simulations.Simu_Thermal):
        return simu
    else:
        raise "Type de simulation inconnue"



def Solveur_1(simu, problemType: str) -> np.ndarray:
    # --       --  --  --   --  --
    # | Aii Aic |  | xi |   | bi |    
    # | Aci Acc |  | xc | = | bc | 
    # --       --  --  --   --  --
    # ui = inv(Aii) * (bi - Aic * xc)

    simu = __Cast_Simu(simu)
    algo = simu.algo

    tic = Tic()

    # Construit le système matricielle
    b = __Construction_b(simu, problemType)
    A, x = __Construction_A_x(simu, problemType, b, resolution=1)

    # Récupère les ddls
    ddl_Connues, ddl_Inconnues = __Construit_ddl_connues_inconnues(simu, problemType)

    # Décomposition du système matricielle en connues et inconnues 
    # Résolution de : Aii * ui = bi - Aic * xc
    Ai = A[ddl_Inconnues, :].tocsc()
    Aii = Ai[:, ddl_Inconnues]
    Aic = Ai[:, ddl_Connues]
    bi = b[ddl_Inconnues,0]
    xc = x[ddl_Connues,0]

    try:
        if problemType == "displacement":
            if algo == "elliptic":
                x0 = simu.displacement[ddl_Inconnues]
            elif algo == "hyperbolic":
                x0 = simu.accel[ddl_Inconnues]
            
        elif problemType == "damage":
            x0 = simu.damage[ddl_Inconnues]

        elif problemType == "thermal":
            x0 = simu.thermal[ddl_Inconnues]
            
        elif problemType == "beam":
            x0 = simu.beamDisplacement[ddl_Inconnues]
    except:
        # proleme pas implémenté ou nouveau maillage
        x0 = []

    bDirichlet = Aic.dot(xc) # Plus rapide
    # bDirichlet = np.einsum('ij,jk->ik', Aic.toarray(), xc.toarray(), optimize='optimal')
    # bDirichlet = sparse.csr_matrix(bDirichlet)

    tic.Tac("Matrices","Construit Ax=b", simu._verbosity)

    if problemType in ["displacement","thermal"]:
        # la matrice est definie symétrique positive on peut donc utiliser cholesky
        useCholesky = True
        A_isSymetric = True
    else:
        #la matrice n'est pas definie symétrique positive
        useCholesky = False
        A_isSymetric = False
    
    if problemType == "damage":
        # Si l'endommagement est supérieur à 1 la matrice A n'est plus symétrique
        isDamaged = True
        solveur = simu.materiau.phaseFieldModel.solveur
        if solveur == "BoundConstrain":
            damage = simu.damage
        else:
            damage = []
    else:
        isDamaged = False
        damage = []

    xi = __Solve_Axb(problemType=problemType, A=Aii, b=bi-bDirichlet, x0=x0, isDamaged=isDamaged, damage=damage, useCholesky=useCholesky, A_isSymetric=A_isSymetric, verbosity=simu._verbosity)

    # Reconstruction de la solution
    x = x.toarray().reshape(x.shape[0])
    x[ddl_Inconnues] = xi

    return x

def Solveur_2(simu, problemType: str):
    # Résolution par la méthode des coefs de lagrange

    simu = __Cast_Simu(simu)

    tic = Tic()

    # Construit le système matricielle pénalisé
    b = __Construction_b(simu, problemType)
    A, x = __Construction_A_x(simu, problemType, b, resolution=2)

    A = A.tolil()
    b = b.tolil()

    ddls_Dirichlet = np.array(simu.Bc_ddls_Dirichlet(problemType))
    values_Dirichlet = np.array(simu.BC_values_Dirichlet(problemType))
    
    list_Bc_Lagrange = simu.Bc_Lagrange
    if len(list_Bc_Lagrange) > 0:
        from Simulations import LagrangeCondition

    nColEnPlusLagrange = len(list_Bc_Lagrange)
    nColEnPlusDirichlet = len(ddls_Dirichlet)
    nColEnPlus = nColEnPlusLagrange + nColEnPlusDirichlet

    decalage = A.shape[0]-nColEnPlus

    listeLignesDirichlet = np.arange(decalage, decalage+nColEnPlusDirichlet)
    
    A[listeLignesDirichlet, ddls_Dirichlet] = 1
    A[ddls_Dirichlet, listeLignesDirichlet] = 1
    b[listeLignesDirichlet] = values_Dirichlet

    # Pour chaque condition de lagrange on va rajouter un coef dans la matrice
    for i, lagrangeBc in enumerate(list_Bc_Lagrange, 1):
        if isinstance(lagrangeBc, LagrangeCondition):
            ddls = lagrangeBc.ddls
            valeurs = lagrangeBc.valeurs_ddls
            coefs = lagrangeBc.lagrangeCoefs

            A[ddls,-i] = coefs
            A[-i,ddls] = coefs

            b[-i] = valeurs[0]

    tic.Tac("Matrices","Construit Ax=b", simu._verbosity)

    x = __Solve_Axb(problemType=problemType, A=A, b=b, x0=None,isDamaged=False, damage=[], useCholesky=False, A_isSymetric=False, verbosity=simu._verbosity)

    # Récupère la solution sans les efforts de réactions
    ddl_Connues, ddl_Inconnues = __Construit_ddl_connues_inconnues(simu, problemType)
    x = x[range(decalage)]

    return x 

def Solveur_3(simu, problemType: str):
    # Résolution par la méthode des pénalisations

    simu = __Cast_Simu(simu)
            
    tic = Tic()

    # Construit le système matricielle pénalisé
    b = __Construction_b(simu, problemType)
    A, x = __Construction_A_x(simu, problemType, b, resolution=3)

    ddl_Connues, ddl_Inconnues = __Construit_ddl_connues_inconnues(simu, problemType)

    tic.Tac("Matrices","Construit Ax=b", simu._verbosity)

    # Résolution du système matricielle pénalisé
    useCholesky=False #la matrice ne sera pas symétrique definie positive
    A_isSymetric=False
    isDamaged = simu.materiau.isDamaged
    if isDamaged:
        solveur = simu.materiau.phaseFieldModel.solveur
        if solveur == "BoundConstrain":
            damage = simu.damage
        else:
            damage = []
    else:
        damage = []

    x = __Solve_Axb(problemType=problemType, A=A, b=b, x0=None,isDamaged=isDamaged, damage=damage, useCholesky=useCholesky, A_isSymetric=A_isSymetric, verbosity=simu._verbosity)

    return x

def __Construction_b(simu, problemType: str):
    """Applique les conditions de Neumann et construit b de Ax=b"""
    
    simu = __Cast_Simu(simu)
    algo = simu.algo
    
    ddls = simu.Bc_ddls_Neumann(problemType)
    valeurs_ddls = simu.Bc_values_Neumann(problemType)

    taille = simu.mesh.Nn
    if problemType == "displacement":
        taille *= simu.dim
    elif problemType == "beam":
        taille *= simu.materiau.beamModel.nbddl_n

    # Dimension supplémentaire lié a l'utilisation des coefs de lagrange
    dimSupl = len(simu.Bc_Lagrange)
    if dimSupl > 0:
        dimSupl += len(simu.Bc_ddls_Dirichlet(problemType))
        
    b = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille+dimSupl,1))
    
    if problemType == "damage" and algo == "elliptic":
        b = b + simu.Fd

    elif problemType == "displacement" and algo == "elliptic":
        b = b + simu.Fu

    elif problemType == "beam" and algo == "elliptic":
        b = b + simu.Fbeam

    elif problemType == "displacement" and algo == "hyperbolic":
        b = b + simu.Fu

        u_n = simu.displacement
        v_n = simu.speed

        Cu = simu.Get_Rayleigh_Damping()
        
        if len(simu.results) == 0 and (b.max() != 0 or b.min() != 0):

            ddl_Connues, ddl_Inconnues = __Construit_ddl_connues_inconnues(simu, problemType)

            bb = b - simu.Ku.dot(sparse.csr_matrix(u_n.reshape(-1, 1)))
            
            bb -= Cu.dot(sparse.csr_matrix(v_n.reshape(-1, 1)))

            bbi = bb[ddl_Inconnues]
            Aii = simu.Mu[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()

            ai_n = __Solve_Axb(problemType=problemType, A=Aii, b=bbi, x0=None, isDamaged=False, damage=[], useCholesky=False, A_isSymetric=True, verbosity=simu._verbosity)

            simu.accel[ddl_Inconnues] = ai_n
        
        a_n = simu.accel

        dt = simu.dt
        gamma = simu.gamma
        betha = simu.betha

        uTild_np1 = u_n + (dt * v_n) + dt**2/2 * (1-2*betha) * a_n
        vTild_np1 = v_n + (1-gamma) * dt * a_n

        # Formulation en accel
        b -= simu.Ku.dot(uTild_np1.reshape(-1,1))
        b -= Cu.dot(vTild_np1.reshape(-1,1))
        b = sparse.csr_matrix(b)


    elif problemType == "thermal" and algo == "elliptic":
        b = b + simu.Ft.copy()

    elif problemType == "thermal" and algo == "parabolic":
        b = b + simu.Ft.copy()

        thermal = simu.thermal
        thermalDot =  simu.thermalDot

        alpha = simu.alpha
        dt = simu.dt

        thermalDotTild_np1 = thermal + (1-alpha) * dt * thermalDot
        thermalDotTild_np1 = sparse.csr_matrix(thermalDotTild_np1.reshape(-1, 1))

        # Resolution de la température
        b = b + simu.Mt.dot(thermalDotTild_np1/(alpha*dt))
            
        # # Résolution de la dérivée temporelle de la température
        # b = b - simu.Kt.dot(thermalDotTild_np1)
            

    else:
        raise "Configuration inconnue"

    return b

def __Construction_A_x(simu, problemType: str, b: sparse.csr_matrix, resolution: int):
    """Applique les conditions de dirichlet en construisant Ax de Ax=b"""

    simu = __Cast_Simu(simu)
    algo = simu.algo

    ddls = simu.Bc_ddls_Dirichlet(problemType)
    valeurs_ddls = simu.BC_values_Dirichlet(problemType)

    taille = simu.mesh.Nn

    if problemType == "damage" and algo == "elliptic":
        A = simu.Kd.copy()

    elif problemType == "displacement" and algo == "elliptic":
        taille *= simu.dim
        A = simu.Ku.copy()

    elif problemType == "beam" and algo == "elliptic":
        taille *= simu.materiau.beamModel.nbddl_n
        A = simu.Kbeam.copy()

    elif problemType == "displacement" and algo == "hyperbolic":
        taille *= simu.dim

        dt = simu.dt
        gamma = simu.gamma
        betha = simu.betha
        
        Cu = simu.Get_Rayleigh_Damping()

        # Forumlation en accel
        A = simu.Mu + (simu.Ku * betha * dt**2)
        A += (gamma * dt * Cu)

        a_n = simu.accel
        valeurs_ddls = a_n[ddls]
                
            
    elif problemType == "thermal" and algo == "elliptic":
        A = simu.Kt.copy()

    elif problemType == "thermal" and algo == "parabolic":
        
        option = 1
        alpha = simu.alpha
        dt = simu.dt

        # Resolution de la température
        A = simu.Kt.copy() + simu.Mt.copy()/(alpha * dt)
            
        # # Résolution de la dérivée temporelle de la température
        # A = simu.Kt.copy() * alpha * dt + simu.Mt.copy()

    else:
        raise "Configuration inconnue"


    if resolution in [1,2]:
        
        # ici on renvoie la solution avec les ddls connues
        x = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1), dtype=np.float64)

        # l,c ,v = sparse.find(x)

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

        # ici on renvoie A pénalisé
        return A.tocsr(), b.tocsr()


def __Construit_ddl_connues_inconnues(simu, problemType: str):
    """Récupère les ddl Connues et Inconnues
    Returns:
        list(int), list(int): ddl_Connues, ddl_Inconnues
    """
    simu = __Cast_Simu(simu)
    

    # Construit les ddls connues
    ddls_Connues = []

    ddls_Connues = simu.Bc_ddls_Dirichlet(problemType)
    unique_ddl_Connues = np.unique(ddls_Connues)

    # Construit les ddls inconnues

    taille = simu.mesh.Nn
    if problemType == "displacement":
        taille *= simu.dim
    elif problemType == "beam":
        taille *= simu.materiau.beamModel.nbddl_n

    ddls_Inconnues = list(range(taille))

    ddls_Inconnues = list(set(ddls_Inconnues) - set(unique_ddl_Connues))
    # [ddls_Inconnues.remove(ddl) for ddl in unique_ddl_Connues]
                            
    ddls_Inconnues = np.array(ddls_Inconnues)
    
    verifTaille = unique_ddl_Connues.shape[0] + ddls_Inconnues.shape[0]
    assert verifTaille == taille, f"Problème dans les conditions ddls_Connues + ddls_Inconnues - taille = {verifTaille-taille}"

    return ddls_Connues, ddls_Inconnues

def __Solve_Axb(problemType: str, A: sparse.csr_matrix, b: sparse.csr_matrix, x0: None, damage: np.ndarray, isDamaged: bool, useCholesky: bool, A_isSymetric: bool, verbosity: bool) -> np.ndarray:
    """Resolution de A x = b

    Parameters
    ----------
    A : sparse.csr_matrix
        Matrice A
    b : sparse.csr_matrix
        vecteur b
    x0 : None
        solution initiale pour les solveurs itératifs
    isDamaged : bool
        
    damage : np.ndarray
        vecteur d'endommagement pour le BoundConstrain
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

    useCholesky = False

    if isDamaged:
        if problemType == "damage" and len(damage) > 0:
            solveur = "BoundConstrain"
            # minimise le residu sous la contrainte
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
    
    tic = Tic()

    if platform.system() == "Linux":
        sla.use_solver(useUmfpack=True)
    else:
        sla.use_solver(useUmfpack=False)
    
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
    tol = 1e-10
    x = optimize.lsq_linear(A, b, bounds=(lb,ub), tol=tol, method='trf', verbose=0)                    
    x = x['x']

    return x




