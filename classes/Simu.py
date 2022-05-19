
from types import LambdaType
from typing import cast

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


from GroupElem import GroupElem
import Affichage
from Mesh import Mesh
from BoundaryCondition import BoundaryCondition
from Materiau import Elas_Isot, Materiau, PhaseFieldModel
from TicTac import TicTac
from Interface_Gmsh import Interface_Gmsh
import Dossier
    
class Simu:

    @staticmethod
    def problemTypes():
        return ["displacement","damage"]
    
    def problemDirections(self, probemType:str, directions:list):
        """Teste si les dimensions sont correctes"""
        assert probemType in Simu.problemTypes()

        match probemType:
            case "damage":
                assert directions == ["d"]
            case "displacement":
                for d in directions:
                    assert d in ["x","y","z"]
                    if self.__dim == 2: assert d != "z", "Lors d'une simulation 2d on ne peut appliquer ques des conditions suivant x et y"
    
    def __get_listElement(self):        
        return list(range(self.mesh.Ne))        
    listElement = property(__get_listElement)

    def __get_mesh(self):
        return self.__mesh
    mesh = cast(Mesh, property(__get_mesh))

# ------------------------------------------- CONSTRUCTEUR ------------------------------------------- 

    def __init__(self, mesh: Mesh, materiau: Materiau, verbosity=True):
        """Creation d'une simulation

        Args:
            dim (int): Dimension de la simulation (2D ou 3D)
            mesh (Mesh): Maillage que la simulation va utiliser
            materiau (Materiau): Materiau utilisé
            verbosity (bool, optional): La simulation ecrira dans la console. Defaults to True.
        """
        dim = mesh.dim

        # Vérification des valeurs
        assert dim == 2 or dim == 3, "Dimesion compris entre 2D et 3D"
        assert materiau.dim == dim, "Doit avoir la meme dimension que dim"

        self.__dim = mesh.dim
        """dimension de la simulation 2D ou 3D"""
        self.__verbosity = verbosity
        """la simulation peut ecrire dans la console"""

        self.__mesh = mesh
        """maillage de la simulation"""
        self.materiau = materiau
        """materiau de la simulation"""
        self.__resultats = {}
        """résultats de la simulation"""
        self.__psiP_e_pg = []
        """densité d'energie elastique en tension PsiPlus(e, pg, 1)"""

        # Conditions Limites
        self.__Bc_Neuman = []
        """Conditions de Neumann list(BoundaryCondition)"""
        self.__Bc_Dirichlet = []
        """Conditions de Dirichlet list(BoundaryCondition)"""
        
    
# ------------------------------------------- PROBLEME EN DEPLACEMENT ------------------------------------------- 

    def ConstruitMatElem_Dep(self):
        """Construit pour chaque elements les matrices elementaires de rigidité en déplacement"""
        return self.__ConstruitMatElem_Dep()

    def __ConstruitMatElem_Dep(self):
        """Construit les matrices de rigidités élementaires pour le problème en déplacement

        Args:
            d (list(float)): Endommagement aux noeuds

        Returns:
            array de dim e: les matrices elementaires pour chaque element
        """
        
        isDamaged = self.materiau.isDamaged

        tic = TicTac()

        matriceType="rigi"

        # Data
        mesh = self.__mesh
        nPg = mesh.get_nPg(matriceType)
        
        # Recupère les matrices pour travailler
        jacobien_e_pg = mesh.get_jacobien_e_pg(matriceType)
        poid_pg = mesh.get_poid_pg(matriceType)
        B_dep_e_pg = mesh.get_B_dep_e_pg(matriceType)

        comportement = self.materiau.comportement

        if not comportement.useVoigtNotation:
            B_dep_e_pg = comportement.AppliqueCoefSurBrigi(B_dep_e_pg)

        mat = comportement.get_C()
        # Ici on le materiau est homogène
        # Il est possible de stpcker ça pour ne plus avoir à recalculer        

        if isDamaged:   # probleme endomagement

            d = self.GetResultat("damage", valeursAuxNoeuds=True)

            u = self.GetResultat("displacement", valeursAuxNoeuds=True)

            
            u = self.__resultats["displacement"]

            phaseFieldModel = self.materiau.phaseFieldModel
                
            # Calcul la deformation nécessaire pour le split
            Epsilon_e_pg = self.__CalculEpsilon_e_pg(u, matriceType)

            # Split de la loi de comportement
            cP_e_pg, cM_e_pg = phaseFieldModel.Calc_C(Epsilon_e_pg)

            # Endommage : c = g(d) * cP + cM
            g_e_pg = phaseFieldModel.get_g_e_pg(d, mesh, matriceType)
            cP_e_pg = np.einsum('ep,epij->epij', g_e_pg, cP_e_pg, optimize=True)

            c = cP_e_pg+cM_e_pg
            
            # Matrice de rigidité élementaire
            Ku_e_pg = np.einsum('ep,p,epki,epkl,eplj->epij', jacobien_e_pg, poid_pg, B_dep_e_pg, c, B_dep_e_pg, optimize=True)
            
        else:   # probleme en déplacement simple

            Ku_e_pg = np.einsum('ep,p,epki,kl,eplj->epij', jacobien_e_pg, poid_pg, B_dep_e_pg, mat, B_dep_e_pg, optimize=True)
        
        # On somme sur les points d'intégrations
        Ku_e = np.sum(Ku_e_pg, axis=1)

        if self.__dim == 2:
            Ku_e = Ku_e * self.materiau.comportement.epaisseur
        
        tic.Tac("Matrices","Calcul des matrices elementaires (déplacement)", self.__verbosity)

        return Ku_e    
 
    def Assemblage_u(self):
        """Construit K global pour le problème en deplacement

        Args:            
            d (np.ndarray, optional): Endommagement à appliquer au matériau. Defaults to [].
            verification (bool, optional): Verification de l'assemblage avec l'ancienne méthode bcp bcp bcp moin rapide. Defaults to False.
        """

        # Data
        mesh = self.__mesh        
        taille = mesh.Nn*self.__dim

        # Construit Ke
        Ku_e = self.__ConstruitMatElem_Dep()

        # Prépare assemblage
        lignesVector_e = mesh.lignesVector_e
        colonnesVector_e = mesh.colonnesVector_e
        
        tic = TicTac()

        # Assemblage
        self.__Ku = sparse.csr_matrix((Ku_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
        """Matrice Kglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

        # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
        self.__Fu = sparse.csr_matrix((taille, 1))
        """Vecteur Fglob pour le problème en déplacement (Nn*dim, 1)"""

        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrices","Assemblage du systême en déplacement", self.__verbosity)
        
        return self.__Ku

    def Solve_u(self, resolution=2, useCholesky=False):
        """Resolution du probleme de déplacement"""

        tic = TicTac()

        Uglob = self.__Solveur(problemType="displacement", resolution=resolution, useCholesky=useCholesky, A_isSymetric=True)

        tic.Tac("Résolution deplacement","Résolution {} pour le problème de déplacement".format(resolution) , self.__verbosity)
        
        self.__Save_u(Uglob)

        return Uglob

    
# ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def CalcPsiPlus_e_pg(self):
            # Pour chaque point de gauss de tout les elements du maillage on va calculer psi+
            # Calcul de la densité denergie de deformation en traction

            assert self.materiau.isDamaged, "pas de modèle d'endommagement"

            phaseFieldModel = self.materiau.phaseFieldModel
            
            u = self.GetResultat("displacement")
            d = self.GetResultat("damage")

            testu = isinstance(u, np.ndarray) and (u.shape[0] == self.__mesh.Nn*self.__dim )
            testd = isinstance(d, np.ndarray) and (d.shape[0] == self.__mesh.Nn )

            assert testu or testd,"Il faut initialiser uglob et damage correctement"

            Epsilon_e_pg = self.__CalculEpsilon_e_pg(u, "masse")
            # ici le therme masse est important sinon on sous intègre

            # Calcul l'energie
            old_psiP = self.__psiP_e_pg

            nPg = self.__mesh.get_nPg("masse")

            if len(old_psiP) == 0:
                # Pas encore d'endommagement disponible
                old_psiP = np.zeros((self.__mesh.Ne, nPg))

            psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)
            
            inc_H = psiP_e_pg - old_psiP

            elements, pdGs = np.where(inc_H < 0)

            psiP_e_pg[elements, pdGs] = old_psiP[elements,pdGs]           

            new = np.linalg.norm(psiP_e_pg)
            old = np.linalg.norm(self.__psiP_e_pg)

            assert new >= old, "Erreur"
            self.__psiP_e_pg = psiP_e_pg

            return self.__psiP_e_pg
    
    def __ConstruitMatElem_Pfm(self):
        
        tic = TicTac()

        phaseFieldModel = self.materiau.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.CalcPsiPlus_e_pg()
        r_e_pg = phaseFieldModel.get_r_e_pg(PsiP_e_pg)
        f_e_pg = phaseFieldModel.get_f_e_pg(PsiP_e_pg)

        matriceType="masse"

        # Recupère les matrices pour travailler
        mesh = self.__mesh
        jacobien_e_pg = mesh.get_jacobien_e_pg(matriceType)
        poid_pg = mesh.get_poid_pg(matriceType)
        Nd_pg = mesh.get_N_scalaire_pg(matriceType)
        Bd_e_pg = mesh.get_B_sclaire_e_pg(matriceType)

        # Partie qui fait intervenir le therme de reaction r
        Kdr_e_pg = np.einsum('ep,p,ep,pki,pkj->epij', jacobien_e_pg, poid_pg, r_e_pg, Nd_pg, Nd_pg, optimize=True)

        # Partie qui fait intervenir le therme de diffusion K
        KdK_e_pg = np.einsum('ep,p,,epki,epkj->epij', jacobien_e_pg, poid_pg, k, Bd_e_pg, Bd_e_pg, optimize=True)

        Kd_e_pg = Kdr_e_pg+KdK_e_pg
        
        Kd_e = np.sum(Kd_e_pg, axis=1)

        # Construit Fd_e
        Fd_e_pg = np.einsum('ep,p,ep,pji->epij', jacobien_e_pg, poid_pg, f_e_pg, Nd_pg) 

        Fd_e = np.sum(Fd_e_pg, axis=1)

        tic.Tac("Matrices","Calcul des matrices elementaires (endommagement)", self.__verbosity)

        return Kd_e, Fd_e

    def Assemblage_d(self):
        """Construit Kglobal pour le probleme d'endommagement
        """
       
        # Data
        mesh = self.__mesh
        taille = mesh.Nn
        lignesScalar_e = mesh.lignesScalar_e
        colonnesScalar_e = mesh.colonnesScalar_e
        
        # Calul les matrices elementaires
        Kd_e, Fd_e = self.__ConstruitMatElem_Pfm()

        # Assemblage
        tic = TicTac()        

        self.__Kd = sparse.csr_matrix((Kd_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
        """Kglob pour le probleme d'endommagement (Nn, Nn)"""
        
        lignes = mesh.connect.reshape(-1)
        self.__Fd = sparse.csr_matrix((Fd_e.reshape(-1), (lignes,np.zeros(len(lignes)))), shape = (taille,1))
        """Fglob pour le probleme d'endommagement (Nn, 1)"""        

        tic.Tac("Matrices","Assemblage du systême en endommagement", self.__verbosity)       

        return self.__Kd, self.__Fd
    
    def Solve_d(self, resolution=2):
        """Resolution du problème d'endommagement"""
         
        tic = TicTac()

        dGlob = self.__Solveur(problemType="damage", resolution=resolution, useCholesky=False, A_isSymetric=False)

        tic.Tac("Résolution endommagement",f"Résolution {resolution} pour le problème de endommagement" , self.__verbosity)
        
        # assert dGlob.max() <= 1, "Doit etre inférieur a 1"
        # assert dGlob.min() >= 0, "Doit etre supérieur 0"

        # if(dGlob.max() > 1):
        #     print("dmax = {}".format(dGlob.max()))

        # if(dGlob.min() < 0):
        #     print("dmin = {}".format(dGlob.min()))

        self.__Save_d(dGlob)

        return dGlob

# ------------------------------------------------- SOLVEUR -------------------------------------------------

    def __Construit_ddl_connues_inconnues(self, problemType: str):
        """Récupère les ddl Connues et Inconnues

        Args:
            vector (bool, optional): Travail sur un vecteur ou un scalaire. Defaults to True.

        Returns:
            list(int), list(int): ddl_Connues, ddl_Inconnues
        """
        
        assert problemType in Simu.problemTypes()

        # Construit les ddls connues
        ddls_Connues = []

        for bcDirichlet in self.__Bc_Dirichlet:
            bcDirichlet = cast(BoundaryCondition, bcDirichlet)
            if bcDirichlet.problemType == problemType:
                ddls_Connues.extend(bcDirichlet.ddls)
        
        ddls_Connues = np.array(ddls_Connues)

        # Construit les ddls inconnues

        match problemType:
            case "damage":
                taille = self.__mesh.Nn
            case "displacement":
                taille = self.__mesh.Nn*self.__dim

        ddls_Inconnues = list(np.arange(taille))
        for ddlConnue in ddls_Connues: ddls_Inconnues.remove(ddlConnue)
        ddls_Inconnues = np.array(ddls_Inconnues)

        assert ddls_Connues.shape[0] + ddls_Inconnues.shape[0] == taille, "Problème dans les conditions"

        return ddls_Connues, ddls_Inconnues

    def __Application_Conditions_Neuman(self, problemType: str):
        """applique les conditions de Neumann"""

        assert problemType in Simu.problemTypes()

        ddls = []
        valeurs_ddls = []
        for bcNeumann in self.__Bc_Neuman:
            bcNeumann = cast(BoundaryCondition, bcNeumann)
            if bcNeumann.problemType == problemType:
                ddls.extend(bcNeumann.ddls)
                valeurs_ddls.extend(bcNeumann.valeurs_ddls)

        taille = self.__mesh.Nn

        if problemType == "displacement":
            taille = self.__mesh.Nn*self.__dim

        b = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1))

        # l,c ,v = sparse.find(b)

        match problemType:
            case "damage":
                b = b + self.__Fd.copy()                
            case "displacement":
                b = b + self.__Fu.copy()

        return b

    def __Application_Conditions_Dirichlet(self, problemType: str, b, resolution):
        """applique les conditions de dirichlet"""

        assert problemType in Simu.problemTypes()

        ddls = []
        valeurs_ddls = []
        for bcDirichlet in self.__Bc_Dirichlet:
            bcDirichlet = cast(BoundaryCondition, bcDirichlet)
            if bcDirichlet.problemType == problemType:
                ddls.extend(bcDirichlet.ddls)
                valeurs_ddls.extend(bcDirichlet.valeurs_ddls)

        taille = self.__mesh.Nn

        match problemType:
            case "damage":
                A = self.__Kd.copy()
            case "displacement":
                taille = taille*self.__dim
                A = self.__Ku.copy()

        if resolution == 1:
            
            A = A.tolil()
            b = b.tolil()            
            
            # Pénalisation A
            A[ddls] = 0.0
            A[ddls, ddls] = 1

            # Pénalisation b
            b[ddls] = valeurs_ddls

            # ici on renvoie A pénalisé
            return A.tocsr(), b.tocsr()

        else:
            
            # ici on renvoir la solution avec les ddls connues
            x = sparse.csr_matrix((valeurs_ddls, (ddls,  np.zeros(len(ddls)))), shape = (taille,1), dtype=np.float64)

            # l,c ,v = sparse.find(x)

            return A, x

    def __Solveur(self, problemType: str, resolution=2, useCholesky=False, A_isSymetric=False):
        """Resolution du de la simulation
        renvoie la solution"""

        assert problemType in Simu.problemTypes()

        # Résolution du plus rapide au plus lent 2, 3, 1
        if resolution == 1:
            useCholesky=False #la matrice ne sera pas symétrique definie positive
            # Résolution par la méthode des pénalisations

            # Construit le système matricielle pénalisé
            b = self.__Application_Conditions_Neuman(problemType)
            A, b = self.__Application_Conditions_Dirichlet(problemType, b, resolution)

            ddl_Connues, ddl_Inconnues = self.__Construit_ddl_connues_inconnues(problemType)

            # Résolution du système matricielle pénalisé
            x = self.__Solve_Axb(A, b, useCholesky, A_isSymetric)

        elif resolution == 2:
            
            # Construit le système matricielle
            b = self.__Application_Conditions_Neuman(problemType)
            A, x = self.__Application_Conditions_Dirichlet(problemType, b, resolution)

            # Récupère les ddls
            ddl_Connues, ddl_Inconnues = self.__Construit_ddl_connues_inconnues(problemType)

            # Décomposition du système matricielle en connues et inconnues 
            # Résout : Aii * ui = bi - Aic * xc
            Aii = A[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()
            Aic = A[ddl_Inconnues, :].tocsc()[:, ddl_Connues].tocsr()
            bi = b[ddl_Inconnues,0]
            xc = x[ddl_Connues,0]

            bDirichlet = Aic.dot(xc)

            test = bi-bDirichlet

            xi = self.__Solve_Axb(Aii, bi-bDirichlet, useCholesky, A_isSymetric)

            # Reconstruction de la solution
            x = x.toarray().reshape(x.shape[0])
            x[ddl_Inconnues] = xi       

        return np.array(x)

    def __Solve_Axb(self, A, b, useCholesky=False, A_isSymetric=False):
        # tic = TicTac()
        if useCholesky:
            # il se trouve que c'est plus rapide de ne pas l'utiliser

            from sksparse.cholmod import cholesky, cholesky_AAt
            # exemple matrice 3x3 : https://www.youtube.com/watch?v=r-P3vkKVutU&t=5s 
            # doc : https://scikit-sparse.readthedocs.io/en/latest/cholmod.html#sksparse.cholmod.analyze
            # Installation : https://www.programmersought.com/article/39168698851/                

            factor = cholesky(A.tocsc())
            # factor = cholesky_AAt(A.tocsc())
            
            # x_chol = factor(b.tocsc())
            x_chol = factor.solve_A(b.tocsc())                

            x = x_chol.toarray().reshape(x_chol.shape[0])

        else:
            # il reste à faire le lien avec umfpack pour réorgraniser encore plus rapidement

            # linear solver scipy : https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#solving-linear-problems
            # minim sous contraintes : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
            useUmfpack = False

            if useUmfpack:
                # from scikits.umfpack import umf
                import scikits.umfpack as um
                lu = um.splu(A)
                x = um.spsolve(A, b)
            else:
                # décomposition Lu derrière https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html
                
                # sparse.linalg.use_solver(useUmfpack=True)
                hideFacto = True

                # permc_spec = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"
                if A_isSymetric and not self.materiau.isDamaged:
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
                    pass
                
            # sp.__config__.show()
            # from scipy.linalg import lapack
        
        # tac = tic.Tac("test","Resol",True)

        return x

# ------------------------------------------- CONDITIONS LIMITES -------------------------------------------

    def Clear_Bc_Dirichlet(self):
        """Enlève les conditions limites de Dirichlet"""
        self.__Bc_Dirichlet = []

    def get_ddls_Dirichlet(self, problemType: str):
        assert problemType in Simu.problemTypes()

        ddls = []
        for bc_Dirichlet in self.__Bc_Dirichlet:
            bc_Dirichlet = cast(BoundaryCondition, bc_Dirichlet)
            if bc_Dirichlet.problemType == problemType:
                ddls.extend(bc_Dirichlet.ddls)
        return np.array(ddls)

    def Clear_Bc_Neuman(self):
        """Enlève les conditions limites de Neumann"""
        self.__Bc_Neuman = []
    
    def get_ddls_Neumann(self, problemType: str):
        assert problemType in Simu.problemTypes()

        ddls = []
        for bc_Neumann in self.__Bc_Neuman:
            bc_Neumann = cast(BoundaryCondition, bc_Neumann)
            if bc_Neumann.problemType == problemType:
                ddls.extend(bc_Neumann.ddls)
        return np.array(ddls)

    def __get_ddls_connect(self, problemType:str, connect_e: np.ndarray, directions: list):

        assert problemType in self.problemTypes()
        self.problemDirections(problemType, directions)

        match problemType:
            case "damage":
                return connect_e.reshape(-1)
            case "displacement":

                indexes = {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                }
                listeIndex=[]
                for dir in directions:
                    listeIndex.append(indexes[dir])

                Ne = connect_e.shape[0]
                nPe = connect_e.shape[1]

                connect_e_repet = np.repeat(connect_e, len(directions), axis=0).reshape(-1,nPe)
                listIndex = np.repeat(np.array(listeIndex*nPe), Ne, axis=0).reshape(-1,nPe)

                ddls_dir = np.array(connect_e_repet*self.__dim + listIndex, dtype=int)

                return ddls_dir.reshape(-1)

    def __get_ddls_noeuds(self, problemType:str, noeuds:np.ndarray, directions: list):

        assert problemType in self.problemTypes()
        self.problemDirections(problemType, directions)

        match problemType:
            case "damage":
                return noeuds.reshape(-1)
            case "displacement":
                ddls_dir = np.zeros((noeuds.shape[0], len(directions)), dtype=int)
                for d, direction in enumerate(directions):
                    match direction:
                        case "x":
                            index = 0
                        case "y":
                            index = 1
                        case "z":
                            assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                            index = 2
                    
                    ddls_dir[:,d] = noeuds * self.__dim + index

                return ddls_dir.reshape(-1)

    def __evalue(self, coordo: np.ndarray, valeurs, option="noeuds"):
        
        assert option in ["noeuds","gauss"]
        match option:
            case "noeuds":
                valeurs_eval = np.zeros((coordo.shape[0],1))
            case "gauss":
                valeurs_eval = np.zeros((coordo.shape[0],coordo.shape[1],1))
        
        if isinstance(valeurs, LambdaType):
            # Evalue la fonction aux coordonnées

            match option:
                case "noeuds":
                    valeurs_eval[:] = valeurs(coordo[:,0], coordo[:,1], coordo[:,2])
                case "gauss":
                    valeurs_eval[:,:] = valeurs(coordo[:,:,0], coordo[:,:,1], coordo[:,:,2])
        else:
            match option:
                case "noeuds":
                    valeurs_eval[:] = valeurs
                case "gauss":
                    valeurs_eval[:,:] = valeurs

        return valeurs_eval

    def add_surfLoad(self, problemType:str, noeuds: np.ndarray, directions: list, valeurs: list,
        description=""):

        match self.__dim:
            case 2:
                valeurs, ddls = self.__lineLoad(problemType=problemType,
                noeuds=noeuds, directions=directions, valeurs=valeurs)

                valeurs = valeurs*self.materiau.comportement.epaisseur
            case 3:
                valeurs, ddls = self.__surfload(problemType=problemType,
                noeuds=noeuds, directions=directions, valeurs=valeurs)


        self.__Add_Bc_Neumann(problemType=problemType,
        ddls=ddls, directions=directions, valeurs_ddls=valeurs,
        description=description)

    def add_lineLoad(self, problemType:str, noeuds: np.ndarray, directions: list, valeurs: list,
        description=""):

        valeurs, ddls = self.__lineLoad(problemType=problemType,
        noeuds=noeuds, directions=directions, valeurs=valeurs)

        self.__Add_Bc_Neumann(problemType=problemType,
        ddls=ddls, directions=directions, valeurs_ddls=valeurs,
        description=description)

    def __lineLoad(self,problemType:str, noeuds: np.ndarray, directions: list, valeurs: list):        

        assert problemType in self.problemTypes()
        self.problemDirections(problemType, directions)

        # Récupération des matrices pour le calcul
        mesh = self.__mesh
        groupElem1D = mesh.get_groupElem(1)

        # Récupère les elements qui utilisent exclusivement les noeuds
        elements = groupElem1D.get_elements(noeuds, exclusivement=True)
        connect_e = groupElem1D.connect[elements]
        Ne = elements.shape[0]
        
        # récupère les coordonnées des points de gauss dans le cas ou on a besoin dévaluer la fonction
        coordo_e_p = groupElem1D.get_coordo_e_p("masse",elements)
        nPg = coordo_e_p.shape[1]

        N_pg = groupElem1D.get_N_pg("masse")

        # objets d'integration
        jacobien_e_pg = groupElem1D.get_jacobien_e_pg("masse")[elements]
        gauss = groupElem1D.get_gauss("masse")
        poid_pg = gauss.poids

        # initialise le vecteur de valeurs pour chaque element et chaque pts de gauss
        valeurs_ddl_dir = np.zeros((Ne*groupElem1D.nPe, len(directions)))

        for d, dir in enumerate(directions):
            eval_e_p = self.__evalue(coordo_e_p, valeurs[d], option="gauss")
            valeurs_e_p = np.einsum('ep,p,epi,pij->epij', jacobien_e_pg, poid_pg, eval_e_p, N_pg, optimize=True)
            valeurs_e = np.sum(valeurs_e_p, axis=1)
            valeurs_ddl_dir[:,d] = valeurs_e.reshape(-1)

        # valeurs_e_p = 
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        
        ddls = self.__get_ddls_connect(problemType, connect_e, directions)        

        return valeurs_ddls, ddls
    
    def __surfload(self,problemType:str, noeuds: np.ndarray, directions: list, valeurs: list):

        assert problemType in self.problemTypes()
        self.problemDirections(problemType, directions)

        # Récupération des matrices pour le calcul
        mesh = self.__mesh
        groupElem2D = mesh.get_groupElem(2)

        # Récupère les elements qui utilisent exclusivement les noeuds
        elements = groupElem2D.get_elements(noeuds, exclusivement=True)
        connect_e = groupElem2D.connect[elements]
        Ne = elements.shape[0]
        
        # récupère les coordonnées des points de gauss dans le cas ou on a besoin dévaluer la fonction
        coordo_e_p = groupElem2D.get_coordo_e_p("masse",elements)
        nPg = coordo_e_p.shape[1]

        N_pg = groupElem2D.get_N_pg("masse")

        # objets d'integration
        jacobien_e_pg = groupElem2D.get_jacobien_e_pg("masse")[elements]
        gauss = groupElem2D.get_gauss("masse")
        poid_pg = gauss.poids

        # initialise le vecteur de valeurs pour chaque element et chaque pts de gauss
        valeurs_ddl_dir = np.zeros((Ne*groupElem2D.nPe, len(directions)))

        for d, dir in enumerate(directions):
            eval_e_p = self.__evalue(coordo_e_p, valeurs[d], option="gauss")
            valeurs_e_p = np.einsum('ep,p,epi,pij->epij', jacobien_e_pg, poid_pg, eval_e_p, N_pg, optimize=True)
            valeurs_e = np.sum(valeurs_e_p, axis=1)
            valeurs_ddl_dir[:,d] = valeurs_e.reshape(-1)

        # valeurs_e_p = 
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        
        ddls = self.__get_ddls_connect(problemType, connect_e, directions)        

        return valeurs_ddls, ddls
    
    def __Add_Bc_Neumann(self, problemType: str, ddls: np.ndarray, directions: list, valeurs_ddls: np.ndarray,
    description=""):
        """Ajoute les conditions de Neumann"""

        tic = TicTac()
        
        assert problemType in Simu.problemTypes(), "Ce type de probleme n'est pas implémenté"

        match problemType:

            case "damage":
                
                new_Bc = BoundaryCondition(problemType=problemType,
                ddls=ddls, directions=[], valeurs_ddls=valeurs_ddls,
                description=description, marker='.', color='red')

            case "displacement":

                new_Bc = BoundaryCondition(problemType=problemType,
                ddls=ddls, directions=directions, valeurs_ddls=valeurs_ddls,
                description=description, marker='.', color='blue')

                # Verifie si les ddls de Neumann ne coincidenet pas avec dirichlet

                ddl_Dirchlet = self.get_ddls_Dirichlet(problemType)
                for d in ddls: 
                    assert d not in ddl_Dirchlet, "On ne peut pas appliquer conditions dirchlet et neumann aux memes ddls"
                

        self.__Bc_Neuman.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Neumann", self.__verbosity)

    def add_dirichlet(self, problemType: str,
        noeuds: np.ndarray, directions: list, valeurs: np.ndarray,
        description=""):

        coordo = self.mesh.coordo
        coordo_n = coordo[noeuds]
        Nn = noeuds.shape[0]

        # initialise le vecteur de valeurs pour chaque noeuds
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__evalue(coordo_n, valeurs[d], option="noeuds")
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        ddls = self.__get_ddls_noeuds(problemType, noeuds, directions)

        self.__Add_Bc_Dirichlet(problemType, ddls, directions, valeurs_ddls, description)
                

    def __Add_Bc_Dirichlet(self, problemType: str,
        ddls: np.ndarray, directions: list, valeurs_ddls: np.ndarray,
        description=""):

        tic = TicTac()

        assert problemType in Simu.problemTypes(), "Ce type de probleme n'est pas implémenté"        

        match problemType:

            case "damage":
                
                new_Bc = BoundaryCondition(problemType=problemType,
                ddls=ddls, valeurs_ddls=valeurs_ddls, directions=directions,
                description=description, marker='.', color='red')

            case "displacement":

                new_Bc = BoundaryCondition(problemType=problemType,
                ddls=ddls, valeurs_ddls=valeurs_ddls, directions=directions,
                description=description, marker='.', color='green')

                # Verifie si les ddls de Neumann ne coincidenet pas avec dirichlet
                ddl_Neumann = self.get_ddls_Neumann(problemType)
                for d in ddls: 
                    assert d not in ddl_Neumann, "On ne peut pas appliquer conditions dirchlet et neumann aux memes ddls"

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Dirichlet", self.__verbosity)
    
# ------------------------------------------- POST TRAITEMENT ------------------------------------------- 
    
    def __CalculEnergieDef(self):
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        sol_u  = self.GetResultat("displacement", valeursAuxNoeuds=True)

        matriceType = "rigi"
        Epsilon_e_pg = self.__CalculEpsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.__mesh.get_jacobien_e_pg(matriceType)
        poid_pg = self.__mesh.get_poid_pg(matriceType)

        if self.__dim == 2:
            ep = self.materiau.comportement.epaisseur
        else:
            ep = 1

        if self.materiau.isDamaged:

            d = self.GetResultat("damage", valeursAuxNoeuds=True)

            phaseFieldModel = self.materiau.phaseFieldModel
            psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

            # Endommage : psiP_e_pg = g(d) * PsiP_e_pg 
            g_e_pg = phaseFieldModel.get_g_e_pg(d, self.__mesh, matriceType)
            psiP_e_pg = np.einsum('ep,ep->ep', g_e_pg, psiP_e_pg, optimize=True)
            psi_e_pg = psiP_e_pg + psiM_e_pg

            Wdef = np.einsum(',ep,p,ep->', ep, jacobien_e_pg, poid_pg, psi_e_pg)

        else:

            Sigma_e_pg = self.__CalculSigma_e_pg(Epsilon_e_pg, matriceType)
            
            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg)

            # # Calcul par Element fini
            # u_e = self.__mesh.Localises_sol_e(sol_u)
            # Ku_e = self.__ConstruitMatElem_Dep()
            # Wdef = 1/2 * np.einsum('ei,eij,ej->', u_e, Ku_e, u_e)
        
        return Wdef

    def __CalculEpsilon_e_pg(self, u: np.ndarray, matriceType="rigi"):
        """Construit epsilon pour chaque element et chaque points de gauss

        Parameters
        ----------
        u : np.ndarray
            Vecteur des déplacements

        Returns
        -------
        np.ndarray
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))
        """
        
        # Localise les deplacement par element
        u_e = self.__mesh.Localises_sol_e(u)
        comportement = self.materiau.comportement

        B_dep_e_pg = self.__mesh.get_B_dep_e_pg(matriceType)
        if not comportement.useVoigtNotation:
            B_dep_e_pg = comportement.AppliqueCoefSurBrigi(B_dep_e_pg)
        
        Epsilon_e_pg = np.einsum('epik,ek->epi', B_dep_e_pg, u_e, optimize=True)        

        return Epsilon_e_pg

    def __CalculSigma_e_pg(self, Epsilon_e_pg: np.ndarray, matriceType="rigi"):
        """Calcul les contraintes depuis les deformations

        Parameters
        ----------
        Epsilon_e_pg : np.ndarray
            Deformations stockées aux elements et points de gauss (Ne,pg,(3 ou 6))

        Returns
        -------
        np.ndarray
            Renvoie les contrainres endommagé ou non (Ne,pg,(3 ou 6))
        """

        assert Epsilon_e_pg.shape[0] == self.__mesh.Ne
        assert Epsilon_e_pg.shape[1] == self.__mesh.get_nPg(matriceType)

        c = self.materiau.comportement.get_C()

        if self.materiau.isDamaged:

            d = self.__resultats["damage"]

            phaseFieldModel = self.materiau.phaseFieldModel

            SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)

            # Endommage : c = g(d) * cP + cM
            g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matriceType)
            SigmaP_e_pg = np.einsum('ep,epi->epi', g_e_pg, SigmaP_e_pg, optimize=True)

            Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg
            
        else:

            Sigma_e_pg = np.einsum('ik,epk->epi', c, Epsilon_e_pg, optimize=True)
            

        return cast(np.ndarray, Sigma_e_pg)

    def __Save_d(self, dGlob: np.ndarray):
        """Sauvegarde dGlob"""        

        self.__resultats["damage"] = dGlob

    def __Save_u(self, Uglob: np.ndarray):
        """Sauvegarde Uglob"""

        self.__resultats["displacement"] = Uglob
        

    def VerificationOption(self, option):
        """Verification que l'option est bien calculable dans GetResultat

        Parameters
        ----------
        option : str
            option

        Returns
        -------
        Bool
            Réponse du test
        """
        # Construit la liste d'otions pour les résultats en 2D ou 3D

        # Verfie si la simulation à un résultat de déplacement ou d'endommagement
        if "displacement" not in self.__resultats.keys() and "damage" not in self.__resultats.keys():
            print("\nLa simulation n'a pas encore de résultats")
            return False 

        dim = self.__dim
        if dim == 2:
            options = {
                "Stress" : ["Sxx", "Syy", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Exy", "Evm","Strain"],
                "Displacement" : ["dx", "dy", "dz","amplitude","displacement"],
                "Energie" :["Wdef"],
                "Damage" :["damage","psiP"]
            }
        elif dim == 3:
            options = {
                "Stress" : ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy", "Evm","Strain"],
                "Displacement" : ["dx", "dy", "dz","amplitude","displacement"],
                "Energie" :["Wdef"]                
            }

        # Verfication que l'option est dans dans les options
        ContenueDansOptions=False
        for opts in options:
            if option in options[opts]:
                ContenueDansOptions=True
                break
        
        if not ContenueDansOptions:
            print("\nL'option doit etre dans : {}".format(options))
            return False

        return ContenueDansOptions

    def GetResultat(self, option: str, valeursAuxNoeuds=False): 

        verif = self.VerificationOption(option)
        if not verif:
            return None

        if option == "Wdef":
            return self.__CalculEnergieDef()

        if option == "damage":
            if not self.materiau.isDamaged: print("Le matériau n'est pas endommageable")
            assert "damage" in self.__resultats.keys(), "Le vecteur de d'endommagement dglob doit être calculé ou initialisé"
            return self.__resultats["damage"]

        if option == "displacement":
            assert "displacement" in self.__resultats.keys(), "Le vecteur de déplacement uglob doit être initialisé ou calculé"
            return self.__resultats["displacement"]

        Uglob = self.__resultats["displacement"]

        if option == "psiP":
            resultat_e_pg = self.CalcPsiPlus_e_pg()
            resultat = np.mean(resultat_e_pg, axis=1)

        dim = self.__dim

        # Localisation        
        u_e_n = self.__mesh.Localises_sol_e(Uglob)

        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__CalculEpsilon_e_pg(Uglob)
        Sigma_e_pg = self.__CalculSigma_e_pg(Epsilon_e_pg)

        # Moyenne sur l'élement
        Epsilon_e = np.mean(Epsilon_e_pg, axis=1)
        Sigma_e = np.mean(Sigma_e_pg, axis=1)

        coef = self.materiau.comportement.coef

        Ne = self.__mesh.Ne
        Nn = self.__mesh.Nn
        
        if "d" in option or "amplitude" in option:

            coordoDef = self.GetCoordUglob()
            dx = coordoDef[:,0]
            dy = coordoDef[:,1]
            dz = coordoDef[:,2]

            if option == "dx":
                resultat = dx
            elif option == "dy":
                resultat = dy
            elif option == "dz":
                resultat = dz
            elif option == "amplitude":
                resultat = np.sqrt(dx**2 + dy**2 + dz**2)

            if not valeursAuxNoeuds:
                # On récupère les déplacements sur chaque element
                listCoordX = np.array(range(0, u_e_n.shape[1], self.__dim))
                dx_e_n = u_e_n[:,listCoordX]; dx_e = np.mean(dx_e_n, axis=1)
                dy_e_n = u_e_n[:,listCoordX+1]; dy_e = np.mean(dy_e_n, axis=1)
                if self.__dim == 3:
                    dz_e_n = u_e_n[:,listCoordX+2]; dz_e = np.mean(dz_e_n, axis=1)

                if option == "dx":
                    resultat = dx_e
                elif option == "dy":
                    resultat = dy_e
                elif option == "dz":
                    resultat = dz_e
                elif option == "amplitude":
                    resultat = np.sqrt(dx_e**2 + dy_e**2 + dz_e**2)
            
            return resultat

        elif "S" in option and not option == "Strain":            

            if dim == 2:

                Sxx_e = Sigma_e[:,0]
                Syy_e = Sigma_e[:,1]
                Sxy_e = Sigma_e[:,2]
                
                if not self.materiau.comportement.useVoigtNotation:
                    Sxy_e = Sxy_e/coef
                
                Svm_e = np.sqrt(Sxx_e**2+Syy_e**2-Sxx_e*Syy_e+3*Sxy_e**2)

                if option == "Sxx":
                    resultat = Sxx_e
                elif option == "Syy":
                    resultat = Syy_e
                elif option == "Sxy":
                    resultat = Sxy_e
                elif option == "Svm":
                    resultat = Svm_e

            elif dim == 3:

                Sxx_e = Sigma_e[:,0]
                Syy_e = Sigma_e[:,1]
                Szz_e = Sigma_e[:,2]
                Syz_e = Sigma_e[:,3]
                Sxz_e = Sigma_e[:,4]
                Sxy_e = Sigma_e[:,5]

                if not self.materiau.comportement.useVoigtNotation:
                    Syz_e = Syz_e/coef
                    Sxz_e = Sxz_e/coef
                    Sxy_e = Sxy_e/coef
                    

                Svm_e = np.sqrt(((Sxx_e-Syy_e)**2+(Syy_e-Szz_e)**2+(Szz_e-Sxx_e)**2+6*(Sxy_e**2+Syz_e**2+Sxz_e**2))/2)

                if option == "Sxx":
                    resultat = Sxx_e
                elif option == "Syy":
                    resultat = Syy_e
                elif option == "Szz":
                    resultat = Szz_e
                elif option == "Syz":
                    resultat = Syz_e
                elif option == "Sxz":
                    resultat = Sxz_e
                elif option == "Sxy":
                    resultat = Sxy_e
                elif option == "Svm":
                    resultat = Svm_e

            if option == "Stress":
                resultat = np.append(Sigma_e, Svm_e.reshape((Ne,1)), axis=1)            
            
        elif option in ["E","Strain"]:

            if dim == 2:

                Exx_e = Epsilon_e[:,0]
                Eyy_e = Epsilon_e[:,1]
                Exy_e = Epsilon_e[:,2]/coef
                
                Evm_e = np.sqrt(Exx_e**2+Eyy_e**2-Exx_e*Eyy_e+3*Exy_e**2)

                if option == "Exx":
                    resultat = Exx_e
                elif option == "Eyy":
                    resultat = Eyy_e
                elif option == "Exy":
                    resultat = Exy_e
                elif option == "Evm":
                    resultat = Evm_e

            elif dim == 3:

                Exx_e = Epsilon_e[:,0]
                Eyy_e = Epsilon_e[:,1]
                Ezz_e = Epsilon_e[:,2]
                Eyz_e = Epsilon_e[:,3]/coef
                Exz_e = Epsilon_e[:,4]/coef
                Exy_e = Epsilon_e[:,5]/coef

                Evm_e = np.sqrt(((Exx_e-Eyy_e)**2+(Eyy_e-Ezz_e)**2+(Ezz_e-Exx_e)**2+6*(Exy_e**2+Eyz_e**2+Exz_e**2))/2)

                if option == "Exx":
                    resultat = Exx_e
                elif option == "Eyy":
                    resultat = Eyy_e
                elif option == "Ezz":
                    resultat = Ezz_e
                elif option == "Eyz":
                    resultat = Eyz_e
                elif option == "Exz":
                    resultat = Exz_e
                elif option == "Exy":
                    resultat = Exy_e
                elif option == "Evm":
                    resultat = Evm_e                
            
            if option == "Strain":
                resultat = np.append(Epsilon_e, Evm_e.reshape((Ne,1)), axis=1)

        if valeursAuxNoeuds:

            if resultat.size > Ne:

                resultats_e = resultat.copy()
                resultats_n = np.zeros((Nn, resultats_e.shape[1]))
                for e in range(resultats_e.shape[1]):
                    resultats_n[:,e] = self.__InterpolationAuxNoeuds(resultats_e[:,e])
                resultat = resultats_n
            else:
                resultat = self.__InterpolationAuxNoeuds(resultat)

        return resultat        
    
    def __InterpolationAuxNoeuds(self, valeurs_e):
        """Pour chaque noeuds on récupère les valeurs des élements autour de lui pour on en fait la moyenne

        Returns:
            [type]: [description]
        """

        # tic = TicTac()
        
        connect_n_e = self.__mesh.connect_n_e

        nombreApparition = np.array(np.sum(connect_n_e, axis=1)).reshape(self.__mesh.Nn,1)

        valeurs_n_e = connect_n_e.dot(valeurs_e.reshape(self.__mesh.Ne,1))

        valeurs_n = valeurs_n_e/nombreApparition

        # tic.Tac("Post Traitement","Interpolation aux noeuds", False)

        return valeurs_n.reshape(-1)

    def Resume(self):

        if not self.VerificationOption("Wdef"):
            return
        
        Wdef = self.GetResultat("Wdef")
        print("\nW def = {:.6f} N.mm".format(Wdef))
        
        Svm = self.GetResultat("Svm", valeursAuxNoeuds=False)
        print("\nSvm max = {:.6f} MPa".format(Svm.max()))

        # Affichage des déplacements
        dx = self.GetResultat("dx", valeursAuxNoeuds=True)
        print("\nUx max = {:.6f} mm".format(dx.max()))
        print("Ux min = {:.6f} mm".format(dx.min()))

        dy = self.GetResultat("dy", valeursAuxNoeuds=True)
        print("\nUy max = {:.6f} mm".format(dy.max()))
        print("Uy min = {:.6f} mm\n".format(dy.min()))

        if self.__dim == 3:
            dz = self.GetResultat("dz", valeursAuxNoeuds=True)
            print("\nUz max = {:.6f} mm".format(dz.max()))
            print("Uz min = {:.6f} mm".format(dz.min()))
    
    def GetCoordUglob(self):
        """Renvoie les déplacements sous la forme [dx, dy, dz] (Nn,3)        """

        Nn = self.__mesh.Nn
        dim = self.__dim        

        if self.VerificationOption("displacement"):

            Uglob = self.__resultats["displacement"]

            coordo = Uglob.reshape((Nn,-1))
           
            if dim == 2:
                coordo = np.append(coordo,np.zeros((Nn,1)), axis=1)                       

            return coordo
        else:
            return None

    def Update(self, displacement=None, damage=None):

        if isinstance(displacement, np.ndarray):
            assert displacement.size == self.__mesh.Nn*self.__dim, "Le vecteur n'a pas la bonne taille (Nn*dim)"
            self.__resultats["displacement"] = displacement

        if isinstance(damage, np.ndarray):
            assert damage.size == self.__mesh.Nn, "Le vecteur n'a pas la bonne taille (Nn)"
            self.__resultats["damage"] = damage

# ====================================

import unittest

class Test_Simu(unittest.TestCase):    
    
    def CreationDesSimusElastique2D(self):
        
        dim = 2

        # Paramètres géométrie
        L = 120;  #mm
        h = 120;    
        b = 13

        # Charge a appliquer
        P = -800 #N

        # Paramètres maillage
        taille = L/2

        comportement = Elas_Isot(dim)

        materiau = Materiau(comportement)

        self.simulations2DElastique = []

        listMesh2D = Interface_Gmsh.Construction2D(L=L, h=h, taille=taille)

        # Pour chaque type d'element 2D
        for mesh in listMesh2D:

            mesh = cast(Mesh, mesh)
            
            simu = Simu(mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

            simu.add_dirichlet("displacement", noeuds_en_0, ["x","y"], [0, 0], description="Encastrement")
            simu.add_lineLoad("displacement",noeuds_en_L, ["y"], [-P/h])

            self.simulations2DElastique.append(simu)

    def CreationDesSimusElastique3D(self):

        dir_path = Dossier.GetPath()
        fichier = dir_path + '\\models\\part.stp'

        dim = 3

        # Paramètres géométrie
        L = 120  #mm
        h = 13    
        b = 13

        P = -800 #N

        # Paramètres maillage        
        taille = L/2

        comportement = Elas_Isot(dim)

        materiau = Materiau(comportement)
        
        self.simulations3DElastique = []

        interfaceGmsh = Interface_Gmsh(verbosity=False)

        for t, elemType in enumerate(GroupElem.get_Types3D()):

            mesh = interfaceGmsh.Importation3D(fichier, elemType=elemType, tailleElement=taille)

            simu = Simu(mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

            simu.add_surfLoad("displacement",noeuds_en_L, ["z"], [-P/h/b])

            simu.add_dirichlet("displacement", noeuds_en_0,["x","y","z"], [0,0,0], "Encastrement")

            self.simulations3DElastique.append(simu)
    
    def setUp(self):
        self.CreationDesSimusElastique2D()
        self.CreationDesSimusElastique3D()  

    def test_ResolutionDesSimulationsElastique2D(self):
        # Pour chaque type de maillage on simule
        for simu in self.simulations2DElastique:
            simu = cast(Simu, simu)
            simu.Solve_u()

    def test_ResolutionDesSimulationsElastique3D(self):
        # Pour chaque type de maillage on simule
        for simu in self.simulations3DElastique:
            simu = cast(Simu, simu)
            simu.Solve_u()

    def test__ConstruitMatElem_Dep(self):
        for simu in self.simulations2DElastique:
            simu = cast(Simu, simu)
            Ke_e = simu.ConstruitMatElem_Dep()
            self.__VerificationConstructionKe(simu, Ke_e)

    # ------------------------------------------- Vérifications ------------------------------------------- 

    def __VerificationConstructionKe(self, simu: Simu, Ke_e, d=[]):
            """Ici on verifie quon obtient le meme resultat quavant verification vectorisation

            Parameters
            ----------
            Ke_e : nd.array par element
                Matrice calculé en vectorisant        
            d : ndarray
                Champ d'endommagement
            """

            tic = TicTac()

            matriceType = "rigi"

            # Data
            mesh = simu.mesh
            nPg = mesh.get_nPg(matriceType)
            listPg = list(range(nPg))
            listElement = simu.listElement
            materiau = simu.materiau
            C = materiau.comportement.get_C()

            listKe_e = []

            B_dep_e_pg = mesh.get_B_dep_e_pg(matriceType)

            if not materiau.comportement.useVoigtNotation:
                B_dep_e_pg = materiau.comportement.AppliqueCoefSurBrigi(B_dep_e_pg)

            jacobien_e_pg = mesh.get_jacobien_e_pg(matriceType)
            poid_pg = mesh.get_poid_pg(matriceType)
            for e in listElement:            
                # Pour chaque poing de gauss on construit Ke
                Ke = 0
                for pg in listPg:
                    jacobien = jacobien_e_pg[e,pg]
                    poid = poid_pg[pg]
                    B_pg = B_dep_e_pg[e,pg]

                    K = jacobien * poid * B_pg.T.dot(C).dot(B_pg)

                    if len(d)==0:   # probleme standart
                        
                        Ke += K
                    else:   # probleme endomagement
                        
                        de = np.array([d[mesh.connect[e]]])
                        
                        # Bourdin
                        g = (1-mesh.N_mass_pg[pg].dot(de))**2
                        # g = (1-de)**2
                        
                        Ke += g * K
                # # print(Ke-listeKe[e.id])
                if mesh.dim == 2:
                    listKe_e.append(Ke)
                else:
                    listKe_e.append(Ke)                

            tic.Tac("Matrices","Calcul des matrices elementaires (boucle)", False)
            
            # Verification
            Ke_comparaison = np.array(listKe_e)
            test = Ke_e - Ke_comparaison

            test = np.testing.assert_array_almost_equal(Ke_e, Ke_comparaison, verbose=False)

            self.assertIsNone(test)
            

if __name__ == '__main__':        
    try:
        Affichage.Clear()
        unittest.main(verbosity=2)    
    except:
        print("")   

