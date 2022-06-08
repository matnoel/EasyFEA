

from types import LambdaType
from typing import cast

import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


from GroupElem import GroupElem
import Affichage
from Mesh import Mesh
from BoundaryCondition import BoundaryCondition
from Materiau import *
from TicTac import TicTac
from Interface_Gmsh import Interface_Gmsh
import Dossier
    
class Simu:

# ------------------------------------------- CONSTRUCTEUR ------------------------------------------- 

    def __init__(self, mesh: Mesh, materiau: Materiau, verbosity=True):
        """Creation d'une simulation

        Args:
            dim (int): Dimension de la simulation (2D ou 3D)
            mesh (Mesh): Maillage que la simulation va utiliser
            materiau (Materiau): Materiau utilisé
            verbosity (bool, optional): La simulation ecrira dans la console. Defaults to True.
        """

        if verbosity:
            Affichage.NouvelleSection("Simulation")

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
        
        # resultats
        self.__init_results()

        # Conditions Limites
        self.Init_Bc_Neumann()
        self.Init_Bc_Dirichlet()

# ------------------------------------------- FONCTIONS ------------------------------------------- 

    @staticmethod
    def CheckProblemTypes(problemType:str):
        """Verifie si ce type de probleme est implénté"""
        problemTypes = ["displacement","damage"]
        assert problemType in problemTypes, "Ce type de probleme n'est pas implémenté"

    @staticmethod
    def CheckDirections(dim: int, problemType:str, directions:list):
        """Verifie si les direcrtions renseignés sont possible pour le probleme"""
        Simu.CheckProblemTypes(problemType)

        match problemType:
            case "damage":
                assert directions == ["d"]
            case "displacement":
                for d in directions:
                    assert d in ["x","y","z"]
                    if dim == 2: assert d != "z", "Lors d'une simulation 2d on ne peut appliquer ques des conditions suivant x et y"
                assert dim >= len(directions)
        
    def __get_mesh(self):
        return self.__mesh
    mesh = cast(Mesh, property(__get_mesh))

    def __getdim(self):
        return self.__dim
    dim = cast(int, property(__getdim))

# ------------------------------------------- RESULTATS ------------------------------------------- 

    def __init_results(self):
        """Construit les arrays qui vont stocker les resultats"""

        self.__displacement = np.zeros(self.__mesh.Nn*self.__dim)
        """déplacements"""
        
        if self.materiau.isDamaged:
            self.__damage = np.zeros(self.__mesh.Nn)
            """endommagement"""
            self.__psiP_e_pg = []
            """densité d'energie elastique en tension PsiPlus(e, pg, 1)"""

            columns = ['displacement', 'damage']
        else:
            columns = ['displacement']
            
        self.__results = pd.DataFrame(columns=columns)
        """tableau panda qui contient les résultats"""

        self.Save_solutions()

    def get_result(self, index=None):
        """Recupère le resultat stocké dans le data frame panda"""
        if index == None:
            return self.__results.loc[[self.__results.shape[0]]]
        else:
            return self.__results.loc[[index]]

    def Get_results(self):
        """Renvoie le data frame panda qui stocke les résultats"""
        return self.__results

    def Save_solutions(self):
        """Sauvegarde les résultats de l'itération"""

        iter = self.__results.index.shape[0]

        if self.materiau.isDamaged:
            iter = {                
                'displacement' : [self.__displacement],
                'damage' : [self.__damage]
            }
        else:
            iter = {                
                'displacement' : [self.__displacement]
            }

        new = pd.DataFrame(iter)

        self.__results = pd.concat([self.__results, new], ignore_index=True)

    def Update_iter(self, iter: int):
        """Met la simulation à literation renseignée"""

        # On va venir récupérer les resultats stocké dans le tableau pandas
        results = self.get_result(iter)

        if self.materiau.isDamaged: 
            self.__psiP_e_pg = []           
            self.__damage = results["damage"].values[0]
            self.__displacement = results["displacement"].values[0]
        else:
            self.__displacement = results["displacement"].values[0]
    
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

        B_dep_e_pg = comportement.AppliqueCoefSurBrigi(B_dep_e_pg)
            

        mat = comportement.get_C()
        # Ici on le materiau est homogène
        # Il est possible de stpcker ça pour ne plus avoir à recalculer        

        if isDamaged:   # probleme endomagement

            d = self.__damage

            u = self.__displacement

            phaseFieldModel = self.materiau.phaseFieldModel
                
            # Calcul la deformation nécessaire pour le split
            Epsilon_e_pg = self.__Calc_Epsilon_e_pg(u, matriceType)

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

    def Solve_u(self, useCholesky=False):
        """Resolution du probleme de déplacement"""        

        tic = TicTac()

        Uglob = self.__Solveur(problemType="displacement", useCholesky=useCholesky, A_isSymetric=True)

        tic.Tac("Résolution deplacement","Résolution pour le problème de déplacement", self.__verbosity)
        
        assert Uglob.shape[0] == self.mesh.Nn*self.__dim

        self.__displacement = Uglob

        return Uglob

# ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def __Calc_PsiPlus_e_pg(self, useHistory=True):
            # Pour chaque point de gauss de tout les elements du maillage on va calculer psi+
            # Calcul de la densité denergie de deformation en traction

            assert self.materiau.isDamaged, "pas de modèle d'endommagement"

            phaseFieldModel = self.materiau.phaseFieldModel
            
            u = self.__displacement
            d = self.__damage

            testu = isinstance(u, np.ndarray) and (u.shape[0] == self.__mesh.Nn*self.__dim )
            testd = isinstance(d, np.ndarray) and (d.shape[0] == self.__mesh.Nn )

            assert testu or testd,"Problème de dimension"

            Epsilon_e_pg = self.__Calc_Epsilon_e_pg(u, "masse")
            # ici le therme masse est important sinon on sous intègre

            # Calcul l'energie
            old_psiP = self.__psiP_e_pg

            nPg = self.__mesh.get_nPg("masse")

            psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

            if useHistory:
                if len(old_psiP) == 0:
                    # Pas encore d'endommagement disponible
                    old_psiP = np.zeros((self.__mesh.Ne, nPg))

                inc_H = psiP_e_pg - old_psiP

                elements, pdGs = np.where(inc_H < 0)

                psiP_e_pg[elements, pdGs] = old_psiP[elements,pdGs]           

                # new = np.linalg.norm(psiP_e_pg)
                # old = np.linalg.norm(self.__psiP_e_pg)
                # assert new >= old, "Erreur"

            self.__psiP_e_pg = psiP_e_pg

            return self.__psiP_e_pg
    
    def __ConstruitMatElem_Pfm(self):
        
        tic = TicTac()

        phaseFieldModel = self.materiau.phaseFieldModel

        # Data
        k = phaseFieldModel.k
        PsiP_e_pg = self.__Calc_PsiPlus_e_pg()
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
    
    def Solve_d(self):
        """Resolution du problème d'endommagement"""
         
        tic = TicTac()

        dGlob = self.__Solveur(problemType="damage", useCholesky=False, A_isSymetric=False)

        tic.Tac("Résolution endommagement",f"Résolution pour le problème de endommagement" , self.__verbosity)

        assert dGlob.shape[0] == self.mesh.Nn

        self.__damage = dGlob

        return dGlob

# ------------------------------------------------- SOLVEUR -------------------------------------------------

    def __Construit_ddl_connues_inconnues(self, problemType: str):
        """Récupère les ddl Connues et Inconnues

        Args:
            vector (bool, optional): Travail sur un vecteur ou un scalaire. Defaults to True.

        Returns:
            list(int), list(int): ddl_Connues, ddl_Inconnues
        """
        
        Simu.CheckProblemTypes(problemType)

        # Construit les ddls connues
        ddls_Connues = []

        ddls_Connues = self.Get_ddls_Dirichlet(problemType)
        unique_ddl_Connues = np.unique(ddls_Connues.copy())

        # Construit les ddls inconnues

        match problemType:
            case "damage":
                taille = self.__mesh.Nn
            case "displacement":
                taille = self.__mesh.Nn*self.__dim

        ddls_Inconnues = list(np.arange(taille))
        for ddlConnue in ddls_Connues:
            if ddlConnue in ddls_Inconnues:
                ddls_Inconnues.remove(ddlConnue)
                                
        ddls_Inconnues = np.array(ddls_Inconnues)
        
        verifTaille = unique_ddl_Connues.shape[0] + ddls_Inconnues.shape[0]
        assert verifTaille == taille, f"Problème dans les conditions ddls_Connues + ddls_Inconnues - taille = {verifTaille-taille}"

        return ddls_Connues, ddls_Inconnues

    def __Application_Conditions_Neuman(self, problemType: str):
        """applique les conditions de Neumann"""

        Simu.CheckProblemTypes(problemType)

        ddls = []
        valeurs_ddls = []
        for bcNeumann in self.__Bc_Neumann:
            assert isinstance(bcNeumann, BoundaryCondition)
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

        Simu.CheckProblemTypes(problemType)

        ddls = []
        valeurs_ddls = []
        for bcDirichlet in self.__Bc_Dirichlet:
            assert isinstance(bcDirichlet, BoundaryCondition)
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

        Simu.CheckProblemTypes(problemType)

        # Résolution du plus rapide au plus lent 2, 1
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

            # from pysparse.direct import umfpack

            if useUmfpack:
                # from scikits.umfpack import umf
                # import scikits.umfpack.umfpack as um
                # import scikits.umfpack as um
                # lu = um.splu(A)
                # x = um.spsolve(A, b)
                pass
            else:
                # décomposition Lu derrière https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html
                # sla.use_solver(useUmfpack=True)
                hideFacto = True

                # permc_spec = "MMD_AT_PLUS_A", "MMD_ATA", "COLAMD", "NATURAL"
                if A_isSymetric and not self.materiau.isDamaged:
                    permute="MMD_AT_PLUS_A"
                else:
                    permute="COLAMD"

                if hideFacto:
                    # tic = TicTac()
                    x = sla.spsolve(A, b, permc_spec=permute)
                    # tic.Tac("resol", "resol ax=b",True)
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

    def Init_Bc_Neumann(self):
        """Enlève les conditions limites de Neumann"""
        self.__Bc_Neumann = []
        """Conditions de Neumann list(BoundaryCondition)"""

    def Init_Bc_Dirichlet(self):
        """Enlève les conditions limites de Dirichlet"""
        self.__Bc_Dirichlet = []
        """Conditions de Dirichlet list(BoundaryCondition)"""

    def Get_Bc_Dirichlet(self):
        return self.__Bc_Dirichlet.copy()
    
    def Get_Bc_Neuman(self):
        return self.__Bc_Neumann.copy()

    def Get_ddls_Dirichlet(self, problemType: str):
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Dirichlet)
    
    def Get_ddls_Neumann(self, problemType: str):        
        return BoundaryCondition.Get_ddls(problemType, self.__Bc_Neumann)

    def __evalue(self, coordo: np.ndarray, valeurs, option="noeuds", nPg=0):
        
        assert option in ["noeuds","gauss"]
        match option:
            case "noeuds":
                valeurs_eval = np.zeros(coordo.shape[0])
            case "gauss":
                valeurs_eval = np.zeros((coordo.shape[0],nPg))
        
        if isinstance(valeurs, LambdaType):
            # Evalue la fonction aux coordonnées
            try:
                match option:                
                    case "noeuds":
                        valeurs_eval[:] = valeurs(coordo[:,0], coordo[:,1], coordo[:,2])
                    case "gauss":
                        valeurs_eval[:,:] = valeurs(coordo[:,:,0], coordo[:,:,1], coordo[:,:,2])
            except:
                raise "Doit fournir une fonction lambda de la forme\n lambda x,y,z, : f(x,y,z)"
        else:
            match option:
                case "noeuds":
                    valeurs_eval[:] = valeurs
                case "gauss":
                    valeurs_eval[:,:] = valeurs

        return valeurs_eval
    
    def add_dirichlet(self, problemType: str, noeuds: np.ndarray, valeurs: np.ndarray,directions: list, description=""):
        """Pour le probleme donné renseigne les contions de dirichlet\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z), -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des neouds
        """

        coordo = self.mesh.coordo
        coordo_n = coordo[noeuds]
        Nn = noeuds.shape[0]

        # initialise le vecteur de valeurs pour chaque noeuds
        valeurs_ddl_dir = np.zeros((Nn, len(directions)))

        for d, dir in enumerate(directions):
            eval_n = self.__evalue(coordo_n, valeurs[d], option="noeuds")
            valeurs_ddl_dir[:,d] = eval_n.reshape(-1)
        
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        ddls = BoundaryCondition.Get_ddls_noeuds(self.__dim, problemType, noeuds, directions)

        self.__Add_Bc_Dirichlet(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_lineLoad(self, problemType:str, noeuds: np.ndarray, valeurs: list, directions: list, description=""):
        """Pour le probleme donné applique une force linéique\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z), -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des neouds
        """

        valeurs_ddls, ddls = self.__lineLoad(problemType, noeuds, valeurs, directions)

        self.__Add_Bc_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)

    def add_surfLoad(self, problemType:str, noeuds: np.ndarray, valeurs: list, directions: list, description=""):
        """Pour le probleme donné applique une force surfacique\n
        valeurs est une liste de constantes ou de fonctions\n
        ex: valeurs = [lambda x,y,z : f(x,y,z), -10]

        les fonctions doivent être de la forme lambda x,y,z : f(x,y,z)\n
        les fonctions utilisent les coordonnées x, y et z des neouds
        """

        match self.__dim:
            case 2:
                valeurs_ddls, ddls = self.__lineLoad(problemType, noeuds, valeurs, directions)
                # multiplie par l'epaisseur
                valeurs_ddls = valeurs_ddls*self.materiau.comportement.epaisseur
            case 3:
                valeurs_ddls, ddls = self.__surfload(problemType, noeuds, valeurs, directions)

        self.__Add_Bc_Neumann(problemType, noeuds, valeurs_ddls, ddls, directions, description)    

    def __lineLoad(self,problemType:str, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force linéique\n
        Renvoie valeurs_ddls, ddls"""

        Simu.CheckProblemTypes(problemType)
        Simu.CheckDirections(self.__dim, problemType, directions)

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
            eval_e_p = self.__evalue(coordo_e_p, valeurs[d], option="gauss", nPg=nPg)
            valeurs_e_p = np.einsum('ep,p,ep,pij->epij', jacobien_e_pg, poid_pg, eval_e_p, N_pg, optimize=True)
            valeurs_e = np.sum(valeurs_e_p, axis=1)
            valeurs_ddl_dir[:,d] = valeurs_e.reshape(-1)

        # valeurs_e_p = 
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        
        ddls = BoundaryCondition.Get_ddls_connect(self.__dim, problemType, connect_e, directions)

        return valeurs_ddls, ddls
    
    def __surfload(self, problemType:str, noeuds: np.ndarray, valeurs: list, directions: list):
        """Applique une force surfacique\n
        Renvoie valeurs_ddls, ddls"""

        Simu.CheckProblemTypes(problemType)
        Simu.CheckDirections(self.__dim, problemType, directions)

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

        # Intégre
        for d, dir in enumerate(directions):
            eval_e_p = self.__evalue(coordo_e_p, valeurs[d], option="gauss", nPg=nPg)
            valeurs_e_p = np.einsum('ep,p,ep,pij->epij', jacobien_e_pg, poid_pg, eval_e_p, N_pg, optimize=True)
            valeurs_e = np.sum(valeurs_e_p, axis=1)
            valeurs_ddl_dir[:,d] = valeurs_e.reshape(-1)

        # valeurs_e_p = 
        valeurs_ddls = valeurs_ddl_dir.reshape(-1)
        
        ddls = BoundaryCondition.Get_ddls_connect(self.__dim, problemType, connect_e, directions)

        return valeurs_ddls, ddls
    
    def __Add_Bc_Neumann(self, problemType: str, noeuds: np.ndarray, valeurs_ddls: np.ndarray, ddls: np.ndarray, directions: list, description=""):
        """Ajoute les conditions de Neumann"""

        tic = TicTac()
        
        Simu.CheckProblemTypes(problemType)
        Simu.CheckDirections(self.__dim, problemType, directions)

        match problemType:

            case "damage":
                
                new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Neumann {description}')

            case "displacement":

                new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Neumann {description}')

                # Verifie si les ddls de Neumann ne coincidenet pas avec dirichlet
                ddl_Dirchlet = self.Get_ddls_Dirichlet(problemType)
                for d in ddls: 
                    assert d not in ddl_Dirchlet, "On ne peut pas appliquer conditions dirchlet et neumann aux memes ddls"

        self.__Bc_Neumann.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Neumann", self.__verbosity)   
     
    def __Add_Bc_Dirichlet(self, problemType: str, noeuds: np.ndarray, valeurs_ddls: np.ndarray, ddls: np.ndarray, directions: list, description=""):
        """Ajoute les conditions de Dirichlet"""

        tic = TicTac()

        Simu.CheckProblemTypes(problemType)

        match problemType:

            case "damage":
                
                new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Dirichlet {description}')

            case "displacement":

                new_Bc = BoundaryCondition(problemType, noeuds, ddls, directions, valeurs_ddls, f'Dirichlet {description}')

                # Verifie si les ddls de Neumann ne coincidenet pas avec dirichlet
                ddl_Neumann = self.Get_ddls_Neumann(problemType)
                for d in ddls: 
                    assert d not in ddl_Neumann, "On ne peut pas appliquer conditions dirchlet et neumann aux memes ddls"

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions","Condition Dirichlet", self.__verbosity)
    
# ------------------------------------------- POST TRAITEMENT ------------------------------------------- 
    
    def __Calc_EnergieDef(self):
        """Calcul de l'energie de deformation cinématiquement admissible endommagé ou non
        Calcul de Wdef = 1/2 int_Omega jacobien * poid * Sig : Eps dOmega x epaisseur"""

        sol_u  = self.Get_Resultat("displacement", valeursAuxNoeuds=True)

        matriceType = "rigi"
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(sol_u, matriceType)
        jacobien_e_pg = self.__mesh.get_jacobien_e_pg(matriceType)
        poid_pg = self.__mesh.get_poid_pg(matriceType)

        if self.__dim == 2:
            ep = self.materiau.comportement.epaisseur
        else:
            ep = 1

        if self.materiau.isDamaged:

            d = self.Get_Resultat("damage", valeursAuxNoeuds=True)

            phaseFieldModel = self.materiau.phaseFieldModel
            psiP_e_pg, psiM_e_pg = phaseFieldModel.Calc_psi_e_pg(Epsilon_e_pg)

            # Endommage : psiP_e_pg = g(d) * PsiP_e_pg 
            g_e_pg = phaseFieldModel.get_g_e_pg(d, self.__mesh, matriceType)
            psiP_e_pg = np.einsum('ep,ep->ep', g_e_pg, psiP_e_pg, optimize=True)
            psi_e_pg = psiP_e_pg + psiM_e_pg

            Wdef = np.einsum(',ep,p,ep->', ep, jacobien_e_pg, poid_pg, psi_e_pg)

        else:

            Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg, matriceType)
            
            Wdef = 1/2 * np.einsum(',ep,p,epi,epi->', ep, jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg)

            # # Calcul par Element fini
            # u_e = self.__mesh.Localises_sol_e(sol_u)
            # Ku_e = self.__ConstruitMatElem_Dep()
            # Wdef = 1/2 * np.einsum('ei,eij,ej->', u_e, Ku_e, u_e)
        
        return Wdef

    def __Calc_Epsilon_e_pg(self, u: np.ndarray, matriceType="rigi"):
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

        B_dep_e_pg = comportement.AppliqueCoefSurBrigi(B_dep_e_pg)            
        
        Epsilon_e_pg = np.einsum('epik,ek->epi', B_dep_e_pg, u_e, optimize=True)        

        return Epsilon_e_pg

    def __Calc_Sigma_e_pg(self, Epsilon_e_pg: np.ndarray, matriceType="rigi"):
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

        if self.materiau.isDamaged:

            d = self.__damage

            phaseFieldModel = self.materiau.phaseFieldModel

            SigmaP_e_pg, SigmaM_e_pg = phaseFieldModel.Calc_Sigma_e_pg(Epsilon_e_pg)

            # Endommage : Sig = g(d) * SigP + SigM
            g_e_pg = phaseFieldModel.get_g_e_pg(d, self.mesh, matriceType)
            SigmaP_e_pg = np.einsum('ep,epi->epi', g_e_pg, SigmaP_e_pg, optimize=True)

            Sigma_e_pg = SigmaP_e_pg + SigmaM_e_pg
            
        else:

            c = self.materiau.comportement.get_C()
            Sigma_e_pg = np.einsum('ik,epk->epi', c, Epsilon_e_pg, optimize=True)            

        return cast(np.ndarray, Sigma_e_pg)


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
        dim = self.__dim
        if dim == 2:
            options = {
                "Stress" : ["Sxx", "Syy", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Exy", "Evm","Strain"],
                "Displacement" : ["dx", "dy", "dz","amplitude","displacement", "coordoDef"],
                "Energie" :["Wdef"],
                "Damage" :["damage","psiP"]
            }
        elif dim == 3:
            options = {
                "Stress" : ["Sxx", "Syy", "Szz", "Syz", "Sxz", "Sxy", "Svm","Stress"],
                "Strain" : ["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy", "Evm","Strain"],
                "Displacement" : ["dx", "dy", "dz","amplitude","displacement", "coordoDef"],
                "Energie" :["Wdef"]
            }

        # Verfication que l'option est dans dans les options
        ContenueDansOptions=False
        for opts in options:
            if option in options[opts]:
                ContenueDansOptions=True
                break
        
        if not ContenueDansOptions:
            print(f"\nL'option {option} doit etre dans : {options}")
            return False

        return ContenueDansOptions

    def Get_Resultat(self, option: str, valeursAuxNoeuds=False, iter=None):              

        verif = self.VerificationOption(option)
        if not verif:
            return None

        if iter != None:
            self.Update_iter(iter)

        if option == "Wdef":
            return self.__Calc_EnergieDef()

        if option == "damage":
            if not self.materiau.isDamaged:
                print("Le matériau n'est pas endommageable")
                return
            resultat = self.__damage

        if option == "displacement":
            return self.__displacement
        
        if option == 'coordoDef':
            return self.GetCoordUglob()

        displacement = self.__displacement

        if option == "psiP":
            resultat_e_pg = self.__Calc_PsiPlus_e_pg(useHistory=False)
            resultat = np.mean(resultat_e_pg, axis=1)

        dim = self.__dim

        # Localisation        
        u_e_n = self.__mesh.Localises_sol_e(displacement)

        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__Calc_Epsilon_e_pg(displacement)
        Sigma_e_pg = self.__Calc_Sigma_e_pg(Epsilon_e_pg)

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
                dz_e = np.zeros(dy_e_n.shape[0])
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
                Sxy_e = Sigma_e[:,2]/coef
                
                # TODO Ici il faudrait calculer Szz si deformation plane
                
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
                Syz_e = Sigma_e[:,3]/coef
                Sxz_e = Sigma_e[:,4]/coef
                Sxy_e = Sigma_e[:,5]/coef               

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

                # TODO Ici il faudrait calculer Ezz si contrainte plane
                
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
        
        Wdef = self.Get_Resultat("Wdef")
        print(f"\nW def = {Wdef:.3f} N.mm")
        
        Svm = self.Get_Resultat("Svm", valeursAuxNoeuds=False)
        print(f"\nSvm max = {Svm.max():.3f} MPa")

        # Affichage des déplacements
        dx = self.Get_Resultat("dx", valeursAuxNoeuds=True)
        print(f"\nUx max = {dx.max():.6f} mm")
        print(f"Ux min = {dx.min():.6f} mm")

        dy = self.Get_Resultat("dy", valeursAuxNoeuds=True)
        print(f"\nUy max = {dy.max():.6f} mm")
        print(f"Uy min = {dy.min():.6f} mm\n")

        if self.__dim == 3:
            dz = self.Get_Resultat("dz", valeursAuxNoeuds=True)
            print(f"\nUz max = {dz.max():.6f} mm")
            print(f"Uz min = {dz.min():.6f} mm")
    
    def GetCoordUglob(self):
        """Renvoie les déplacements sous la forme [dx, dy, dz] (Nn,3)        """

        Nn = self.__mesh.Nn
        dim = self.__dim        

        if self.VerificationOption("displacement"):

            Uglob = self.__displacement

            coordo = Uglob.reshape((Nn,-1))
           
            if dim == 2:
                coordo = np.append(coordo,np.zeros((Nn,1)), axis=1)                       

            return coordo
        else:
            return None


