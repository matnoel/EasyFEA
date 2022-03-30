
from typing import cast
from Affichage import Affichage

import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
# from scikits.umfpack import spsolve

from Element import Element
from Mesh import Mesh
from Materiau import Elas_Isot, Materiau
from TicTac import TicTac
from ModelGmsh import ModelGmsh
import Dossier
    
class Simu:
    
    def __get_listElement(self):
        """Renvoie la liste d'element du maillage

        Returns:
            list(int): Element du maillage
        """
        return list(range(self.mesh.Ne))        
    listElement = property(__get_listElement)    

# ------------------------------------------- CONSTRUCTEUR ------------------------------------------- 

    def __init__(self, dim: int, mesh: Mesh, materiau: Materiau, verbosity=True):
        """Creation d'une simulation

        Args:
            dim (int): Dimension de la simulation (2D ou 3D)
            mesh (Mesh): Maillage que la simulation va utiliser
            materiau (Materiau): Materiau utilisé
            verbosity (bool, optional): La simulation ecrira dans la console. Defaults to True.
        """

        # Vérification des valeurs
        assert dim == 2 or dim == 3, "Dimesion compris entre 2D et 3D"
        assert mesh.dim == dim, "Doit avoir la meme dimension que dim"
        assert materiau.dim == dim, "Doit avoir la meme dimension que dim"

        self.__dim = dim
        """dimension de la simulation 2D ou 3D"""
        self.__verbosity = verbosity
        """la simulation peut ecrire dans la console"""

        self.mesh = mesh
        """maillage de la simulation"""
        self.materiau = materiau
        """materiau de la simulation"""
        self.resultats = {}
        """résultats de la simulation"""
        self.PsiP_e_pg = []
        """densité d'energie elastique en tension PsiPlus(e, pg, 1)"""

        # Conditions Limites en déplacement
        self.__BC_Neuman_u = [[],[]]
        """Conditions Limites Dirichlet pour le déplacement list((noeuds, conditions))"""
        self.__BC_Dirichlet_u = [[],[]]
        """Conditions Limites Neumann pour le déplacement list((noeuds, conditions))"""

        # Conditions Limites en endommagement
        self.__BC_Neuman_d = [[],[]]
        """Conditions Limites Neumann pour l'endommagement' list((noeuds, conditions))"""
        self.__BC_Dirichlet_d = [[],[]]
        """Conditions Limites Drichlet pour l'endommagement' list((noeuds, conditions))"""
    
# ------------------------------------------- PROBLEME EN DEPLACEMENT ------------------------------------------- 

    def ConstruitMatElem_Dep(self, d=[]):
        return self.__ConstruitMatElem_Dep(d)

    def __ConstruitMatElem_Dep(self, d):
        """Construit les matrices de rigidités élementaires pour le problème en déplacement

        Args:
            d (list(float)): Endommagement aux noeuds

        Returns:
            array de dim e: les matrices elementaires pour chaque element
        """

        tic = TicTac()

        # Data
        mesh = cast(Mesh, self.mesh)
        nPg = len(mesh.poid_pg)
        
        # Recupère les matrices pour travailler
        jacobien_e_pg = mesh.jacobien_e_pg
        poid_pg = mesh.poid_pg
        B_rigi_e_pg = mesh.B_rigi_e_pg
        mat = self.materiau.comportement.get_C()
        # Ici on le materiau est homogène
        # Il est possible de stpcker ça pour ne plus avoir à recalculer        

        if len(d) !=0 :   # probleme endomagement
            
            N_mass_pg = np.array(mesh.N_mass_pg)

            d_e_n = self.mesh.Localise_e(d)

            d_e_pg = np.einsum('pij,ej->ep', N_mass_pg, d_e_n, optimize=True)

            k_residu = 1e-10

            g_e_pg = (1-d_e_pg)**2 + k_residu

            # Bourdin
            mat_e_pg = np.einsum('ep,ij->epij', g_e_pg, mat, optimize=True)

            Ku_e_pg = np.einsum('ep,p,epki,epkl,eplj->epij', jacobien_e_pg, poid_pg, B_rigi_e_pg, mat_e_pg, B_rigi_e_pg, optimize=True)
            
        else:   # probleme en déplacement simple

            Ku_e_pg = np.einsum('ep,p,epki,kl,eplj->epij', jacobien_e_pg, poid_pg, B_rigi_e_pg, mat, B_rigi_e_pg, optimize=True)
        
        # On somme sur les points d'intégrations
        Ku_e = np.sum(Ku_e_pg, axis=1)

        if self.__dim == 2:
            Ku_e = Ku_e * self.materiau.comportement.epaisseur
        
        tic.Tac("Matrices","Calcul des matrices elementaires (déplacement)", self.__verbosity)

        return Ku_e    
 
    def Assemblage_u(self, d=[]):
        """Construit K global pour le problème en deplacement

        Args:            
            d (list, optional): Endommagement à appliquer au matériau. Defaults to [].
            verification (bool, optional): Verification de l'assemblage avec l'ancienne méthode bcp bcp bcp moin rapide. Defaults to False.
        """

        # Data
        mesh = self.mesh        
        taille = mesh.Nn*self.__dim

        # Construit Ke
        Ku_e = self.__ConstruitMatElem_Dep(d)
        self.__Ku_e = Ku_e # Sauvegarde Ke pour calculer Energie plus rapidement
        
        # Prépare assemblage
        lignesVector_e = mesh.lignesVector_e
        colonnesVector_e = mesh.colonnesVector_e
        
        tic = TicTac()

        # Assemblage
        self.__Ku = sp.sparse.csr_matrix((Ku_e.reshape(-1), (lignesVector_e.reshape(-1), colonnesVector_e.reshape(-1))), shape=(taille, taille))
        """Matrice Kglob pour le problème en déplacement (Nn*dim, Nn*dim)"""

        # Ici j'initialise Fu calr il faudrait calculer les forces volumiques dans __ConstruitMatElem_Dep !!!
        self.__Fu = sp.sparse.csr_matrix((taille, 1))
        """Vecteur Fglob pour le problème en déplacement (Nn*dim, 1)"""

        # plt.spy(self.__Ku)
        # plt.show()

        tic.Tac("Matrices","Assemblage du systême en déplacement", self.__verbosity)
        
        return self.__Ku

    def Solve_u(self, resolution=2, calculContraintesEtDeformation=False, interpolation=False):
        """Resolution du système matricielle A * x = b -> K * u = f
         

        Args:
            resolution (int, optional): Mode de résolution. Defaults to 2.
            calculContraintesEtDeformation (bool, optional): Calcul les Contraintes et les deformations. Defaults to False.
            interpolation (bool, optional): Interpolation aux noeuds. Defaults to False.
       
        """

        tic = TicTac()

        Uglob = self.__Solveur(vector=True, resolution=resolution)

        tic.Tac("Résolution deplacement","Résolution {} pour le problème de déplacement".format(resolution) , self.__verbosity)
        
        self.__Save_u(Uglob, calculContraintesEtDeformation, interpolation)

        return Uglob

    
# ------------------------------------------- PROBLEME ENDOMMAGEMENT ------------------------------------------- 

    def __CalculEpsilon_e_pg(self, u: np.ndarray):
        
        # Localise les deplacement par element
        u_e = self.mesh.Localise_e(u)
        
        # Construit epsilon pour chaque element et chaque points de gauss
        Epsilon_e_pg = np.einsum('epik,ek->epi', self.mesh.B_rigi_e_pg, u_e, optimize=True)
        # Epsilon_e_pg = np.array([[self.__mesh.B_rigi_e_pg[e,pg].dot(u_e[e]) for pg in list(range(self.__mesh.B_rigi_e_pg.shape[1]))] for e in self.listElement])

        return Epsilon_e_pg

    def __CalculSigma_e_pg(self, Epsilon_e_pg: np.ndarray):        

        mat = self.materiau.comportement.get_C()

        Sigma_e_pg = np.einsum('ik,epk->epi', mat, Epsilon_e_pg)
        # Sigma_e_pg = np.array([[self.__materiau.C.dot(Epsilon_e_pg[e,pg]) for pg in list(range(Epsilon_e_pg.shape[1]))] for e in self.listElement])

        return Sigma_e_pg


    def CalcPsiPlus(self, u: np.ndarray):
            # Pour chaque point de gauss de tout les elements du maillage on va calculer psi+
            # Calcul de la densité denergie de deformation en traction
            
            Epsilon_e_pg = self.__CalculEpsilon_e_pg(u)
            
            mat = self.materiau.comportement.get_C()

            # Calcul l'energie
            old_PsiP = self.PsiP_e_pg

            if len(old_PsiP) == 0:
                old_PsiP = np.zeros((self.mesh.Ne, self.mesh.nPg))

            # Bourdin
            h = 1/2 * np.einsum('epk,kl,epl->ep', Epsilon_e_pg, mat, Epsilon_e_pg, optimize=True).reshape((self.mesh.Ne, self.mesh.nPg))
            
            inc_H = h-old_PsiP

            # Pour chaque point d'intégration on verifie que la densité dernerie évolue
            for pg in range(self.mesh.nPg):
                
                # Récupères les noeuds ou la densité d'energie diminue
                noeuds = np.where(inc_H[:,pg] < 0)[0]

                if noeuds.shape[0] > 0:
                    h[noeuds] = old_PsiP[noeuds]

            new = np.linalg.norm(h)
            old = np.linalg.norm(self.PsiP_e_pg)
            assert new >= old, "Erreur"
            self.PsiP_e_pg = h
    
    def __ConstruitMatElem_Pfm(self, Gc, l):
        
        tic = TicTac()

        # Data
        K = Gc*l
        PsiP_e_pg = self.PsiP_e_pg
        r_e_pg = 2*PsiP_e_pg + Gc/l

        # Recupère les matrices pour travailler
        mesh = self.mesh
        jacobien_e_pg = mesh.jacobien_e_pg
        poid_pg = mesh.poid_pg
        Nd_pg = np.array(mesh.N_mass_pg)
        Bd_e_pg = mesh.B_mass_e_pg

        # Partie qui fait intervenir le therme de reaction r
        Kdr_e_pg = np.einsum('ep,p,ep,pki,pkj->epij', jacobien_e_pg, poid_pg, r_e_pg, Nd_pg, Nd_pg, optimize=True)

        # Partie qui fait intervenir le therme de diffusion K
        KdK_e_pg = np.einsum('ep,p,,epki,epkj->epij', jacobien_e_pg, poid_pg, K, Bd_e_pg, Bd_e_pg, optimize=True)

        Kd_e_pg = Kdr_e_pg+KdK_e_pg
        
        Kd_e = np.sum(Kd_e_pg, axis=1)

        # Construit Fd_e
        Energie_e_pg = 2*PsiP_e_pg

        Fd_e_pg = np.einsum('ep,p,ep,pji->epij', jacobien_e_pg, poid_pg, Energie_e_pg, Nd_pg) 

        Fd_e = np.sum(Fd_e_pg, axis=1)

        tic.Tac("Matrices","Calcul des matrices elementaires (endommagement)", self.__verbosity)

        return Kd_e, Fd_e

    def Assemblage_d(self, Gc=1, l=0.001):
        """Construit Kglobal pour le probleme d'endommagement
        """
       
        # Data
        mesh = self.mesh
        taille = mesh.Nn
        lignesScalar_e = mesh.lignesScalar_e
        colonnesScalar_e = mesh.colonnesScalar_e
        
        # Calul les matrices elementaires
        Kd_e, Fd_e = self.__ConstruitMatElem_Pfm(Gc, l)

        # Assemblage
        tic = TicTac()        

        self.__Kd = sp.sparse.csr_matrix((Kd_e.reshape(-1), (lignesScalar_e.reshape(-1), colonnesScalar_e.reshape(-1))), shape = (taille, taille))
        """Kglob pour le probleme d'endommagement (Nn, Nn)"""
        
        lignes = mesh.connect.reshape(-1)
        self.__Fd = sp.sparse.csr_matrix((Fd_e.reshape(-1), (lignes,np.zeros(len(lignes)))), shape = (taille,1))
        """Fglob pour le probleme d'endommagement (Nn, 1)"""        

        tic.Tac("Matrices","Assemblage du systême en endommagement", self.__verbosity)       

        return self.__Kd, self.__Fd
    
    def Solve_d(self, resolution=2):
        """Resolution du problème d'endommagement"""
         
        tic = TicTac()

        dGlob = self.__Solveur(vector=False, resolution=resolution)

        tic.Tac("Résolution endommagement","Résolution {} pour le problème de endommagement".format(resolution) , self.__verbosity)
        
        # assert dGlob.max() <= 1, "Doit etre inférieur a 1"
        # assert dGlob.min() >= 0, "Doit etre supérieur 0"

        if(dGlob.max() > 1):
            print("dmax = {}".format(dGlob.max()))

        if(dGlob.min() < 0):
            print("dmin = {}".format(dGlob.min()))

        return dGlob

# ------------------------------------------------- SOLVEUR -------------------------------------------------

    def __Construit_ddl_connues_inconnues(self, vector: bool):
        """Récupère les ddl Connues et Inconnues

        Args:
            vector (bool, optional): Travail sur un vecteur ou un scalaire. Defaults to True.

        Returns:
            list(int), list(int): ddl_Connues, ddl_Inconnues
        """

        taille = self.mesh.Nn

        if vector :
            taille = taille*self.__dim
            ddl_Connues = self.__BC_Dirichlet_u[0]
        else:                
            ddl_Connues = self.__BC_Dirichlet_d[0]

        ddl_ConnuesNouveau = []
        for ddl in ddl_Connues:
            if ddl not in ddl_ConnuesNouveau:
                ddl_ConnuesNouveau.append(ddl)
        
        ddl_Connues = ddl_ConnuesNouveau

        ddl_Inconnues = list(range(taille))
        for ddl in ddl_Connues: ddl_Inconnues.remove(ddl)

        assert len(ddl_Connues) + len(ddl_Inconnues) == taille, "Problème dans les conditions"

        return ddl_Connues, ddl_Inconnues

    def __Application_Conditions_Neuman(self, vector: bool):
        """applique les conditions de Neumann"""

        taille = self.mesh.Nn

        if vector :
            taille = taille*self.__dim
            BC_Neuman = np.array(self.__BC_Neuman_u).T
        else:                
            BC_Neuman = np.array(self.__BC_Neuman_d).T

        # Renseigne les conditions de Neuman
        lignes = BC_Neuman[:,0]
        valeurs = BC_Neuman[:,1]
        b = sp.sparse.csr_matrix((valeurs, (lignes,  np.zeros(len(lignes)))), shape = (taille,1))

        if vector:
            b = b + self.__Fu.copy()
        else:
            b = b + self.__Fd.copy()

        return b

    def __Application_Conditions_Dirichlet(self, vector: bool, b, resolution):
        """applique les conditions de dirichlet"""
        
        taille = self.mesh.Nn

        if vector :
            taille = taille*self.__dim
            BC_Dirichlet = np.array(self.__BC_Dirichlet_u).T
            A = self.__Ku.copy()
        else:                
            BC_Dirichlet = np.array(self.__BC_Dirichlet_d).T
            A = self.__Kd.copy()

        # Renseigne les conditions de Dirichlet

        lignes = BC_Dirichlet[:,0].astype('int')
        valeurs = BC_Dirichlet[:,1]

        if resolution == 1:
            
            A = A.tolil()
            b = b.tolil()            
            
            # Pénalisation A
            A[lignes] = 0.0
            A[lignes, lignes] = 1

            # Pénalisation b
            b[lignes] = valeurs

            return A.tocsr(), b.tocsr()

        else:
            
            x = sp.sparse.csr_matrix((valeurs, (lignes,  np.zeros(len(lignes)))), shape = (taille,1), dtype=np.float64)

            return A, x

    def __Solveur(self, vector: bool, resolution=2):
        """Resolution du système matricielle A * x = b

        Args:
            vector (bool): Système vectorielle ou sclaire
            resolution (int, optional): Type de résolution. Defaults to 2.
        resolution=[1,2,3] -> [Pénalisation, Décomposition, FreeMatrix]

        Returns:
            nd.array: Renvoie la solution (x)
        """

        # Résolution du plus rapide au plus lent 2, 3, 1
        if resolution == 1:
            # Résolution par la méthode des pénalisations

            # Construit le système matricielle pénalisé
            b = self.__Application_Conditions_Neuman(vector)
            A, b = self.__Application_Conditions_Dirichlet(vector, b, resolution)

            # Résolution du système matricielle pénalisé
            x = spsolve(A, b)

        elif resolution == 2:
            
            # Construit le système matricielle
            b = self.__Application_Conditions_Neuman(vector)
            A, x = self.__Application_Conditions_Dirichlet(vector, b, resolution)

            # Récupère les ddls
            ddl_Connues, ddl_Inconnues = self.__Construit_ddl_connues_inconnues(vector)

            # Décomposition du système matricielle en connues et inconnues 
            # Résout : Aii * ui = bi - Aic * xc
            Aii = A[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()
            Aic = A[ddl_Inconnues, :].tocsc()[:, ddl_Connues].tocsr()
            bi = b[ddl_Inconnues,0]
            xc = x[ddl_Connues,0]
            
            vector=False

            if vector:

                from scipy.sparse.linalg import cholesky

                L = cholesky(Aii, lower=True)
                LT = L.T

                x_chol =  spsolve(L, bi-Aic.dot(xc))

                xi =  spsolve(LT, x_chol)

            else:
                # from scikits.umfpack import umfpack
                # from scikits.umfpack import spsolve
                # sp.sparse.linalg.use_solver(useUmfpack=True)

                # Résolution du sytème sur les ddl inconnues
                xi = spsolve(Aii, bi-Aic.dot(xc), use_umfpack=True)            
                # xi = spsolve(Aii, bi-Aic.dot(xc))


            # Reconstruction de la solution
            x = x.toarray().reshape(x.shape[0])
            x[ddl_Inconnues] = xi

        elif resolution == 3:

            # Récupère les constantes
            b = self.__Application_Conditions_Neuman(vector)
            A, x = self.__Application_Conditions_Dirichlet(resolution, b)

            # Récupère les ddls
            ddl_Connues, ddl_Inconnues = self.__Construit_ddl_connues_inconnues(vector)

            # Construit le système matricielle libéré
            Aii = A.tocsr()[ddl_Inconnues, :].tocsc()[:, ddl_Inconnues].tocsr()
            bi = b.tocsr()[ddl_Inconnues]

            # Résolution du sytème
            xi = spsolve(Aii, bi)

            # Reconstruction de la solution
            x = x.toarray().reshape(x.shape[0])
            x[ddl_Inconnues] = xi

        return x


# ------------------------------------------- CONDITIONS LIMITES ------------------------------------------- 

    def Clear_Condition_Neuman(self, option="u"):
        """Enlève les conditions limites de Neumann"""
        if option == "u":
            self.__BC_Neuman_d = [[],[]]
        else:
            self.__BC_Neuman_d = [[],[]]

    def Condition_Neumann(self, noeuds: list, directions: list, valeur=0.0, option="u"):
        """Applique les conditions en force

        Parameters
        ----------
        noeuds : list, optional
            Liste de int, by default []
        force : float, optional
            Force que l'on veut appliquer aux noeuds, by default 0.0
        directions : list, optional
            ["x", "y", "z"] vecteurs sur lesquelles on veut appliquer la force , by default [] 
        """

        tic = TicTac()
        
        assert option in ["u", "d"], "Mauvaise option"

        noeuds = np.array(noeuds)
        nbn = len(noeuds)

        if option == "d":
            assert len(directions) == 0, "lorsque on renseigne d on a pas besoin de direction"
            assert not valeur == 0.0, "Doit être différent de 0"

            if noeuds not in self.__BC_Neuman_d[0]:
                self.__BC_Neuman_d[0].extend(noeuds)
                self.__BC_Neuman_d[1].extend([valeur/nbn]*nbn)                

        elif option == "u":
            assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"
            ddl = []
            for direction in directions:
                assert direction in ["x", "y", "z"] , "direction doit etre x y ou z"
                if direction == "x":
                    ddl.extend(noeuds * self.__dim)
                if direction == "y":
                    ddl.extend(noeuds * self.__dim + 1)
                if direction == "z":
                    assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                    ddl.extend(noeuds * self.__dim + 2)

        for d in ddl: assert d not in self.__BC_Dirichlet_u[0], "Impossible d'appliquer un déplacement et un effort au meme noeud"        
        self.__BC_Neuman_u[0].extend(ddl)
        self.__BC_Neuman_u[1].extend([valeur/nbn]*len(ddl))
        
        tic.Tac("Boundary Conditions","Condition Neumann", self.__verbosity)

    def Clear_Condition_Dirichlet(self, option="u"):
        """Enlève les conditions limites de Dirichlet"""
        if option == "u":
            self.__BC_Dirichlet_u = [[],[]]
        else:
            self.__BC_Dirichlet_d = [[],[]]

    def Condition_Dirichlet(self, noeuds: np.ndarray, directions=[] , valeur=0.0, option="u"):
        
        # assert isinstance(noeuds[0], int), "Doit être une liste d'indices"        
        assert option in ["u", "d"], "Mauvaise option"

        tic = TicTac()
        
        noeuds = np.array(noeuds)
        nbn = len(noeuds)

        if option == "d":
            assert len(directions) == 0, "lorsque on renseigne d on a pas besoin de direction"
            assert valeur >= 0 or valeur <= 1, "d doit etre compris entre [0;1]"
           
            if noeuds not in self.__BC_Dirichlet_d[0]:
                self.__BC_Dirichlet_d[0].extend(noeuds)
                self.__BC_Dirichlet_d[1].extend([valeur]*nbn)      

        elif option == "u":
            assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"
            ddl = []
            for direction in directions:
                assert direction in ["x", "y", "z"] , "direction doit etre x y ou z"
                if direction == "x":
                    ddl.extend(noeuds * self.__dim)
                if direction == "y":
                    ddl.extend(noeuds * self.__dim + 1)
                if direction == "z":
                    assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                    ddl.extend(noeuds * self.__dim + 2)

            for d in ddl: assert d not in self.__BC_Dirichlet_u[0], "Impossible d'appliquer un déplacement et un effort au meme noeud"            
            self.__BC_Dirichlet_u[0].extend(ddl)
            self.__BC_Dirichlet_u[1].extend([valeur]*len(ddl))

        tic.Tac("Boundary Conditions","Condition Dirichlet", self.__verbosity)

    
# ------------------------------------------- POST TRAITEMENT ------------------------------------------- 
    
    def __Save_u(self, Uglob, calculContraintesEtDeformation, interpolation=False, verification=False):
        
        tic = TicTac()

        # Energie de deformation
        u_e = self.mesh.Localise_e(Uglob)        
        Ke_e = self.__Ku_e
        Wdef = 1/2 * np.einsum('ei,eij,ej->', u_e, Ke_e, u_e)
        
        self.resultats["Wdef"] = Wdef

        if verification:
            Kglob = self.__Ku.todense()
            WdefVerif = 1/2 * Uglob.T.dot(Kglob).dot(Uglob)
            tic.Tac("Post Traitement","Wdef verif", True)
            diff = np.round(float(WdefVerif) - Wdef,0)
            assert np.isclose(diff, 0),"Erreur"

        # Récupère les déplacements
        ddlx = np.arange(0, self.mesh.Nn*self.__dim, self.__dim)
        ddly = ddlx + 1 
        ddlz = ddlx + 2
        
        dx = Uglob[ddlx]
        dy = Uglob[ddly]
        if self.__dim == 2:
            dz = np.zeros(self.mesh.Nn)
        else:
            dz = Uglob[ddlz]
                
        self.resultats["dx_n"] = dx
        self.resultats["dy_n"] = dy        
        if self.__dim == 3:
            self.resultats["dz_n"] = dz
            
        self.resultats["amplitude"] = np.sqrt(dx**2+dy**2+dz**2)        

        self.resultats["deplacementCoordo"] = np.array([dx, dy, dz]).T

        if calculContraintesEtDeformation:
            self.__CalculDeformationEtContrainte(Uglob, interpolation=interpolation)

        tic.Tac("Post Traitement","Sauvegarde", self.__verbosity)
            
    def __CalculDeformationEtContrainte(self, Uglob: np.ndarray, interpolation=False):
        """Calcul les deformations et contraints sur chaque element
        Si interpolation == True on fait l'interpolation aux noeuds

        Args:
            u_e (ndarray): déplacement pour chaque noeud de l'element [dx1 dy1 dx2 dy2 ... dxn dyn]
            interpolation (bool, optional): Interpolation aux noeuds. Defaults to False.

        Returns:
            [type]: [description]
        """

        # Data
        u_e = self.mesh.Localise_e(Uglob)

        # Deformation et contraintes pour chaque element et chaque points de gauss        
        Epsilon_e_pg = self.__CalculEpsilon_e_pg(Uglob)
        Sigma_e_pg = self.__CalculSigma_e_pg(Epsilon_e_pg)        

        # Moyenne sur l'élement
        Epsilon_e = np.mean(Epsilon_e_pg, axis=1)
        Sigma_e = np.mean(Sigma_e_pg, axis=1)


        # On récupère les déplacements sur chaque element
        listCoordX = np.array(range(0, u_e.shape[1], self.__dim))
        dx_e_n = u_e[:,listCoordX]; dx_e = np.mean(dx_e_n, axis=1)
        dy_e_n = u_e[:,listCoordX+1]; dy_e = np.mean(dy_e_n, axis=1)
        if self.__dim == 3:
            dz_e_n = u_e[:,listCoordX+2]; dz_e = np.mean(dz_e_n, axis=1)

        # On récupère les deformations et contraintes puis on calcul la contraint de von Mises
        if self.__dim == 2 :
            Exx_e = Epsilon_e[:,0]; Sxx_e = Sigma_e[:,0]
            Eyy_e = Epsilon_e[:,1]; Syy_e = Sigma_e[:,1]
            Exy_e = Epsilon_e[:,2]; Sxy_e = Sigma_e[:,2]
            
            Svm_e = np.sqrt(Sxx_e**2+Syy_e**2-Sxx_e*Syy_e+3*Sxy_e**2)
        else :
            Exx_e = Epsilon_e[:,0]; Sxx_e = Sigma_e[:,0]
            Eyy_e = Epsilon_e[:,1]; Syy_e = Sigma_e[:,1]
            Ezz_e = Epsilon_e[:,2]; Szz_e = Sigma_e[:,2]
            Eyz_e = Epsilon_e[:,3]/2; Syz_e = Sigma_e[:,3]
            Exz_e = Epsilon_e[:,4]/2; Sxz_e = Sigma_e[:,4]
            Exy_e = Epsilon_e[:,5]/2; Sxy_e = Sigma_e[:,5]

            Svm_e = np.sqrt(((Sxx_e-Syy_e)**2+(Syy_e-Szz_e)**2+(Szz_e-Sxx_e)**2+6*(Sxy_e**2+Syz_e**2+Sxz_e**2))/2)

        # On stock les données dans resultats
        self.resultats["dx_e"] = dx_e; self.resultats["dy_e"] = dy_e
        self.resultats["Exx_e"] = Exx_e; self.resultats["Eyy_e"] = Eyy_e; self.resultats["Exy_e"] = Exy_e
        self.resultats["Sxx_e"] = Sxx_e; self.resultats["Syy_e"] = Syy_e; self.resultats["Sxy_e"] = Sxy_e
        self.resultats["Svm_e"] = Svm_e
        if self.__dim == 3:
            self.resultats["dz_e"] = dz_e
            self.resultats["Ezz_e"] = Ezz_e; self.resultats["Eyz_e"] = Eyz_e; self.resultats["Exz_e"] = Exz_e
            self.resultats["Szz_e"] = Szz_e; self.resultats["Syz_e"] = Syz_e; self.resultats["Sxz_e"] = Sxz_e
        
        if interpolation:
            self.__InterpolationAuxNoeuds()

        return Epsilon_e, Sigma_e 
    
    def __InterpolationAuxNoeuds(self, verification=False):
        """Pour chaque noeuds on récupère les valeurs des élements autour de lui pour on en fait la moyenne

        Returns:
            [type]: [description]
        """

        tic = TicTac()

        # Data
        listNoeud = list(range(self.mesh.Nn))
        listElement = list(range(self.mesh.Ne))
        connect = self.mesh.connect

        # Consruit la matrice de connection Noeud
        # # connecNoeud = np.zeros((self.__mesh.Nn, self.__mesh.Ne))
        # Ici l'objectif est de construire une matrice qui lorsque quon va la multiplier a un vecteur valeurs_e de taille ( Ne x 1 ) va donner
        # valeurs_n_e(Nn,1) = connecNoeud(Nn,Ne) valeurs_n_e(Ne,1)
        # ou connecNoeud(Nn,:) est un vecteur ligne composé de 0 et de 1 qui permetra de sommer valeurs_e[noeuds]
        # Ensuite, il suffit juste par divisier par le nombre de fois que le noeud apparait dans la ligne
        # Il faut encore optimiser la façon dont jobtient connecNoeud
        # L'idéal serait dobtenir connectNoeud (Nn x nombre utilisation du noeud par element) rapidement
        # Construit connect des noeuds rapidement
        connect_n_e = sp.sparse.lil_matrix((self.mesh.Nn, self.mesh.Ne))

        # connectNoeuds = [[e for e in listElement if n in self.connect[e]] for n in listNoeud]

        connectNoeud = [list(np.where(n == connect)[0]) for n in listNoeud]
        # connectNoeud = np.array([np.where(n == connect)[0] for n in listNoeud])
        
        # colonnes = np.ravel(connectNoeud, axis=0)
        # connect_n_e = sp.sparse.csr_matrix((1, (listNoeud, colonnes)), shape = (self.__mesh.Nn, self.__mesh.Ne)).tolil()
        for n in listNoeud:
            connect_n_e[n, connectNoeud[n]] = 1
        # connect_n_e[listNoeud, connectNoeud] = 1

        # tic.Tac("Temps boucle", True)

        connect_n_e.tocsr()
        nombreApparition = np.array(1/np.sum(connect_n_e, axis=1)).reshape(self.mesh.Nn,1)

        # Fonction d'interpolation
        def InterPolation(valeurs_e):

            valeurs_n_e = connect_n_e.dot(valeurs_e.reshape(self.mesh.Ne,1))
            valeurs_n = valeurs_n_e*nombreApparition
            
            if verification:
                connectNoeudVerif = np.array([np.where(n == connect)[0] for n in listNoeud])
                valeurs_n2 = [np.mean(valeurs_e[connectNoeudVerif[n]]) for n in listNoeud]
                test = np.round(np.array(valeurs_n2).reshape(self.mesh.Nn,1) - valeurs_n)
                assert test.mean() == 0, "Erreur dans l'interpolation"

            return valeurs_n.reshape(-1)

        self.resultats["Exx_n"] = InterPolation(self.resultats["Exx_e"]); self.resultats["Eyy_n"] = InterPolation(self.resultats["Eyy_e"]); self.resultats["Exy_n"] = InterPolation(self.resultats["Exy_e"])
        self.resultats["Sxx_n"] = InterPolation(self.resultats["Sxx_e"]); self.resultats["Syy_n"] = InterPolation(self.resultats["Syy_e"]); self.resultats["Sxy_n"] = InterPolation(self.resultats["Sxy_e"])
        self.resultats["Svm_n"] = InterPolation(self.resultats["Svm_e"])

        if self.__dim == 3:            
            self.resultats["Ezz_n"] = InterPolation(self.resultats["Ezz_e"]); self.resultats["Eyz_n"] = InterPolation(self.resultats["Eyz_e"]); self.resultats["Exz_n"] = InterPolation(self.resultats["Exz_e"])
            self.resultats["Szz_n"] = InterPolation(self.resultats["Szz_e"]); self.resultats["Syz_n"] = InterPolation(self.resultats["Syz_e"]); self.resultats["Sxz_n"] = InterPolation(self.resultats["Sxz_e"])

        tic.Tac("Post Traitement","Calcul deformations et contraintse aux noeuds", self.__verbosity)

    def Resume(self):
        print("\nW def = {:.6f} N.mm".format(self.resultats["Wdef"])) 

        print("\nSvm max = {:.6f} MPa".format(np.max(self.resultats["Svm_e"]))) 

        print("\nUx max = {:.6f} mm".format(np.max(self.resultats["dx_n"]))) 
        print("Ux min = {:.6f} mm".format(np.min(self.resultats["dx_n"]))) 

        print("\nUy max = {:.6f} mm".format(np.max(self.resultats["dy_n"]))) 
        print("Uy min = {:.6f} mm".format(np.min(self.resultats["dy_n"])))

        if self.__dim == 3:
            print("\nUz max = {:.6f} mm".format(np.max(self.resultats["dz_n"])))
            print("Uz min = {:.6f} mm".format(np.min(self.resultats["dz_n"])))

# ====================================

import unittest
import os

class Test_Simu(unittest.TestCase):    
    
    def CreationDesSimusElastique2D(self):
        
        dim = 2

        # Paramètres géométrie
        L = 120;  #mm
        h = 13;    
        b = 13

        # Charge a appliquer
        P = -800 #N

        # Paramètres maillage
        taille = L

        comportement = Elas_Isot(dim)

        materiau = Materiau(comportement)

        self.simulations2DElastique = []        

        # Pour chaque type d'element 2D
        for i in range(len(Element.get_Types2D())):
            # Construction du modele et du maillage 
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=i, tailleElement=taille, verbosity=False)

            (coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim, mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

            simu.Condition_Neumann(noeuds=noeuds_en_L, valeur=P, directions=["y"])

            simu.Condition_Dirichlet(noeuds=noeuds_en_0, valeur=0, directions=["x", "y"])

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
        taille = L

        comportement = Elas_Isot(dim)

        materiau = Materiau(comportement)
        
        self.simulations3DElastique = []        

        for i in range(len(Element.get_Types3D())):
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=i, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False, verbosity=False)

            (coordo, connect) = modelGmsh.Importation3D(fichier)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim,mesh, materiau, verbosity=False)

            simu.Assemblage_u()

            noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
            noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

            simu.Condition_Neumann(noeuds=noeuds_en_L, valeur=P, directions=["z"])

            simu.Condition_Dirichlet(noeuds=noeuds_en_0, valeur=0, directions=["x", "y", "z"])

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

            # Data
            mesh = cast(Mesh, simu.mesh)
            nPg = len(mesh.poid_pg)
            listPg = list(range(nPg))
            listElement = simu.listElement
            materiau = simu.materiau
            C = materiau.comportement.get_C()

            listKe_e = []

            for e in listElement:            
                # Pour chaque poing de gauss on construit Ke
                Ke = 0
                for pg in listPg:
                    jacobien = mesh.jacobien_e_pg[e][pg]
                    poid = mesh.poid_pg[pg]
                    B_pg = mesh.B_rigi_e_pg[e][pg]

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

