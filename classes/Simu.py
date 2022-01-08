from ctypes import memset
import os
from typing import cast

import numpy as np
from numpy import matrixlib, zeros
from numpy.core.records import array
from numpy.random.mtrand import rand
import scipy as sp
from scipy.sparse.linalg import spsolve

try:
    from classes.ModelGmsh import ModelGmsh
    from classes.Element import Element
    from classes.Mesh import Mesh
    from classes.Materiau import Materiau
    from classes.TicTac import TicTac
except:
    from ModelGmsh import ModelGmsh
    from Element import Element
    from Mesh import Mesh
    from Materiau import Materiau
    from TicTac import TicTac

class Simu:
    
    def get_listElement(self):
        return list(range(self.__mesh.Ne))        
    listElement = property(get_listElement)

    def __init__(self, dim: int,mesh: Mesh, materiau: Materiau, verbosity=True):
        """Creation d'une simulation

        Parameters
        ----------
        dim : int
            Dimension de la simulation 2D ou 3D
        verbosity : bool, optional
            La simulation ecrit tout les details de commande dans la console, by default False
        """
        
        # Vérification des valeurs
        assert dim == 2 or dim == 3, "Dimesion compris entre 2D et 3D"
        assert isinstance(mesh, Mesh) and mesh.get_dim() == dim, "Doit etre un maillage et doit avoir la meme dimension que dim"
        assert isinstance(materiau, Materiau) and materiau.get_dim() == dim, "Doit etre un materiau et doit avoir la meme dimension que dim"


        self.__dim = dim      
        self.__verbosity = verbosity        
        self.__mesh = mesh        
        self.__materiau = materiau        
        self.resultats = {}
        self.list_H = []

        # Conditions Limites
        self.__BC_Neuman_u = [[],[]]
        self.__BC_Dirichlet_u = [[],[]]
        self.__BC_Neuman_d = [[],[]]
        self.__BC_Dirichlet_d = [[],[]]


    
    def __Construit_Ke(self, epaisseur, d, verification=False):

        mesh = cast(Mesh, self.__mesh)
        nPg = len(mesh.poid_pg)
        listPg = list(range(nPg))
        listElement = self.listElement
        
        # Calcul Ke        
        ticKe = TicTac()
        jacobien = mesh.jacobien_e_pg
        poid = mesh.poid_pg
        B_rigi = mesh.B_rigi_e_pg
        mat = self.__materiau.C
        Ke_e_pg = [[jacobien[e,pg]*poid[pg]*B_rigi[e,pg].T.dot(mat).dot(B_rigi[e,pg]) for pg in listPg] for e in listElement]

        if len(d) !=0 :   # probleme endomagement            
            print("Non implémenté")
            # # Bourdin
            # g = (1-mesh.N_mass_pg[pg].dot(np.array([d[mesh.connect[e]]])))**2

            # Ke_e_pg = Ke_e_pg[e][pg]*g
        
        Ke_e = np.sum(Ke_e_pg, axis=1)
        
        if self.__dim == 2:
            Ke_e = epaisseur * Ke_e

        ticKe.Tac("Calcul des matrices elementaires", self.__verbosity)

        if verification:
            listKe_e = []
            for e in listElement:            
                # Pour chaque poing de gauss on construit Ke
                Ke = 0
                for pg in listPg:
                    jacobien = mesh.jacobien_e_pg[e][pg]
                    poid = mesh.poid_pg[pg]
                    B_pg = mesh.B_rigi_e_pg[e][pg]

                    K = jacobien * poid * B_pg.T.dot(self.__materiau.C).dot(B_pg)

                    if len(d)==0:   # probleme standart
                        
                        Ke += K
                    else:   # probleme endomagement
                        
                        de = np.array([d[mesh.connect[e]]])
                        
                        # Bourdin
                        g = (1-mesh.N_mass_pg[pg].dot(de))**2
                        # g = (1-de)**2
                        
                        Ke += g * K
                # # print(Ke-listeKe[e.id])
                if self.__dim == 2:
                    listKe_e.append(epaisseur * Ke)
                else:
                    listKe_e.append(Ke)                

            ticKe.Tac("Calcul des matrices elementaires (boucle)", True)
            
            # Verification
            Ke_comparaison = np.array(listKe_e)
            test = Ke_e - Ke_comparaison           
            assert test.max() == 0 and test.min() == 0, "Problème"
        
        return Ke_e

    def Assemblage_u(self, epaisseur=1, d=[], verification=False):
        """Construit Kglobal

        mettre en option u ou d ?

        """

        if self.__dim == 2:        
            assert epaisseur>0,"Doit être supérieur à 0"

        tic = TicTac()

        # Construit Ke
        Ke_e = self.__Construit_Ke(epaisseur, d)
        self.__Ke_e = Ke_e # Sauvegarde Ke pour calculer Energie plus rapidement
        
        # Assemblage
        mesh = self.__mesh
        listElement = self.listElement
        taille = mesh.Nn*self.__dim
        
        # Matrices du système
        self.__Ku = sp.sparse.lil_matrix((taille, taille))  # self.__Ku = np.zeros((taille, taille))
        self.__Fu = sp.sparse.lil_matrix((taille,1))    # self.__Fu = np.zeros(taille)
        self.__Uc = sp.sparse.lil_matrix((taille,1))
        

        lignes_e = np.array([[i for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        colonnes_e = np.array([[j for i in mesh.assembly_e[e] for j in mesh.assembly_e[e]] for e in listElement])
        
        # V0 Plus rapide
        taille = self.__mesh.Nn*self.__dim
        self.__Ku = sp.sparse.csr_matrix((Ke_e.reshape(-1), (lignes_e.reshape(-1), colonnes_e.reshape(-1))), shape = (taille, taille)).tolil()

        if verification: self.__AssembleMatrice(lignes_e, colonnes_e, Ke_e)
        
        self.__Ku_penal = self.__Ku.copy()
        self.__Fu_penal = self.__Fu.copy()

        tic.Tac("Assemblage du syteme en déplacement", self.__verbosity)



    def Assemblage_d(self, Gc=1, l=0.001):
        """Construit Kglobal

        mettre en option u ou d ?

        """

        ticAssemblage = TicTac()
        
        taille = self.__mesh.Nn

        self.__Kd = np.zeros((taille, taille))
        self.__Fd = np.zeros(taille)
        
        self.__d_Inconnues = [i for i in range(self.__mesh.Nn)]
        self.__d_Connues = []
        self.__dc = np.zeros(taille)
        
        for e in self.__mesh.elements:            
            e = cast(Element, e)
            
            Ke = 0
            fe = 0
            # Pour chaque point de gauss construit Ke et fe
            for pg in range(len(e.listJacobien_pg)):
                
                jacobien = e.listJacobien_pg[pg]
                poid = e.listPoid_pg[pg]
                h = self.list_H[e.id][pg]
                Nd = e.listN_d_pg[pg]
                Bd = e.listB_d_pg[pg]
                
                nTn = Nd.T.dot(Nd)
                bTb = Bd.T.dot(Bd)

                Ke += jacobien * poid * ((Gc/l+2*h) * nTn + Gc * l * bTb)

                fe += jacobien * poid * 2 * Nd.T * h
                
            list_IdNoeuds = [n.id for n in e.noeuds]

            # Assemblage Kd_glob et Fd_glob
            lignes = []
            colonnes = []
            for i in list_IdNoeuds:
                lignes.extend(list_IdNoeuds)
                for j in range(e.nPe):
                    colonnes.append(i)

            self.__Kd[lignes, colonnes] += np.ravel(Ke)
            self.__Fd[list_IdNoeuds] += np.ravel(fe)            
            
        ticAssemblage.Tac("Assemblage d", self.__verbosity)

        self.__Kd_penal = np.copy(self.__Kd)
        self.__Fd_penal = np.copy(self.__Fd)

        return self.__Kd, self.__Fd

    def __AssembleMatrice(self,lignes_e, colonnes_e, valeurs_e):
        
        lignes = np.ravel(lignes_e)
        colonnes = np.ravel(colonnes_e)
        valeurs = valeurs_e.reshape(-1)
        
        indincesOrdo = np.lexsort((colonnes, lignes))

        coord = np.array([lignes,colonnes]).T
        coord = coord[indincesOrdo]

        lignesRaveldSorted = lignes[indincesOrdo]
        colonnesRaveldSorted = colonnes[indincesOrdo]        
        KeRaveldSorted = valeurs[indincesOrdo]

        ticVersion = TicTac()
        taille = self.__mesh.Nn*self.__dim
        Ku = sp.sparse.lil_matrix((taille, taille))
        version = 0

        if version == 0:
            # V0 Plus rapide            
            Ku = sp.sparse.lil_matrix(sp.sparse.csr_matrix((KeRaveldSorted, (lignesRaveldSorted, colonnesRaveldSorted)), shape = (taille, taille)))
        elif version == 1:
            # V1            
            for i in range(len(indincesOrdo)-1):
                if i+1 != coord.shape[0] and (coord[i][0] == coord[i+1][0] and coord[i][1] == coord[i+1][1]):                
                    KeRaveldSorted[i+1] += KeRaveldSorted[i]
                    KeRaveldSorted[i]=0                
            Ku[lignesRaveldSorted, colonnesRaveldSorted] += KeRaveldSorted
        elif version == 2:
            # # V2
            # Il faut d'abord réussir à construire la liste suivante sans boucle !
            listIndices = np.array([i for i in range(len(indincesOrdo)-1) if i+1 != coord.shape[0] and (coord[i,0] == coord[i+1,0] and coord[i,1] == coord[i+1,1])])
            # Construit la liste sans la boucle
            unique, unique_indices, unique_inverse, unique_counts  = np.unique(coord,axis=0,return_index=True, return_inverse=True, return_counts=True)
            list_i = np.array(range(len(unique_inverse)-1))        
            listIndicesRapide = np.where(unique_inverse[list_i] == unique_inverse[list_i+1])[0]
            # Verification que la liste est bien construite
            assert np.sum(listIndices - listIndicesRapide)==0,"Erreur dans la construction de la liste" 
            # Somme des valeurs pour les coordonnées identiques
            for i in listIndicesRapide:
                KeRaveldSorted[i+1] += KeRaveldSorted[i]
                KeRaveldSorted[i] = 0
            # KeRaveldSorted[listIndicesRapide+1] = KeRaveldSorted[listIndicesRapide+1] + KeRaveldSorted[listIndicesRapide]
            # KeRaveldSorted[listIndicesRapide] = 0
            # Assemblage
            Ku[lignesRaveldSorted, colonnesRaveldSorted] += KeRaveldSorted
        elif version == 3:
            # V3
            unique, unique_indices, unique_inverse, unique_counts  = np.unique(coord,axis=0,return_index=True, return_inverse=True, return_counts=True)
            list_i = np.array(range(len(unique_inverse)-1))        
            listIndicesRapide = np.flipud(np.where(unique_inverse[list_i] == unique_inverse[list_i+1])[0])
            # V3.1
            for i in listIndicesRapide:
                KeRaveldSorted[i] += KeRaveldSorted[i+1]
                KeRaveldSorted[i+1] = 0
            # # V3.2
            # KeRaveldSorted[listIndicesRapide] += KeRaveldSorted[listIndicesRapide+1]
            # KeRaveldSorted[listIndicesRapide+1] = 0
            taille = self.__mesh.Nn*self.__dim
            Ku = sp.sparse.lil_matrix(sp.sparse.csr_matrix((KeRaveldSorted[unique_indices], (unique[:,0],unique[:,1])), shape = (taille, taille)))
       
        ticVersion.Tac("Assemblage version {}".format(version), True)

        # Verification de l'assemblage
        ticVerification = TicTac()
        taille = self.__mesh.Nn*self.__dim
        mesh = self.__mesh
        listElement = self.listElement
        indices = range(0, valeurs_e[0].shape[0])

        Ku_comparaison = sp.sparse.lil_matrix((taille, taille))

        liste_ligne = []
        liste_colonne = []
        liste_Ke = []
        
        for e in listElement:
            for i in indices:
                ligne = mesh.assembly_e[e][i]
                for j in indices:
                    colonne = mesh.assembly_e[e][j]
                    Ku_comparaison[ligne, colonne] =  Ku_comparaison[ligne, colonne] + valeurs_e[e][i,j]
                    
                    liste_ligne.append(ligne)
                    liste_colonne.append(colonne)
                    liste_Ke.append(valeurs_e[e][i,j])
        
        # Tests
        test1 = np.array(liste_ligne) - lignes
        assert test1.max() == 0 and test1.min() == 0, "Erreur dans la liste d'assemblage"
        test2 = np.array(liste_colonne) - colonnes
        assert test2.max() == 0 and test2.min() == 0, "Erreur dans la liste d'assemblage"
        test3 = np.array(liste_Ke) - valeurs
        assert test3.max() == 0 and test3.min() == 0, "Erreur dans Ke_e Ravel"
        test4 = Ku_comparaison - self.__Ku
        assert test4.max() == 0 and test4.min() == 0, "Erreur dans l'assemblage"

        ticVerification.Tac("Assemblage lent avec verification", True)
    
    def Condition_Neumann(self, noeuds: np.ndarray, directions: list, valeur=0.0, option="u"):
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

        assert isinstance(noeuds[0], int), "Doit être une liste d'indices'"
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

        if ddl not in self.__BC_Neuman_u[0]:
            self.__BC_Neuman_u[0].extend(ddl)
            self.__BC_Neuman_u[1].extend([valeur/nbn]*len(ddl))    
        
        tic.Tac("Condition Neumann", self.__verbosity)

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

            if ddl not in self.__BC_Dirichlet_u[0]:
                self.__BC_Dirichlet_u[0].extend(ddl)
                self.__BC_Dirichlet_u[1].extend([valeur]*len(ddl))

        tic.Tac("Condition Dirichlet", self.__verbosity)

    def Solve_u(self, resolution=1, save=True):
        
        tic = TicTac()

        # Résolution du plus rapide au plus lent
        if resolution == 1:

            # Renseigne les conditions de Neumman
            
            BC_Neuman = np.array(self.__BC_Neuman_u).T
            BC_Dirichlet = np.array(self.__BC_Dirichlet_u).T

            self.__Fu_penal[BC_Neuman[:,0]] += BC_Neuman[:,1].reshape(BC_Neuman.shape[0],1)
            # Renseigne les conditions de Dirichlet
            self.__Fu_penal[BC_Dirichlet[:,0]] = BC_Dirichlet[:,1]
            self.__Ku_penal[BC_Dirichlet[:,0]] = 0.0
            self.__Ku_penal[BC_Dirichlet[:,0], BC_Dirichlet[:,0]] = 1


            Uglob = sp.sparse.linalg.spsolve(self.__Ku_penal.tocsr(), self.__Fu_penal.tocsr())

        elif resolution == 2:

            taille = self.__mesh.Nn*self.__dim
            ddl_Connues = []
            ddl_Connues.extend(self.__BC_Dirichlet_u[0])
            # ddl_Connues.extend(self.__BC_Neuman_u[0])
            ddl_Inconnues = list(range(taille))
            for ddl in ddl_Connues: ddl_Inconnues.remove(ddl)

            
            assert len(ddl_Connues) + len(ddl_Inconnues) == taille, "Problème dans les conditions"

            Kligne = self.__Ku.tocsr()[ddl_Inconnues, :]

            Kii = Kligne.tocsc()[:, ddl_Inconnues].tocsr()
            Kic = Kligne.tocsc()[:, ddl_Connues].tocsr()
            Fi = self.__Fu[ddl_Inconnues].tocsr()

            uc = self.__Uc[ddl_Connues].tocsr()
            
            ui = sp.sparse.linalg.spsolve(Kii, Fi-Kic.dot(uc))
            
            Uglob = np.zeros(taille)
            
            Uglob[ddl_Inconnues] = ui
            uc = uc.toarray().reshape(-1)
            Uglob[ddl_Connues] = uc

        tic.Tac("Résolution {}".format(resolution) , self.__verbosity)        
        
        if save:
            self.__Save_u(Uglob)

        return Uglob

    def __Save_u(self, Uglob: np.ndarray, verification=False):
        
        tic = TicTac()

        mesh = self.__mesh
        listElement = self.listElement
        
        # Energie de deformation
        ue_e = np.array([Uglob[mesh.assembly_e[e]] for e in listElement])
        Ke_e = self.__Ke_e
        Wdef = 1/2*np.sum([ue_e[e].T.dot(Ke_e[e]).dot(ue_e[e]) for e in listElement])
        tic.Tac("Calcul de l'energie de deformation", verification)

        if verification:
            Kglob = self.__Ku.todense()
            WdefVerif = 1/2 * Uglob.T.dot(Kglob).dot(Uglob)
            tic.Tac("Wdef verif", True)
            diff = float(WdefVerif) - Wdef
            assert np.isclose(diff, 0,10**-6),"Erreur" 
        self.resultats["Wdef"] = Wdef

        # Récupère les déplacements
        ddlx = list(range(0, mesh.Nn*self.__dim, self.__dim))
        ddly = list(range(1, mesh.Nn*self.__dim, self.__dim))
        ddlz = list(range(2, mesh.Nn*self.__dim, self.__dim))

        dx = Uglob[ddlx]
        dy = Uglob[ddly]
        if self.__dim == 2:
            dz = np.zeros(self.__mesh.Nn)
        else:
            dy = Uglob[ddlz]
                
        self.resultats["dx_n"] = dx
        self.resultats["dy_n"] = dy        
        if self.__dim == 3:
            self.resultats["dz_n"] = dz

        self.resultats["deplacementCoordo"] = np.array([dx, dy, dz]).T
        
        # self.__CalculDeformationEtContrainte(Uglob)

        tic.Tac("Sauvegarde", self.__verbosity)
            

    def __CalculDeformationEtContrainte(self, Uglob: np.ndarray, calculAuxNoeuds=True):
        
        tic = TicTac()

        list_Epsilon_e = []
        list_Sigma_e = []
        dim = self.__dim
        
        # Prépare les vecteurs de stockage par element
        dx_e = []; dy_e = []
        Exx_e = []; Eyy_e = []; Exy_e = []
        Sxx_e = []; Syy_e = []; Sxy_e = []
        Svm_e = []

        if dim == 3:
            dz_e = []
            Ezz_e = []; Eyz_e = []; Exz_e = []
            Szz_e = []; Syz_e = []; Sxz_e = []

        # Pour chaque element on va calculer pour chaque point de gauss Epsilon et Sigma
        for e in range(self.__mesh.Ne):

            dx = []
            dy = []
            if dim == 3:
                dz = []

            # Construit ue
            ue = []
            for n in self.__mesh.connect[e]:
                
                for j in range(self.__dim):
                    valeur = Uglob[int(n)*self.__dim+j]
                    ue.append(valeur)
                    if j == 0:
                        dx.append(valeur)
                    if j == 1:
                        dy.append(valeur)
                    if j == 2:
                        dz.append(valeur)

            list_epsilon_pg = []
            list_sigma_pg = []

            # Récupère B pour chaque pt de gauss
            for B_pg in self.__mesh.B_rigi_e_pg[e]:
                epsilon_pg = B_pg.dot(ue)
                list_epsilon_pg.append(epsilon_pg)

                sigma_pg = self.__materiau.C.dot(list(epsilon_pg))
                list_sigma_pg.append(list(sigma_pg))

            list_epsilon_pg = np.array(list_epsilon_pg)
            list_sigma_pg = np.array(list_sigma_pg)

            list_Epsilon_e.append(list_epsilon_pg)
            list_Sigma_e.append(list_sigma_pg)

            if dim == 2:
                dx_e.append(np.mean(dx))
                dy_e.append(np.mean(dy))
                
                Exx_e.append(np.mean(list_epsilon_pg[:, 0])), Sxx_e.append(np.mean(list_sigma_pg[:, 0]))
                Eyy_e.append(np.mean(list_epsilon_pg[:, 1])), Syy_e.append(np.mean(list_sigma_pg[:, 1]))
                Exy_e.append(np.mean(list_epsilon_pg[:, 2])), Sxy_e.append(np.mean(list_sigma_pg[:, 2]))

                Svm_e.append(np.sqrt(Sxx_e[-1]**2+Syy_e[-1]**2-Sxx_e[-1]*Syy_e[-1]+3*Sxy_e[-1]**2))

            elif dim == 3:
                dz_e.append(np.mean(dz))

                Exx_e.append(np.mean(list_epsilon_pg[:, 0])), Sxx_e.append(np.mean(list_sigma_pg[:, 0]))
                Eyy_e.append(np.mean(list_epsilon_pg[:, 1])), Syy_e.append(np.mean(list_sigma_pg[:, 1]))
                Ezz_e.append(np.mean(list_epsilon_pg[:, 2])), Szz_e.append(np.mean(list_sigma_pg[:, 2]))
                Exy_e.append(np.mean(list_epsilon_pg[:, 3])), Sxy_e.append(np.mean(list_sigma_pg[:, 3]))
                Eyz_e.append(np.mean(list_epsilon_pg[:, 4])), Syz_e.append(np.mean(list_sigma_pg[:, 4]))
                Exz_e.append(np.mean(list_epsilon_pg[:, 5])), Sxz_e.append(np.mean(list_sigma_pg[:, 5]))

                Svm_e.append(np.sqrt(((Sxx_e[-1]-Syy_e[-1])**2+(Syy_e[-1]-Szz_e[-1])**2+(Szz_e[-1]-Sxx_e[-1])**2+6*(Sxy_e[-1]**2+Syz_e[-1]**2+Sxz_e[-1]**2))/2)) 

        self.resultats["dx_e"]=np.array(dx_e); self.resultats["dy_e"]=np.array(dy_e)
        self.resultats["Exx_e"]=np.array(Exx_e); self.resultats["Eyy_e"]=np.array(Eyy_e); self.resultats["Exy_e"]=np.array(Exy_e)
        self.resultats["Sxx_e"]=np.array(Sxx_e); self.resultats["Syy_e"]=np.array(Syy_e); self.resultats["Sxy_e"]=np.array(Sxy_e)
        self.resultats["Svm_e"]=np.array(Svm_e)

        if dim == 3:
            self.resultats["dz_e"]=np.array(dz_e)
            self.resultats["Ezz_e"]=np.array(Ezz_e); self.resultats["Eyz_e"]=np.array(Eyz_e); self.resultats["Exz_e"]=np.array(Exz_e)                
            self.resultats["Szz_e"]=np.array(Szz_e); self.resultats["Syz_e"]=np.array(Syz_e); self.resultats["Sxz_e"]=np.array(Sxz_e)
            

        tic.Tac("Calcul deformations et contraintes aux elements", self.__verbosity)
        
        # if calculAuxNoeuds:
        #     self.__ExtrapolationAuxNoeuds(self.resultats["deplacementCoordo"])

        return list_Epsilon_e, list_Sigma_e 
    
    def __ExtrapolationAuxNoeuds(self, deplacementCoordo: np.ndarray, option = 'mean'):
        
        tic = TicTac()

        # Extrapolation des valeurs aux noeuds  

        dx = deplacementCoordo[:,0]; dy = deplacementCoordo[:,1]; dz = deplacementCoordo[:,2]

        Exx_n = []; Eyy_n = []; Ezz_n = []; Exy_n = []; Eyz_n = []; Exz_n = []
        Sxx_n = []; Syy_n = []; Szz_n = []; Sxy_n = []; Syz_n = []; Sxz_n = []
        Svm_n = []
        
        for noeud in self.__mesh.noeuds:
            # noeud = cast(Noeud, noeud)
            
            list_Exx = []; list_Eyy = []; list_Exy = []
            list_Sxx = []; list_Syy = []; list_Sxy = []
            list_Svm = []
                
            if self.__dim == 3:
                list_Ezz = []; list_Eyz = []; list_Exz = []                                
                list_Szz = []; list_Syz = []; list_Sxz = []
                        
            for element in noeud.elements:
                element = cast(Element, element)
                            
                listIdNoeuds = list(self.__mesh.connect[element.id])
                index = listIdNoeuds.index(noeud.id)
                BeDuNoeud = element.listB_u_n[index]
                
                # Construit ue
                ue = []
                for noeudDeLelement in element.noeuds:
                    # noeudDeLelement = cast(Noeud, noeudDeLelement)
                    
                    if self.__dim == 2:
                        ue.append(dx[noeudDeLelement.id])
                        ue.append(dy[noeudDeLelement.id])
                    if self.__dim == 3:
                        ue.append(dx[noeudDeLelement.id])
                        ue.append(dy[noeudDeLelement.id])
                        ue.append(dz[noeudDeLelement.id])
                        
                ue = np.array(ue)
                
                vect_Epsilon = BeDuNoeud.dot(ue)
                vect_Sigma = self.__materiau.C.dot(vect_Epsilon)
                
                if self.__dim == 2:                
                    list_Exx.append(vect_Epsilon[0]); list_Eyy.append(vect_Epsilon[1]); list_Exy.append(vect_Epsilon[2])
                    Sxx = vect_Sigma[0]; Syy = vect_Sigma[1]; Sxy = vect_Sigma[2]
                    list_Sxx.append(Sxx); list_Syy.append(Syy); list_Sxy.append(Sxy)
                    list_Svm.append(np.sqrt(Sxx**2+Syy**2-Sxx*Syy+3*Sxy**2))
                    
                elif self.__dim == 3:
                    list_Exx.append(vect_Epsilon[0]); list_Eyy.append(vect_Epsilon[1]); list_Ezz.append(vect_Epsilon[2])                    
                    list_Exy.append(vect_Epsilon[3]); list_Eyz.append(vect_Epsilon[4]); list_Exz.append(vect_Epsilon[5])                    
                    Sxx = vect_Sigma[0]; Syy = vect_Sigma[1]; Szz = vect_Sigma[2]
                    Sxy = vect_Sigma[3]; Syz = vect_Sigma[4]; Sxz = vect_Sigma[5]                    
                    list_Sxx.append(Sxx); list_Syy.append(Syy); list_Szz.append(Szz)
                    list_Sxy.append(Sxy); list_Syz.append(Syz); list_Sxz.append(Sxz)                    
                    Svm = np.sqrt(((Sxx-Syy)**2+(Syy-Szz)**2+(Szz-Sxx)**2+6*(Sxy**2+Syz**2+Sxz**2))/2)
                    list_Svm.append(Svm)
            
            def TrieValeurs(source:list, option: str):
                # Verifie si il ny a pas une valeur bizzare
                max = np.max(source)
                min = np.min(source)
                mean = np.mean(source)
                    
                valeurAuNoeud = 0
                if option == 'max':
                    valeurAuNoeud = max
                elif option == 'min':
                    valeurAuNoeud = min
                elif option == 'mean':
                    valeurAuNoeud = mean
                elif option == 'first':
                    valeurAuNoeud = source[0]
                    
                return valeurAuNoeud
            
            Exx_n.append(TrieValeurs(list_Exx, option))
            Eyy_n.append(TrieValeurs(list_Eyy, option)) 
            Exy_n.append(TrieValeurs(list_Exy, option))
            
            Sxx_n.append(TrieValeurs(list_Sxx, option))
            Syy_n.append(TrieValeurs(list_Syy, option))
            Sxy_n.append(TrieValeurs(list_Sxy, option))
            
            Svm_n.append(TrieValeurs(list_Svm, option))
        
            if self.__dim == 3:
                Ezz_n.append(TrieValeurs(list_Ezz, option))
                Eyz_n.append(TrieValeurs(list_Eyz, option))
                Exz_n.append(TrieValeurs(list_Exz, option))
                
                Szz_n.append(TrieValeurs(list_Szz, option))
                Syz_n.append(TrieValeurs(list_Syz, option))
                Sxz_n.append(TrieValeurs(list_Sxz, option))
            
        
        self.resultats["Exx_n"] = Exx_n; self.resultats["Eyy_n"] = Eyy_n; self.resultats["Exy_n"] = Exy_n
        self.resultats["Sxx_n"] = Sxx_n; self.resultats["Syy_n"] = Syy_n; self.resultats["Sxy_n"] = Sxy_n      
        self.resultats["Svm_n"] = Svm_n
        
        if self.__dim == 3:            
            self.resultats["Ezz_n"] = Ezz_n; self.resultats["Eyz_n"] = Eyz_n; self.resultats["Exz_n"] = Exz_n
            self.resultats["Szz_n"] = Szz_n; self.resultats["Syz_n"] = Syz_n; self.resultats["Sxz_n"] = Sxz_n
    
        tic.Tac("Calcul contraintes et deformations aux noeuds", self.__verbosity)

    def ConstruitH(self, u: np.ndarray):
            # Pour chaque point de gauss de tout les elements du maillage on va calculer phi+

            list_new_H = []
            for e in self.__mesh.elements:
                e = cast(Element, e)

                # Construit ui
                ui = []
                for n in e.noeuds:
                    # n = cast(Noeud, n)
                    for j in range(self.__dim):
                        valeur = u[n.id*self.__dim+j]
                        ui.append(valeur)            
                ui = np.array(ui)

                h_pg = []
                
                for pg in range(len(e.listB_u_pg)):
                    
                    B_pg = np.array(e.listB_u_pg[pg])                    
                    # N_pg = e.listN_d_pg[pg]
                    # jacobien = e.listJacobien_pg[pg]
                    # poid = e.listPoid_pg[pg]
                    
                    epsilon = B_pg.dot(ui)

                    h = 1/2 * epsilon.T.dot(self.__materiau.C).dot(epsilon)

                    tr = epsilon[0] + epsilon[1]

                    lamb = self.__materiau.lamb
                    mu = self.__materiau.mu

                    h2 = 1/2*lamb*tr**2+mu*epsilon.T.dot(epsilon)

                    # assert np.isclose(h,h2), "Erreur"

                    h=float(h2)

                    if(len(self.list_H)==0):
                        h_pg.append(h)
                    else:                        
                        h_pg.append(max(h, self.list_H[e.id][pg]))
                
                list_new_H.append(h_pg)
            
            new = np.linalg.norm(list_new_H)
            old = np.linalg.norm(self.list_H)
            assert new >= old, "Erreur"
            self.list_H = list_new_H


    def Solve_d(self, resolution=1):
            
        def Construit_Dglob():                
            taille = self.__mesh.Nn
            dGlob = np.zeros(taille)
            for i in range(taille):
                if i in self.__d_Connues:
                    dGlob[i] = self.__dc[i]
                elif i in self.__d_Inconnues:
                    ligne = self.__d_Inconnues.index(i)
                    dGlob[i] = di[ligne]

            return dGlob

        tic = TicTac()

        # Résolution du plus rapide au plus lent
        if resolution == 1:
            dGlob = sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(self.__Kd_penal), self.__Fd_penal)
        elif resolution == 2:
            d_Connues = self.__d_Connues
            d_Inconnues = self.__d_Inconnues

            assert len(d_Connues) + len(d_Inconnues) == self.__mesh.Nn, "Problème dans les conditions"

            Kii = self.__Kd[d_Inconnues, :][:, d_Inconnues]
            Kic = self.__Kd[d_Inconnues, :][:, d_Connues]
            dc = self.__dc[d_Connues]
            Fi = self.__Fd[d_Inconnues]

            di = sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(Kii), Fi-Kic.dot(dc))
            dGlob = Construit_Dglob()
        elif resolution == 3:            
            dGlob = sp.linalg.solve(self.__Kd_penal, self.__Fd_penal)
        elif resolution == 4:            
            dGlob = np.linalg.solve(self.__Kd_penal, self.__Fd_penal)

        tic.Tac("Résolution d", self.__verbosity)        
        
        # assert dGlob.max() <= 1, "Doit etre inférieur a 1"
        # assert dGlob.min() >= 0, "Doit etre supérieur 0"

        if(dGlob.max() > 1):
            print("dmax = {}".format(dGlob.max()))

        if(dGlob.min() < 0):
            print("dmin = {}".format(dGlob.min()))

        return dGlob   
         
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

        materiau = Materiau(dim)

        self.simulations2DElastique = []

        # Pour chaque type d'element 2D
        for i in range(len(Element.get_Types(dim))):
            # Construction du modele et du maillage 
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=i, tailleElement=taille, verbosity=False)

            (coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim, mesh, materiau, verbosity=False)

            simu.Assemblage_u(epaisseur=b)

            noeuds_en_L = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == L]
            noeuds_en_0 = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == 0]  

            simu.Condition_Neumann(noeuds=noeuds_en_L, valeur=P, directions=["y"])

            simu.Condition_Dirichlet(noeuds=noeuds_en_0, valeur=0, directions=["x", "y"])

            self.simulations2DElastique.append(simu)

    def CreationDesSimusElastique3D(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        fichier = dir_path + '\\models\\part.stp'

        dim = 3

        # Paramètres géométrie
        L = 120  #mm
        h = 13    
        b = 13

        P = -800 #N

        # Paramètres maillage        
        taille = L

        materiau = Materiau(dim)
        
        self.simulations3DElastique = []

        for i in range(len(Element.get_Types(dim))):
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=i, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False, verbosity=False)

            (coordo, connect) = modelGmsh.Importation3D(fichier)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim,mesh, materiau, verbosity=False)

            simu.Assemblage_u(epaisseur=b)

            noeuds_en_L = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == L]
            noeuds_en_0 = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == 0]

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


if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")    

        
            
