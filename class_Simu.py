import os
from typing import cast

import numpy as np
from numpy.core.records import array
import scipy as sp
from scipy.sparse.linalg import spsolve

from class_ModelGmsh import ModelGmsh
from class_Noeud import Noeud
from class_Element import Element
from class_Mesh import Mesh
from class_Materiau import Materiau
from class_TicTac import TicTac

class Simu:
    
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
    
    def Assemblage_u(self, epaisseur=0, d=[]):
        """Construit Kglobal

        mettre en option u ou d ?

        """
        
        if self.__dim == 2:        
            assert epaisseur>0,"Doit être supérieur à 0"

        TicTac.Tic()
        
        taille = self.__mesh.Nn*self.__dim

        self.__Ku = np.zeros((taille, taille))
        self.__Fu = np.zeros(taille)
        
        self.__ddl_Inconnues = [i for i in range(taille)]
        self.__ddl_Connues = []
        self.__Uc = np.zeros(taille)


        # if len(d)==0:
        #     listeKe = [[e.listJacobien_pg[pg] * e.listPoid_pg[pg] * e.listB_u_pg[pg].T.dot(self.__materiau.C).dot(e.listB_u_pg[pg]) for pg in range(len(e.listB_u_pg))] for e in self.__mesh.elements]    
        # else:
        #     listeKe = [[e.listJacobien_pg[pg] * e.listPoid_pg[pg] - (1-e.listN_d_pg[pg].dot(np.array([d[n.id] for n in e.noeuds])))**2* e.listB_u_pg[pg].T.dot(self.__materiau.C).dot(e.listB_u_pg[pg]) for pg in range(len(e.listB_u_pg))] for e in self.__mesh.elements]    

        

        for e in self.__mesh.elements:            
            e = cast(Element, e)

            # Pour chaque poing de gauss on construit Ke
            Ke = 0
            for pg in range(len(e.listB_u_pg)):
                jacobien = e.listJacobien_pg[pg]
                poid = e.listPoid_pg[pg]
                B_pg = e.listB_u_pg[pg]

                if len(d)==0:
                    # probleme standart
                    Ke += jacobien * poid * B_pg.T.dot(self.__materiau.C).dot(B_pg)

                else:
                    # probleme endomagement
                    de = np.array([d[n.id] for n in e.noeuds])
                    # Bourdin
                    Ke += jacobien * poid * (1-e.listN_d_pg[pg].dot(de))**2 *B_pg.T.dot(self.__materiau.C).dot(B_pg)

            # # print(Ke-listeKe[e.id])
            
            # Ke = np.array(listeKe[e.id])

            # Assemble Ke dans Kglob
            lignes = []
            colonnes = []
            for i in e.assembly:
                lignes.extend(e.assembly)
                for j in range(len(e.assembly)):
                    colonnes.append(i)

            if self.__dim == 2:                
                self.__Ku[lignes, colonnes] += np.ravel(epaisseur * Ke)
            elif self.__dim == 3:
                self.__Ku[lignes, colonnes] += np.ravel(Ke)

        TicTac.Tac("Assemblage u", self.__verbosity)
        
        self.__Ku = sp.sparse.lil_matrix(self.__Ku).T
        self.__Fu = sp.sparse.lil_matrix(self.__Fu).T

        # self.__Ku_penal = np.copy(self.__Ku)
        # self.__Fu_penal = np.copy(self.__Fu)

        self.__Ku_penal = self.__Ku.copy()
        self.__Fu_penal = self.__Fu.copy()

        return self.__Ku, self.__Fu

    def Assemblage_d(self, Gc=1, l=0.001):
        """Construit Kglobal

        mettre en option u ou d ?

        """

        TicTac.Tic()
        
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
            
        TicTac.Tac("Assemblage d", self.__verbosity)

        self.__Kd_penal = np.copy(self.__Kd)
        self.__Fd_penal = np.copy(self.__Fd)

        return self.__Kd, self.__Fd

    

    def Condition_Neumann(self, noeuds: list, directions: list, valeur=0.0, option="u"):
        """Applique les conditions en force


        Parameters
        ----------
        noeuds : list, optional
            Liste de noeuds, by default []
        force : float, optional
            Force que l'on veut appliquer aux noeuds, by default 0.0
        directions : list, optional
            ["x", "y", "z"] vecteurs sur lesquelles on veut appliquer la force , by default [] 
        """

        TicTac.Tic()

        nbn = len(noeuds)

        assert isinstance(noeuds[0], Noeud), "Doit être une liste de Noeuds"
        assert option in ["u", "d"], "Mauvaise option"        
        if option == "d":
            assert len(directions) == 0, "lorsque on renseigne d on a pas besoin de direction"
            assert not valeur == 0.0, "Doit être différent de 0"

            for n in noeuds:
                n = cast(Noeud, n)
                self.__Fd[n.id] += valeur/nbn
                self.__Fd_penal[n.id] += valeur/nbn

        elif option == "u":
            assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"
            for direction in directions:
                assert direction in ["x", "y", "z"] , "direction doit etre x y ou z"

            for direction in directions:
                for n in noeuds:
                    n = cast(Noeud, n)
                    # Récupère la ligne sur laquelle on veut appliquer la force
                    if direction == "x":
                        ligne = n.id * self.__dim
                    if direction == "y":
                        ligne = n.id * self.__dim + 1
                    if direction == "z":
                        assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant z"
                        ligne = n.id * self.__dim + 2
                        
                    # self.__Fu[ligne] += valeur/nbn
                    # self.__Fu_penal[ligne] += valeur/nbn

                    self.__Fu[ligne,0] += valeur/nbn
                    self.__Fu_penal[ligne,0] += valeur/nbn
        
        TicTac.Tac("Condition Neumann", self.__verbosity)

    def Condition_Dirichlet(self, noeuds: list, directions=[] , valeur=0.0, option="u"):
        
        assert isinstance(noeuds[0], Noeud), "Doit être une liste de Noeuds"        
        assert option in ["u", "d"], "Mauvaise option"
        if option == "d":
            assert len(directions) == 0, "lorsque on renseigne d on a pas besoin de direction"
            assert valeur >= 0 or valeur <= 1, "d doit etre compris entre [0;1]"
        elif option == "u":
            assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"
            for direction in directions:
                assert direction in ["x", "y", "z"] , "direction doit etre x y ou z"

        TicTac.Tic()

        if option == "d":
            for n in noeuds:
                n = cast(Noeud, n)
                ligne = n.id

                if ligne in self.__d_Inconnues:
                    self.__d_Inconnues.remove(ligne)
                if ligne not in self.__d_Connues:
                    self.__dc[ligne] = valeur
                    self.__d_Connues.append(ligne)

                self.__Fd_penal[ligne] = valeur
                self.__Kd_penal[ligne,:] = 0.0
                self.__Kd_penal[ligne, ligne] = 1

        elif option == "u":
            for n in noeuds:
                n = cast(Noeud, n)
                for direction in directions:
                    if direction == "x":
                        ligne = n.id * self.__dim
                    if direction == "y":
                        ligne = n.id * self.__dim + 1
                    if direction == "z":
                        ligne = n.id * self.__dim + 2
                    
                    # Decomposition
                    if ligne in self.__ddl_Inconnues:
                        self.__ddl_Inconnues.remove(ligne)
                    if ligne not in self.__ddl_Connues:
                        self.__Uc[ligne] = valeur
                        self.__ddl_Connues.append(ligne)

                    # Pénalisation
                    self.__Fu_penal[ligne] = valeur
                    self.__Ku_penal[ligne,:] = 0.0
                    self.__Ku_penal[ligne, ligne] = 1
        
                

        TicTac.Tac("Condition Dirichlet", self.__verbosity)

    def Solve_u(self, resolution=1, save=True):
        
        def ConstruitUglob():
            # Reconstruit Uglob
            taille = self.__mesh.Nn*self.__dim
            Uglob = np.zeros(taille)
            for i in range(taille):
                if i in self.__ddl_Connues:
                    Uglob[i] = self.__Uc[i]
                elif i in self.__ddl_Inconnues:
                    ligne = self.__ddl_Inconnues.index(i)
                    Uglob[i] = ui[ligne]
            
            return Uglob

        TicTac.Tic()

        # Résolution du plus rapide au plus lent
        if resolution == 1:
            Uglob = sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(self.__Ku_penal), self.__Fu_penal)
        elif resolution == 2:
            ddl_Connues = self.__ddl_Connues
            ddl_Inconnues = self.__ddl_Inconnues

            assert len(ddl_Connues) + len(ddl_Inconnues) == self.__mesh.Nn*self.__dim, "Problème dans les conditions"

            Kii = self.__Ku.toarray()[ddl_Inconnues, :][:, ddl_Inconnues]
            Kic = self.__Ku.toarray()[ddl_Inconnues, :][:, ddl_Connues]
            Fi = self.__Fu.toarray()[ddl_Inconnues]
            
            # Kii = self.__Ku[ddl_Inconnues, :][:, ddl_Inconnues]
            # Kic = self.__Ku[ddl_Inconnues, :][:, ddl_Connues]
            # Fi = self.__Fu[ddl_Inconnues]

            uc = self.__Uc[ddl_Connues]  
            
            ui = sp.sparse.linalg.spsolve(sp.sparse.csr_matrix(Kii), Fi-Kic.dot(uc))
            
            Uglob = ConstruitUglob() 
        elif resolution == 3:
            Uglob = np.linalg.solve(self.__Ku_penal, self.__Fu_penal)
        elif resolution == 4:
            Uglob = sp.linalg.solve(self.__Ku_penal, self.__Fu_penal)

        TicTac.Tac("Résolution {}".format(resolution) , self.__verbosity)        
        
        if save:
            self.__Save_u(Uglob)

        return Uglob

    def __Save_u(self, Uglob: np.ndarray):
        # Energie de deformation
        Kglob = np.array(self.__Ku.todense())
        self.resultats["Wdef"] = 1/2 * Uglob.T.dot(Kglob).dot(Uglob)

        # Récupère les déplacements
        dx = np.array([Uglob[i*self.__dim] for i in range(self.__mesh.Nn)])
        dy = np.array([Uglob[i*self.__dim+1] for i in range(self.__mesh.Nn)])
        if self.__dim == 2:
            dz = np.zeros(self.__mesh.Nn)
        else:
            dz = np.array([Uglob[i*self.__dim+2] for i in range(self.__mesh.Nn)])
        
        self.resultats["dx_n"] = dx
        self.resultats["dy_n"] = dy        
        if self.__dim == 3:
            self.resultats["dz_n"] = dz

        self.resultats["deplacementCoordo"] = np.array([dx, dy, dz]).T
        
        self.__CalculDeformationEtContrainte(Uglob)
            

    def __CalculDeformationEtContrainte(self, Uglob: np.ndarray, calculAuxNoeuds=True):
        
        TicTac.Tic()

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
        for e in self.__mesh.elements:
            e = cast(Element, e)

            dx = []
            dy = []
            if dim == 3:
                dz = []

            # Construit ue
            ue = []
            for n in e.noeuds:
                n = cast(Noeud, n)
                for j in range(self.__dim):
                    valeur = Uglob[n.id*self.__dim+j]
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
            for B_pg in e.listB_u_pg:
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
            

        TicTac.Tac("Calcul deformations et contraintes aux elements", self.__verbosity)
        
        if calculAuxNoeuds:
            self.__ExtrapolationAuxNoeuds(self.resultats["deplacementCoordo"])

        return list_Epsilon_e, list_Sigma_e 
    
    def __ExtrapolationAuxNoeuds(self, deplacementCoordo: np.ndarray, option = 'mean'):
        
        TicTac.Tic()

        # Extrapolation des valeurs aux noeuds  

        dx = deplacementCoordo[:,0]; dy = deplacementCoordo[:,1]; dz = deplacementCoordo[:,2]

        Exx_n = []; Eyy_n = []; Ezz_n = []; Exy_n = []; Eyz_n = []; Exz_n = []
        Sxx_n = []; Syy_n = []; Szz_n = []; Sxy_n = []; Syz_n = []; Sxz_n = []
        Svm_n = []
        
        for noeud in self.__mesh.noeuds:
            noeud = cast(Noeud, noeud)
            
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
                    noeudDeLelement = cast(Noeud, noeudDeLelement)
                    
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
    
        TicTac.Tac("Calcul contraintes et deformations aux noeuds", self.__verbosity)

    def ConstruitH(self, u: np.ndarray):
            # Pour chaque point de gauss de tout les elements du maillage on va calculer phi+

            list_new_H = []
            for e in self.__mesh.elements:
                e = cast(Element, e)

                # Construit ui
                ui = []
                for n in e.noeuds:
                    n = cast(Noeud, n)
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

        TicTac.Tic()

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

        TicTac.Tac("Résolution d", self.__verbosity)        
        
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
        for type in ModelGmsh.get_typesMaillage2D():
            # Construction du modele et du maillage 
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille, verbosity=False)

            (coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim, mesh, materiau, verbosity=False)

            simu.Assemblage_u(epaisseur=b)

            noeud_en_L = []
            noeud_en_0 = []
            for n in mesh.noeuds:            
                    n = cast(Noeud, n)
                    if n.coordo[0] == L:
                            noeud_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeud_en_0.append(n)

            simu.Condition_Neumann(noeuds=noeud_en_L, valeur=P, directions=["y"])

            simu.Condition_Dirichlet(noeuds=noeud_en_0, valeur=0, directions=["x", "y"])

            self.simulations2DElastique.append(simu)

    def CreationDesSimusElastique3D(self):

        fichier = "part.stp"

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

        for type in ModelGmsh.get_typesMaillage3D():
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False, verbosity=False)

            (coordo, connect) = modelGmsh.Importation3D(fichier)
            mesh = Mesh(dim, coordo, connect, verbosity=False)

            simu = Simu(dim,mesh, materiau, verbosity=False)

            simu.Assemblage_u(epaisseur=b)

            noeuds_en_L = []
            noeuds_en_0 = []
            for n in mesh.noeuds:
                    n = cast(Noeud, n)        
                    if n.coordo[0] == L:
                            noeuds_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeuds_en_0.append(n)

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

        
            
