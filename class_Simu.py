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
    
    def AssemblageKglobFglob(self, epaisseur=0):
        """Construit Kglobal

        mettre en option u ou d ?

        """

        TicTac.Tic()
        
        if self.__dim == 2:        
            assert epaisseur>0,"Doit être supérieur à 0"

        taille = self.__mesh.Nn*self.__dim

        self.__Kglob = np.zeros((taille, taille))
        self.__Fglob = np.zeros(taille)
        
        for e in self.__mesh.elements:            
            e = cast(Element, e)
            nPe = e.nPe
            assembly = e.assembly
            Ke = e.Construit_Ke_deplacement(self.__materiau.C)
            
            # Assemble Ke dans Kglob 
            
            # # Méthode 1
            # for i in range(nPe*self.__dim):
            #     ligne = assembly[i]
            #     for j in range(nPe*self.__dim):
            #         colonne = assembly[j]
            #         if self.__dim == 2:
            #             self.__Kglob[ligne, colonne] += epaisseur * Ke[i, j]
            #         elif self.__dim ==3:
            #             self.__Kglob[ligne, colonne] += Ke[i, j]

            # Méthode 2
            vect1 = []
            vect2 = []
            for i in assembly:
                vect1.extend(assembly)
                for j in range(len(assembly)):
                    vect2.append(i)

            if self.__dim == 2:
                self.__Kglob[vect1, vect2] = self.__Kglob[vect1, vect2] + np.ravel(epaisseur * Ke)
            elif self.__dim == 3:
                self.__Kglob[vect1, vect2] = self.__Kglob[vect1, vect2] + np.ravel(Ke)



            test = self.__Kglob[assembly, :][:, assembly]
            pass

        TicTac.Tac("Assemblage", self.__verbosity)


    def ConstruitH(self, d, u):
        # Pour chaque point de gauss de tout les elements du maillage on va calculer phi+

        pass



    def ConditionEnForce(self, noeuds=[], force=None, directions=[]):
        
        assert isinstance(noeuds[0], Noeud), "Doit être une liste de Noeuds"
        assert not force == 0, "Doit être différent de 0"
        assert isinstance(directions[0], str), "Doit être une liste de chaine de caractère"

        TicTac.Tic()
        
        nbn = len(noeuds)

        for direction in directions:
            for n in noeuds:
                n = cast(Noeud, n)

                if direction == "x":
                    ligne = n.id * self.__dim
                if direction == "y":
                    ligne = n.id * self.__dim + 1
                if direction == "z":
                    assert self.__dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant Z"
                    ligne = n.id * self.__dim + 2
                    
                self.__Fglob[ligne] += force/nbn
        
        TicTac.Tac("Condition en force", self.__verbosity)

    def ConditionEnDeplacement(self, noeuds=[], direction="", deplacement=0):
        
        TicTac.Tic()
               
        for n in noeuds:
            n = cast(Noeud, n)
            
            if direction == "x":
                ligne = n.id * self.__dim
                
            if direction == "y":
                ligne = n.id * self.__dim + 1
                
            if direction == "z":
                ligne = n.id * self.__dim + 2
            
            self.__Fglob[ligne] = deplacement
            self.__Kglob[ligne,:] = 0.0
            self.__Kglob[ligne, ligne] = 1

        TicTac.Tac("Condition en déplacement", self.__verbosity)

    def Solve(self):

        TicTac.Tic()
        
        # Méthodes moints rapide
        # self.__Uglob = np.linalg.solve(self.__Kglob, self.__Fglob)
        # self.__Uglob = sp.linalg.solve(self.__Kglob, self.__Fglob)

        # Résolution du plus rapide au plus lent  
        Uglob = spsolve(sp.sparse.csr_matrix(self.__Kglob), self.__Fglob)
        
        # Reconstruit Uglob
        Uglob = np.array(Uglob)

        # Energie de deformation
        self.resultats["Wdef"] = 1/2 * Uglob.T.dot(self.__Kglob).dot(Uglob)

        # Récupère les déplacements       
        dim = self.__dim
        Nn = self.__mesh.Nn

        dx = np.array([Uglob[i*dim] for i in range(Nn)])
        dy = np.array([Uglob[i*dim+1] for i in range(Nn)])
        if dim == 2:
            dz = np.zeros(Nn)
        else:
            dz = np.array([Uglob[i*dim+2] for i in range(Nn)])
        
        self.resultats["dx_n"] = dx
        self.resultats["dy_n"] = dy        
        if self.__dim == 3:
            self.resultats["dz_n"] = dz

        self.resultats["deplacementCoordo"] = np.array([dx, dy, dz]).T

        TicTac.Tac("Résolution", self.__verbosity)
        
        self.CalculDeformationEtContrainte(Uglob)



    def CalculDeformationEtContrainte(self, Uglob: np.ndarray, sauvegarde=True):
        
        TicTac.Tic()

        list_Epsilon_e = []
        list_Sigma_e = []
        dim = self.__dim
        
        # Prépare les vecteurs de stockage par element
        dx_e = []
        dy_e = []

        Exx_e = []
        Eyy_e = []
        Exy_e = []

        Sxx_e = []
        Syy_e = []
        Sxy_e = []

        Svm_e = []

        if dim == 3:
            dz_e = []

            Ezz_e = []
            Eyz_e = []
            Exz_e = []

            Szz_e = []
            Syz_e = []
            Sxz_e = []

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
            for B_pg in e.listB_pg:
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

        if sauvegarde:
            self.resultats["dx_e"]=np.array(dx_e)
            self.resultats["dy_e"]=np.array(dy_e)

            self.resultats["Exx_e"]=np.array(Exx_e)
            self.resultats["Eyy_e"]=np.array(Eyy_e)
            self.resultats["Exy_e"]=np.array(Exy_e)
            
            self.resultats["Sxx_e"]=np.array(Sxx_e)
            self.resultats["Syy_e"]=np.array(Syy_e)
            self.resultats["Sxy_e"]=np.array(Sxy_e)

            self.resultats["Svm_e"]=np.array(Svm_e)

            if dim == 3:
                self.resultats["dz_e"]=np.array(dz_e)

                self.resultats["Ezz_e"]=np.array(Ezz_e)
                self.resultats["Eyz_e"]=np.array(Eyz_e)
                self.resultats["Exz_e"]=np.array(Exz_e)
                
                self.resultats["Szz_e"]=np.array(Szz_e)
                self.resultats["Syz_e"]=np.array(Syz_e)
                self.resultats["Sxz_e"]=np.array(Sxz_e)

        TicTac.Tac("Calcul deformations et contraintes aux elements", self.__verbosity)
        
        self.__ExtrapolationAuxNoeuds(self.resultats["deplacementCoordo"])
    
    def __ExtrapolationAuxNoeuds(self, deplacementCoordo: np.ndarray, option = 'mean'):
        
        TicTac.Tic()

        # Extrapolation des valeurs aux noeuds  

        dx = deplacementCoordo[:,0]
        dy = deplacementCoordo[:,1]
        dz = deplacementCoordo[:,2]

        Exx_n = []
        Eyy_n = []
        Ezz_n = []
        Exy_n = []
        Eyz_n = []
        Exz_n = []
        
        Sxx_n = []
        Syy_n = []
        Szz_n = []
        Sxy_n = []
        Syz_n = []
        Sxz_n = []
        
        Svm_n = []
        
        for noeud in self.__mesh.noeuds:
            noeud = cast(Noeud, noeud)
            
            list_Exx = []
            list_Eyy = []
            list_Exy = []
            
            list_Sxx = []
            list_Syy = []
            list_Sxy = []
            
            list_Svm = []      
                
            if self.__dim == 3:                
                list_Ezz = []
                list_Eyz = []
                list_Exz = []
                                
                list_Szz = []                
                list_Syz = []
                list_Sxz = []
                        
            for element in noeud.elements:
                element = cast(Element, element)
                            
                listIdNoeuds = list(self.__mesh.connect[element.id])
                index = listIdNoeuds.index(noeud.id)
                BeDuNoeud = element.listB_n[index]
                
                # Construit ue
                deplacement = []
                for noeudDeLelement in element.noeuds:
                    noeudDeLelement = cast(Noeud, noeudDeLelement)
                    
                    if self.__dim == 2:
                        deplacement.append(dx[noeudDeLelement.id])
                        deplacement.append(dy[noeudDeLelement.id])
                    if self.__dim == 3:
                        deplacement.append(dx[noeudDeLelement.id])
                        deplacement.append(dy[noeudDeLelement.id])
                        deplacement.append(dz[noeudDeLelement.id])
                        
                deplacement = np.array(deplacement)
                
                vect_Epsilon = BeDuNoeud.dot(deplacement)
                vect_Sigma = self.__materiau.C.dot(vect_Epsilon)
                
                if self.__dim == 2:                
                    list_Exx.append(vect_Epsilon[0])
                    list_Eyy.append(vect_Epsilon[1])
                    list_Exy.append(vect_Epsilon[2])
                    
                    Sxx = vect_Sigma[0]
                    Syy = vect_Sigma[1]
                    Sxy = vect_Sigma[2]                    
                    
                    list_Sxx.append(Sxx)
                    list_Syy.append(Syy)
                    list_Sxy.append(Sxy)
                    list_Svm.append(np.sqrt(Sxx**2+Syy**2-Sxx*Syy+3*Sxy**2))
                    
                elif self.__dim == 3:
                    list_Exx.append(vect_Epsilon[0]) 
                    list_Eyy.append(vect_Epsilon[1])
                    list_Ezz.append(vect_Epsilon[2])                    
                    list_Exy.append(vect_Epsilon[3])
                    list_Eyz.append(vect_Epsilon[4])
                    list_Exz.append(vect_Epsilon[5])
                    
                    Sxx = vect_Sigma[0]
                    Syy = vect_Sigma[1]
                    Szz = vect_Sigma[2]                    
                    Sxy = vect_Sigma[3]
                    Syz = vect_Sigma[4]
                    Sxz = vect_Sigma[5]
                    
                    list_Sxx.append(Sxx)
                    list_Syy.append(Syy)
                    list_Szz.append(Szz)
                    
                    list_Sxy.append(Sxy)
                    list_Syz.append(Syz)
                    list_Sxz.append(Sxz)
                    
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
            
        
        self.resultats["Exx_n"] = Exx_n
        self.resultats["Eyy_n"] = Eyy_n
        self.resultats["Exy_n"] = Exy_n
        self.resultats["Sxx_n"] = Sxx_n
        self.resultats["Syy_n"] = Syy_n
        self.resultats["Sxy_n"] = Sxy_n      
        self.resultats["Svm_n"] = Svm_n
        
        if self.__dim == 3:            
            self.resultats["Ezz_n"] = Ezz_n
            self.resultats["Eyz_n"] = Eyz_n
            self.resultats["Exz_n"] = Exz_n
            
            self.resultats["Szz_n"] = Szz_n
            self.resultats["Syz_n"] = Syz_n
            self.resultats["Sxz_n"] = Sxz_n
    
        TicTac.Tac("Calcul contraintes et deformations aux noeuds", self.__verbosity)
                
    
         
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

            simu.AssemblageKglobFglob(epaisseur=b)

            noeud_en_L = []
            noeud_en_0 = []
            for n in mesh.noeuds:            
                    n = cast(Noeud, n)
                    if n.coordo[0] == L:
                            noeud_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeud_en_0.append(n)

            simu.ConditionEnForce(noeuds=noeud_en_L, force=P, directions=["y"])

            simu.ConditionEnDeplacement(noeuds=noeud_en_0, deplacement=0, direction="x")
            simu.ConditionEnDeplacement(noeuds=noeud_en_0, deplacement=0, direction="y")

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

            simu.AssemblageKglobFglob(epaisseur=b)

            noeuds_en_L = []
            noeuds_en_0 = []
            for n in mesh.noeuds:
                    n = cast(Noeud, n)        
                    if n.coordo[0] == L:
                            noeuds_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeuds_en_0.append(n)

            simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, directions=["z"])

            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="x")
            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="y")
            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="z")

            self.simulations3DElastique.append(simu)
    
    def setUp(self):
        self.CreationDesSimusElastique2D()
        self.CreationDesSimusElastique3D()  

    def test_ResolutionDesSimulationsElastique2D(self):
        # Pour chaque type de maillage on simule
        for simu in self.simulations2DElastique:
            simu = cast(Simu, simu)
            simu.Solve()

    def test_ResolutionDesSimulationsElastique3D(self):
        # Pour chaque type de maillage on simule
        for simu in self.simulations3DElastique:
            simu = cast(Simu, simu)
            simu.Solve()


if __name__ == '__main__':        
    try:
        os.system("cls")    #nettoie terminal
        unittest.main(verbosity=2)    
    except:
        print("")    

        
            
