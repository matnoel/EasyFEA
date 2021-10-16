from typing import cast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import *
import scipy.sparse as sp
from scipy.sparse.linalg import inv
import numpy as np
import gmsh
import os
import time
from class_ModelGmsh import ModelGmsh

from class_Noeud import Noeud
from class_Element import Element
from class_Mesh import Mesh
from class_Materiau import Materiau

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


        self.dim = dim
      
        self.verbosity = verbosity
        
        self.mesh = mesh
        
        self.materiau = materiau
        
        self.resultats = {}
    
    def Assemblage(self, epaisseur=0):
        """Construit Kglobal

        mettre en option u ou d ?

        """

        START = time.time()
        
        if self.dim == 2:        
            assert epaisseur>0,"Doit être supérieur à 0"

        taille = self.mesh.get_Nn()*self.dim

        self.__Kglob = np.zeros((taille, taille))
        self.__Fglob = np.zeros(taille)
        
        for e in self.mesh.elements:            
            e = cast(Element, e)
            
            Ke = e.ConstruitKe(self.materiau.C)
            
            test = Ke[:,0]

            # Assemble Ke dans Kglob 
            nPe = e.nPe
            vect = e.assembly
            i = 0
            while i<nPe*self.dim:
                ligne = vect[i] 
                j=0
                while j<nPe*self.dim:
                    colonne = vect[j]
                    
                    if self.dim == 2:
                        self.__Kglob[ligne, colonne] += epaisseur * Ke[i, j]
                    elif self.dim ==3:
                        self.__Kglob[ligne, colonne] += Ke[i, j]
                    j += 1                                  
                i += 1
           
            # # todo a essayer
            # Kglob = np.zeros((taille, taille))            
            # vect = e.assembly                                
            # if self.dim == 2:
            #     K1 = self.__Kglob
            #     K2 = self.__Kglob[vect,:][:,vect]                 
            #     # Kglob[vect,:][:,vect] += Kglob[vect,:][:,vect] + epaisseur * Ke[:,:]
            #     Kglob[vect,vect] += Ke
            #     pass
            # elif self.dim == 3:    
            #     Kglob[vect,:][:,vect] = Kglob[vect,:][:,vect] + Ke
                
        

            
        
        END = START - time.time()
        if self.verbosity:
            print("\nAssemblage ({:.3f} s)".format(np.abs(END)))


    def ConstruitH(self, d, u):
        # Pour chaque point de gauss de tout les elements du maillage on va calculer phi+

        pass



    def ConditionEnForce(self, noeuds=[], direction="", force=0):
        START = time.time()
        
        nbn = len(noeuds)
        for n in noeuds:
            n = cast(Noeud, n)
            
            if direction == "X":
                ligne = n.id * self.dim
                
            if direction == "Y":
                ligne = n.id * self.dim + 1
                
            if direction == "Z":
                assert self.dim == 3,"Une étude 2D ne permet pas d'appliquer des forces suivant Z"
                ligne = n.id * self.dim + 2
                
            self.__Fglob[ligne] += force/nbn
            
        END = START - time.time()
        if self.verbosity:
            print("\nCondition en force ({:.3f} s)".format(np.abs(END)))

    def ConditionEnDeplacement(self, noeuds=[], direction="", deplacement=0):
        START = time.time()
               
        for n in noeuds:
            n = cast(Noeud, n)
            
            if direction == "X":
                ligne = n.id * self.dim
                
            if direction == "Y":
                ligne = n.id * self.dim + 1
                
            if direction == "Z":
                ligne = n.id * self.dim + 2
            
            self.__Fglob[ligne] = deplacement
            self.__Kglob[ligne,:] = 0.0
            self.__Kglob[ligne, ligne] = 1
            
            
        END = START - time.time()
        if self.verbosity:
            print("\nCondition en deplacement ({:.3f} s)".format(np.abs(END)))   

    def Solve(self):
        START = time.time()
        
        # Transformatoion en matrice creuse
        self.__Kglob = sp.csc_matrix(self.__Kglob)
        self.__Fglob = sp.lil_matrix(self.__Fglob).T
        
        # Résolution 
        Uglob = inv(self.__Kglob).dot(self.__Fglob)
        
        # Récupération des données
        self.resultats["Wdef"] = 1/2 * Uglob.T.dot(self.__Kglob).dot(Uglob).data[0]

        # Récupère les déplacements

        Uglob = Uglob.toarray()
                
        dx = []
        dy = []
        dz = []
        for n in self.mesh.noeuds:
            n = cast(Noeud, n)
            
            idNoeud = n.id 
            if self.dim == 2:
                dx.append(Uglob[idNoeud * 2][0])
                dy.append(Uglob[idNoeud * 2 + 1][0])
            elif self.dim == 3:
                dx.append(Uglob[idNoeud * 3][0])
                dy.append(Uglob[idNoeud * 3 + 1][0])
                dz.append(Uglob[idNoeud * 3 + 2][0])
                
        dx  = np.array(dx)
        dy  = np.array(dy)
        dz  = np.array(dz)
        
        self.resultats["dx"] = dx
        self.resultats["dy"] = dy        
        if self.dim == 3:
            self.resultats["dz"] = dz
        
        self.__ExtrapolationAuxNoeuds(dx, dy, dz)        
        
        END = START - time.time()
        if self.verbosity:
            print("\nRésolution ({:.3f} s)".format(np.abs(END)))

    def __ExtrapolationAuxNoeuds(self, dx, dy, dz, option = 'mean'):
        # Extrapolation des valeurs aux noeuds  
        
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
        
        for noeud in self.mesh.noeuds:            
            noeud = cast(Noeud, noeud)
            
            list_Exx = []
            list_Eyy = []
            list_Exy = []
            
            list_Sxx = []
            list_Syy = []
            list_Sxy = []
            
            list_Svm = []      
                
            if self.dim == 3:                
                list_Ezz = []
                list_Eyz = []
                list_Exz = []
                                
                list_Szz = []                
                list_Syz = []
                list_Sxz = []
                        
            for element in noeud.elements:
                element = cast(Element, element)
                            
                listIdNoeuds = list(self.mesh.connection[element.id])
                index = listIdNoeuds.index(noeud.id)
                BeDuNoeud = element.listBeAuNoeuds[index]
                
                # Construit ue
                deplacement = []
                for noeudDeLelement in element.noeuds:
                    noeudDeLelement = cast(Noeud, noeudDeLelement)
                    
                    if self.dim == 2:
                        deplacement.append(dx[noeudDeLelement.id])
                        deplacement.append(dy[noeudDeLelement.id])
                    if self.dim == 3:
                        deplacement.append(dx[noeudDeLelement.id])
                        deplacement.append(dy[noeudDeLelement.id])
                        deplacement.append(dz[noeudDeLelement.id])
                        
                deplacement = np.array(deplacement)
                
                vect_Epsilon = BeDuNoeud.dot(deplacement)
                vect_Sigma = self.materiau.C.dot(vect_Epsilon)
                
                if self.dim == 2:                
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
                    
                elif self.dim == 3:
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
        
            if self.dim == 3:
                Ezz_n.append(TrieValeurs(list_Ezz, option))
                Eyz_n.append(TrieValeurs(list_Eyz, option))
                Exz_n.append(TrieValeurs(list_Exz, option))
                
                Szz_n.append(TrieValeurs(list_Szz, option))
                Syz_n.append(TrieValeurs(list_Syz, option))
                Sxz_n.append(TrieValeurs(list_Sxz, option))
            
        
        self.resultats["Exx"] = Exx_n
        self.resultats["Eyy"] = Eyy_n
        self.resultats["Exy"] = Exy_n
        self.resultats["Sxx"] = Sxx_n
        self.resultats["Syy"] = Syy_n
        self.resultats["Sxy"] = Sxy_n      
        self.resultats["Svm"] = Svm_n
        
        if self.dim == 3:            
            self.resultats["Ezz"] = Ezz_n
            self.resultats["Eyz"] = Eyz_n
            self.resultats["Exz"] = Exz_n
            
            self.resultats["Szz"] = Szz_n
            self.resultats["Syz"] = Syz_n
            self.resultats["Sxz"] = Sxz_n
    
    def PlotResult(self, resultat="", deformation=False, affichageMaillage=False):
        """Affiche le resultat de la simulation 

        Parameters
        ----------
        resultat : str, optional 
            dx, dy, dz \n
            Exx, Eyy, Ezz, Exy, Exz, Eyz \n
            Sxx, Syy, Szz, Sxy, Sxz, Syz \n
            Svm\n
            by default ""
        deformation : bool, optional
            Affichage de la deformation, by default False
        affichageMaillage : bool, optional
            Affichage du maillage, by default False        
        """
        
        # Construit la nouvelle matrice de connection 
        # ou de coordonnées si elle n'ont pas deja ete faite
        if len(self.mesh.connectionPourAffichage) == 0:
            self.__ConstruitConnectPourAffichage()        
        if len(self.mesh.new_coordo) == 0:
            self.__ConstruitNouvelleCoordo()
        
        # Va chercher les valeurs
        
        valeurs = self.resultats[resultat]
        
        if self.dim == 2:
            
            fig, ax = plt.subplots()
            
            # Trace le maillage
            if affichageMaillage:
                ax = self.__PlotMesh2D(ax, deformation, False)
            
            coordo = []
            
            if deformation:
                coordo = self.mesh.new_coordo
            else:
                coordo = self.mesh.coordo
                
            pc = ax.tricontourf(coordo[:,0], coordo[:,1], self.mesh.connectionPourAffichage, valeurs,
                                cmap='jet', antialiased=True)
            
            fig.colorbar(pc, ax=ax)
            ax.axis('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            
        
        elif self.dim == 3:
            fig = plt.figure()            
            ax = fig.add_subplot(projection="3d")
            
            valeursAuFaces = []
            
            for face in self.mesh.connectionPourAffichage:
                somme = 0
                i = 1
                for id in face:
                    somme += valeurs[id]
                    i += 1
                valeursAuFaces.append(somme/i)
            
            valeursAuFaces = np.array(valeursAuFaces)
            
            # norm = colors.BoundaryNorm(boundaries=valeursAuFaces, ncolors=256)                        

            norm = plt.Normalize(valeursAuFaces.min(), valeursAuFaces.max())
            colors = plt.cm.jet(norm(valeursAuFaces))
            
            if affichageMaillage:            
                pc = Poly3DCollection(self.mesh.new_coordo[self.mesh.connectionPourAffichage],
                                    alpha=1,
                                    cmap='jet',
                                    facecolors=colors,
                                    lw=0.5,
                                    edgecolor='black')
            else:
                pc = Poly3DCollection(self.mesh.new_coordo[self.mesh.connectionPourAffichage],
                                    alpha=1,
                                    cmap='jet',
                                    facecolors=colors,
                                    lw=0.5)
            
            fig.colorbar(pc, ax=ax)       
            ax.add_collection(pc)            
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("y [mm]")            
            
            self.__ChangeEchelle(ax)
            
        unite = ""
        if "S" in resultat:
            unite = " en Mpa"
        if "d" in resultat:
            unite = " en mm"
        ax.set_title(resultat+unite)
    
    def PlotMesh(self, deformation=False):
        """Dessine le maillage de la simulation
        """
        
        # Construit la nouvelle matrice de connection 
        # ou de coordonnées si elle n'ont pas deja ete faite
        if len(self.mesh.connectionPourAffichage) == 0:
            self.__ConstruitConnectPourAffichage()
        if len(self.mesh.new_coordo) == 0 and deformation:
            self.__ConstruitNouvelleCoordo()
        
        # ETUDE 2D
        if self.dim == 2:
            
            fig, ax = plt.subplots()
            
            ax = self.__PlotMesh2D(ax, deformation, True)      
            ax.axis('equal')
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title("Ne = {} et Nn = {}".format(self.mesh.Ne, self.mesh.Nn))
        # ETUDE 3D    
        if self.dim == 3:
            
            fig = plt.figure()            
            ax = fig.add_subplot(projection="3d")
            
            pc = Poly3DCollection(self.mesh.new_coordo[self.mesh.connectionPourAffichage],
                                  alpha=1,
                                  facecolors='c',
                                  lw=0.5,
                                  edgecolor='black')            
            ax.add_collection(pc)
            
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("y [mm]")
            ax.set_title("Ne = {} et Nn = {}".format(self.mesh.Ne, self.mesh.Nn))
            
            self.__ChangeEchelle(ax)

    def __PlotMesh2D(self,
                        ax: plt.Axes,
                        deformation: bool,
                        fill: bool):
                
        for element in self.mesh.elements:
            element = cast(Element, element)
            numérosNoeudsTriés = list(element.RenvoieLesNumsDeNoeudsTriés())
            
            if deformation:                
                x = self.mesh.new_coordo[numérosNoeudsTriés,0]
                y = self.mesh.new_coordo[numérosNoeudsTriés,1]                    
            else:
                x = self.mesh.coordo[numérosNoeudsTriés,0]
                y = self.mesh.coordo[numérosNoeudsTriés,1]
            
            ax.fill(x, y, 'c', edgecolor='black', fill=fill, lw=0.5)
        
        return ax
    
    def __ConstruitConnectPourAffichage(self):
        """Transforme la matrice de connectivité pour la passer dans le trisurf en 2D
        ou construit les faces pour la 3D
        Par exemple pour un quadrangle on construit deux triangles
        pour un triangle à 6 noeuds on construit 4 triangles
        POur la 3D on construit des faces pour passer en Poly3DCollection
        """
        
        connection = self.mesh.connection
        new_connection = []
        
        for listIdNoeuds in connection:
            npe = len(listIdNoeuds)
            
            if self.dim == 2:            
                # Tri3
                if npe == 3:
                    new_connection = connection
                    break            
                # Tri6
                elif npe == 6:
                    n1 = listIdNoeuds[0]
                    n2 = listIdNoeuds[1]
                    n3 = listIdNoeuds[2]
                    n4 = listIdNoeuds[3]
                    n5 = listIdNoeuds[4]
                    n6 = listIdNoeuds[5]

                    new_connection.append([n1, n4, n6])
                    new_connection.append([n4, n2, n5])
                    new_connection.append([n6, n5, n3])
                    new_connection.append([n4, n5, n6])                    
                # Quad4
                elif npe == 4:
                    n1 = listIdNoeuds[0]
                    n2 = listIdNoeuds[1]
                    n3 = listIdNoeuds[2]
                    n4 = listIdNoeuds[3]                

                    new_connection.append([n1, n2, n4])
                    new_connection.append([n2, n3, n4])                    
                # Quad8
                elif npe == 8:
                    n1 = listIdNoeuds[0]
                    n2 = listIdNoeuds[1]
                    n3 = listIdNoeuds[2]
                    n4 = listIdNoeuds[3]
                    n5 = listIdNoeuds[4]
                    n6 = listIdNoeuds[5]
                    n7 = listIdNoeuds[6]
                    n8 = listIdNoeuds[7]

                    new_connection.append([n5, n6, n8])
                    new_connection.append([n6, n7, n8])
                    new_connection.append([n1, n5, n8])
                    new_connection.append([n5, n2, n6])
                    new_connection.append([n6, n3, n7])
                    new_connection.append([n7, n4, n8])                    
                
            elif self.dim ==3:
                # Tetra4
                if npe == 4:
                    n1 = listIdNoeuds[0]
                    n2 = listIdNoeuds[1]
                    n3 = listIdNoeuds[2]
                    n4 = listIdNoeuds[3]
                                    
                    new_connection.append([n1 ,n2, n3])
                    new_connection.append([n1, n2, n4])
                    new_connection.append([n1, n3, n4])
                    new_connection.append([n2, n3, n4])
        
        self.mesh.connectionPourAffichage = new_connection
    
    def __ConstruitNouvelleCoordo(self, facteurDef=2):
        # Calcul des nouvelles coordonnées
        nouvelleCoordo = []        
        x = self.resultats["dx"]*facteurDef
        y = self.resultats["dy"]*facteurDef
        if self.dim == 2:
            z = np.zeros(len(x))
        elif self.dim == 3:
            z = self.resultats["dz"]*facteurDef
            
        dxyz = np.array([x, y, z]).T

        nouvelleCoordo = self.mesh.coordo + dxyz
        
        self.mesh.new_coordo = nouvelleCoordo
       
    def __ChangeEchelle(self, ax):
        """Change la taille des axes pour l'affichage 3D

        Parameters
        ----------
        ax : plt.Axes
            Axes dans lequel on va creer la figure
        """
        # Change la taille des axes
        xmin = np.min(self.mesh.new_coordo[:,0]); xmax = np.max(self.mesh.new_coordo[:,0])
        ymin = np.min(self.mesh.new_coordo[:,1]); ymax = np.max(self.mesh.new_coordo[:,1])
        zmin = np.min(self.mesh.new_coordo[:,2]); zmax = np.max(self.mesh.new_coordo[:,2])
        
        max = np.max(np.abs([xmin, xmax, ymin, ymax, zmin, zmax]))
        
        ax.set_xlim3d(xmin, xmax)
        ax.set_ylim3d(ymin, ymax)
        ax.set_zlim3d(zmin, zmax)
        
        factX = np.max(np.abs([xmin, xmax]))/max
        factY = np.max(np.abs([ymin, ymax]))/max
        factZ = np.max(np.abs([zmin, zmax]))/max
        
        ax.set_box_aspect((factX, factY, factZ))
                
    
         
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
            mesh = Mesh(dim, coordo, connect)

            simu = Simu(dim, mesh, materiau, verbosity=False)

            simu.Assemblage(epaisseur=b)

            noeud_en_L = []
            noeud_en_0 = []
            for n in simu.mesh.noeuds:            
                    n = cast(Noeud, n)
                    if n.coordo[0] == L:
                            noeud_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeud_en_0.append(n)

            simu.ConditionEnForce(noeuds=noeud_en_L, force=P, direction="Y")

            simu.ConditionEnDeplacement(noeuds=noeud_en_0, deplacement=0, direction="X")
            simu.ConditionEnDeplacement(noeuds=noeud_en_0, deplacement=0, direction="Y")

            self.simulations2DElastique.append(simu)

    def CreationDesSimusElastique3D(self):

        fichier = "part.stp"

        dim = 3

        # Paramètres géométrie
        L = 120;  #mm
        h = 13;    
        b = 13

        P = -800 #N

        # Paramètres maillage        
        taille = L

        materiau = Materiau(dim)
        
        self.simulations3DElastique = []

        for type in ModelGmsh.get_typesMaillage3D():
            modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False, verbosity=False)

            (coordo, connect) = modelGmsh.Importation3D(fichier)
            mesh = Mesh(dim, coordo, connect)

            simu = Simu(dim,mesh, materiau, verbosity=False)

            simu.Assemblage(epaisseur=b)

            noeuds_en_L = []
            noeuds_en_0 = []
            for n in simu.mesh.noeuds:
                    n = cast(Noeud, n)        
                    if n.coordo[0] == L:
                            noeuds_en_L.append(n)
                    if n.coordo[0] == 0:
                            noeuds_en_0.append(n)

            simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, direction="Z")

            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="X")
            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Y")
            simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Z")

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

        
            
