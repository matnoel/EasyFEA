from typing import cast
import gmsh
import sys
import numpy as np

from Element import ElementIsoparametrique
from TicTac import TicTac
from Affichage import Affichage
import matplotlib.pyplot as plt

class GmshElem:

        @staticmethod
        def __Get_ElemInFos(type):

                match type:
                        case 1: 
                                name = "SEG2"; nPe = 2; dim = 1
                        case 2: 
                                name = "TRI3"; nPe = 3; dim = 2
                        case 3: 
                                name = "QUAD4"; nPe = 4; dim = 2 
                        case 4: 
                                name = "TETRA4"; nPe = 4; dim = 3
                        case 5: 
                                name = "CUBE8"; nPe = 8; dim = 3
                        case 6: 
                                name = "PRISM6"; nPe = 6; dim = 3
                        case 7: 
                                name = "PYRA5"; nPe = 5; dim = 3
                        case 8: 
                                name = "SEG3"; nPe = 3; dim = 1
                        case 9: 
                                name = "TRI6"; nPe = 6; dim = 2
                        case 10: 
                                name = "QUAD9"; nPe = 9; dim = 2
                        case 11: 
                                name = "TETRA10"; nPe = 10; dim = 3
                        case 12: 
                                name = "CUBE27"; nPe = 27; dim = 3
                        case 13: 
                                name = "PRISM18"; nPe = 18; dim = 3
                        case 14: 
                                name = "PYRA14"; nPe = 17; dim = 3
                        case 15: 
                                name = "POINT"; nPe = 1; dim = 0
                        case 16: 
                                name = "QUAD8"; nPe = 8; dim = 2
                        case 18: 
                                name = "PRISM15"; nPe = 15; dim = 3
                        case 19: 
                                name = "PYRA13"; nPe = 13; dim = 3
                        case _: 
                                raise "Type inconnue"

                return name, nPe, dim 
        
        def __get_type(self):
                return self.__gmshId
        gmshType = property(__get_type)

        def __get_name(self):
                return GmshElem.__Get_ElemInFos(self.__gmshId)[0]
        name = property(__get_name)

        def __get_nPe(self):
                return GmshElem.__Get_ElemInFos(self.__gmshId)[1]
        nPe = property(__get_nPe)

        def __get_dim(self):
                return GmshElem.__Get_ElemInFos(self.__gmshId)[2]
        dim = property(__get_dim)

        def __get_Ne(self):
                return self.__elements.shape[0]
        Ne = property(__get_Ne)

        def __get_Nn(self):
                return self.__nodes.shape[0]
        Nn = property(__get_Nn)

        def __init__(self, gmshId: int, elementTags: np.ndarray, nodeTags: np.ndarray, coordo: np.ndarray):
                
                self.__gmshId = gmshId

                # Elements
                self.__elements = elementTags-1
                self.connect = (nodeTags-1).reshape(self.Ne,-1)
                
                # Noeuds
                self.__nodes = np.unique(nodeTags-1)

                # Test si il n'existe pas un noeud en trop
                Nmax = int(self.connect.max())
                ecart = Nmax - (self.Nn-1)
                assert ecart == 0, f"Erreur dans la récupération, il ya {ecart} noeuds de trop"

                self.coordo = np.array(coordo[self.__nodes])

class Interface_Gmsh:        

        def __init__(self,dim: int, organisationMaillage=False, affichageGmsh=False, gmshVerbosity=False, verbosity=True, typeElement=0, tailleElement=0.0):
                
                assert tailleElement > 0.0 , "La taille de maille doit être > 0"
                
                self.__dim = dim
                """dimension du model Gmsh"""

                if dim == 2:
                        self.__typeElement = ElementIsoparametrique.get_Types2D()[typeElement]
                        """type d'element"""
                elif dim == 3:
                        self.__typeElement = ElementIsoparametrique.get_Types3D()[typeElement]
                        """type d'element"""

                self.__tailleElement = tailleElement
                """taille d'element pour le maillage"""
                
                self.__organisationMaillage = organisationMaillage
                """organisation du maillage"""
                self.__affichageGmsh = affichageGmsh
                """affichage du maillage sur gmsh"""
                self.__verbosity = verbosity
                """modelGmsh peut ecrire dans la console"""

                if verbosity:
                        Affichage.NouvelleSection("Gmsh")

                gmsh.initialize()
                if gmshVerbosity == False:
                        gmsh.option.setNumber('General.Verbosity', 0)
                gmsh.model.add("model")

        def ConstructionRectangle(self, largeur, hauteur):
                
                tic = TicTac()

                # Créer les points
                p1 = gmsh.model.geo.addPoint(0, 0, 0, self.__tailleElement)
                p2 = gmsh.model.geo.addPoint(largeur, 0, 0, self.__tailleElement)
                p3 = gmsh.model.geo.addPoint(largeur, hauteur, 0, self.__tailleElement)
                p4 = gmsh.model.geo.addPoint(0, hauteur, 0, self.__tailleElement)

                # Créer les lignes reliants les points
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p4)
                l4 = gmsh.model.geo.addLine(p4, p1)

                # Créer une boucle fermée reliant les lignes     
                boucle = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

                # Créer une surface
                surface = gmsh.model.geo.addPlaneSurface([boucle])
                
                tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
                
                self.__ConstructionMaillageGmsh(surface)
                
                return self.__ConstructionCoordoConnect()

        def ConstructionRectangleAvecFissure(self, largeur, hauteur, openCrack=False):
                
                tic = TicTac()

                gmsh.model.add("square with cracks")                

                # Créer les points du rectangle
                p1 = gmsh.model.occ.addPoint(0, 0, 0, self.__tailleElement)
                p2 = gmsh.model.occ.addPoint(largeur, 0, 0, self.__tailleElement)
                p3 = gmsh.model.occ.addPoint(largeur, hauteur, 0, self.__tailleElement)
                p4 = gmsh.model.occ.addPoint(0, hauteur, 0, self.__tailleElement)

                # Creer les points de la fissure
                p5 = gmsh.model.occ.addPoint(0, hauteur/2, 0, self.__tailleElement)
                p6 = gmsh.model.occ.addPoint(largeur/2, hauteur/2, 0, self.__tailleElement)

                # Créer les lignes reliants les points pour la surface
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p5)
                l5 = gmsh.model.occ.addLine(p5, p1)

                # Créer une boucle fermée reliant les lignes pour la surface
                boucle = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5])

                # Créer une surface
                surface = gmsh.model.occ.addPlaneSurface([boucle])
                
                # Creer la ligne de fissure
                line = gmsh.model.occ.addLine(p5, p6)
                
                gmsh.model.occ.synchronize()

                # Ajoute la ligne dans la surface
                gmsh.model.mesh.embed(1, [line], 2, surface)

                if openCrack:
                        point = gmsh.model.addPhysicalGroup(0, [p5])
                        crack = gmsh.model.addPhysicalGroup(1, [line])
                        surface = gmsh.model.addPhysicalGroup(2, [surface])
                        
                
                tic.Tac("Mesh","Construction Rectangle Fissuré", self.__verbosity)

                self.__organisationMaillage=False
                
                if openCrack:
                        self.__ConstructionMaillageGmsh(surface, crack=crack, openBoundary=point)
                else:
                        self.__ConstructionMaillageGmsh(surface)
                
                return self.__ConstructionCoordoConnect()

        def Importation3D(self,fichier=""):
                
                tic = TicTac()

                # Importation du fichier
                gmsh.model.occ.importShapes(fichier)

                tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

                self.__ConstructionMaillageGmsh()

                return self.__ConstructionCoordoConnect()

        def __ConstructionMaillageGmsh(self, surface=None, crack=None, openBoundary=None):

                tic = TicTac()                

                if self.__dim == 2:
                        # Impose que le maillage soit organisé                        
                        if self.__organisationMaillage:
                                gmsh.model.geo.mesh.setTransfiniteSurface(surface)

                        # Synchronisation
                        gmsh.model.geo.synchronize()

                        if self.__typeElement in ["QUAD4","QUAD8"]:
                                gmsh.model.mesh.setRecombine(2, surface)
                        
                        # Génère le maillage
                        gmsh.model.mesh.generate(2)

                        if self.__typeElement in ["QUAD8"]:
                                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

                        if self.__typeElement in ["TRI3","QUAD4"]:
                                gmsh.model.mesh.set_order(1)
                        elif self.__typeElement in ["TRI6","QUAD8"]:
                                gmsh.model.mesh.set_order(2)

                        if crack != None:
                                gmsh.plugin.setNumber("Crack", "Dimension", self.__dim-1)
                                gmsh.plugin.setNumber("Crack", "PhysicalGroup", crack)
                                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openBoundary)
                                gmsh.plugin.setNumber("Crack", "NormalX", 0)
                                gmsh.plugin.setNumber("Crack", "NormalY", 0)
                                gmsh.plugin.setNumber("Crack", "NormalZ", 1)
                                gmsh.plugin.run("Crack")

                elif self.__dim == 3:

                        gmsh.model.occ.synchronize()

                        gmsh.option.setNumber("Mesh.MeshSizeMin", self.__tailleElement)
                        gmsh.option.setNumber("Mesh.MeshSizeMax", self.__tailleElement)
                        gmsh.model.mesh.generate(3)
                
                # Ouvre l'interface de gmsh si necessaire
                if '-nopopup' not in sys.argv and self.__affichageGmsh:
                        gmsh.fltk.run()   
                
                tic.Tac("Mesh","Construction du maillage gmsh", self.__verbosity)

        def __ConstructionCoordoConnect(self):
                """construit connect et coordo pour l'importation du maillage"""
                
                tic = TicTac()

                physicalGroups = gmsh.model.getPhysicalGroups()
                entities = np.array(gmsh.model.getEntities())

                elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements()
                nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()

                # Redimensionne sous la forme d'un tableau
                coordo = coord.reshape(-1,3)

                listGmshElem = {}
                for t, type in enumerate(elementTypes):
                        gmshElem = GmshElem(type, elementTags[t], nodeTags[t], coordo)
                        listGmshElem[gmshElem.dim] = gmshElem

                gmshElem = cast(GmshElem, listGmshElem[self.__dim])

                connect = gmshElem.connect
                coordo = gmshElem.coordo                

                gmsh.finalize()

                tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

                return coordo, connect


# TEST ==============================

import unittest
import os

class Test_ModelGmsh(unittest.TestCase):
        def setUp(self):
                pass
        
        def test_ConstructionS(self):

                from Mesh import Mesh
                
                dim = 2

                L = 120
                h = 13

                organisations=[True, False]

                for organisationMaillage in organisations:
                        # Pour chaque type d'element 2D
                        for t, type in enumerate(ElementIsoparametrique.get_Types2D()):
                                modelGmsh = Interface_Gmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                coordo, connect = modelGmsh.ConstructionRectangle(L, h)
                                mesh = Mesh(2, coordo=coordo, connect=connect, verbosity=False)

                                Affichage.Plot_NoeudsMaillage(mesh)
                                plt.pause(0.5)

                                modelGmsh2 = Interface_Gmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                modelGmsh2.ConstructionRectangleAvecFissure(L, h)

                                # Affiche le maillage

        
        def test_Importation3D(self):

                import Dossier        
            
                dim = 3

                # Pour chaque type d'element 3D
                for t, type in enumerate(ElementIsoparametrique.get_Types3D()):
                        modelGmsh = Interface_Gmsh(dim, organisationMaillage=True, typeElement=t, tailleElement=120, verbosity=False)
                        path = Dossier.GetPath()
                        fichier = path + "\\models\\part.stp" 
                        modelGmsh.Importation3D(fichier)

    
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")   
                
        
        
        

