import gmsh
import sys
import numpy as np

from Element import Element
from TicTac import TicTac
from Affichage import Affichage

class ModelGmsh:        
        
        def __init__(self,dim: int, organisationMaillage=False, affichageGmsh=False, gmshVerbosity=False, verbosity=True, typeElement=0, tailleElement=0.0):
                
                assert tailleElement > 0.0 , "La taille de maille doit être > 0"
                
                self.__dim = dim
                """dimension du model Gmsh"""

                if dim == 2:
                        self.__typeElement = Element.get_Types2D()[typeElement]
                        """type d'element"""
                elif dim == 3:
                        self.__typeElement = Element.get_Types3D()[typeElement]
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

        def ConstructionRectangleAvecFissure(self, largeur, hauteur):
                
                tic = TicTac()

                # Créer les points
                p1 = gmsh.model.geo.addPoint(0, 0, 0, self.__tailleElement)
                p2 = gmsh.model.geo.addPoint(largeur, 0, 0, self.__tailleElement)
                p3 = gmsh.model.geo.addPoint(largeur, hauteur, 0, self.__tailleElement)
                p4 = gmsh.model.geo.addPoint(0, hauteur, 0, self.__tailleElement)

                p5 = gmsh.model.geo.addPoint(0, hauteur/2, 0, self.__tailleElement)
                p6 = gmsh.model.geo.addPoint(largeur/2, hauteur/2, 0, self.__tailleElement)        
                

                # Créer les lignes reliants les points
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p4)
                l4 = gmsh.model.geo.addLine(p4, p5)
                l5 = gmsh.model.geo.addLine(p5, p1)
                
                l6 = gmsh.model.geo.addLine(p5, p6)

                # Créer une boucle fermée reliant les lignes     
                boucle = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5])

                # Créer une surface
                surface = gmsh.model.geo.addPlaneSurface([boucle])

                gmsh.model.geo.synchronize()

                # surface = gmsh.model.mesh.embed(1, [l6], 2, surface)
                gmsh.model.mesh.embed(1, [l6], 2, surface)
                
                tic.Tac("Mesh","Construction Rectangle Fissuré", self.__verbosity)

                self.__organisationMaillage=False
                
                self.__ConstructionMaillageGmsh(surface)
                
                return self.__ConstructionCoordoConnect()

        def Importation3D(self,fichier=""):
                
                tic = TicTac()

                # Importation du fichier
                gmsh.model.occ.importShapes(fichier)

                tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

                self.__ConstructionMaillageGmsh()

                return self.__ConstructionCoordoConnect()

        def __ConstructionMaillageGmsh(self, surface=None):

                tic = TicTac()

                type = self.__typeElement
                if self.__verbosity:
                        print("\nType d'elements: {}".format(type))

                if self.__dim == 2:
                        # Impose que le maillage soit organisé                        
                        if self.__organisationMaillage:
                                gmsh.model.geo.mesh.setTransfiniteSurface(surface)
                                        

                        # Synchronisation
                        gmsh.model.geo.synchronize()

                        if type in ["QUAD4","QUAD8"]:                        
                                gmsh.model.mesh.setRecombine(2, surface)

                        # Génère le maillage
                        gmsh.model.mesh.generate(2)

                        if type in ["QUAD8"]:
                                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

                        if type in ["TRI3","QUAD4"]:
                                gmsh.model.mesh.set_order(1)
                        elif type in ["TRI6","QUAD8"]:
                                gmsh.model.mesh.set_order(2)

                elif self.__dim == 3:

                        gmsh.model.occ.synchronize()

                        gmsh.option.setNumber("Mesh.MeshSizeMin", self.__tailleElement)
                        gmsh.option.setNumber("Mesh.MeshSizeMax", self.__tailleElement)
                        gmsh.model.mesh.generate(3)
                
                # Ouvre l'interface de gmsh si necessaire
                if '-nopopup' not in sys.argv and self.__affichageGmsh:
                        gmsh.fltk.run()   
                
                tic.Tac("Mesh","Construction du maillage gmsh", self.__verbosity)

        def __ConstructionCoordoConnect(self, option = 2):
                """construit connect et coordo pour l'importation du maillage"""
                
                tic = TicTac()

                if option == 1:
                        # Construit Connect
                        types, elements, nodeTags = gmsh.model.mesh.getElements(self.__dim)
                        Ne = len(elements[0])
                        connect = np.array(nodeTags).reshape(Ne,-1)-1

                        # Construit la matrice coordonée
                        noeuds, coord, parametricCoord = gmsh.model.mesh.getNodes()
                        Nn = noeuds.shape[0]                
                        coordo = coord.reshape(Nn,-1)

                else:
                        # type = gmsh.model.mesh.getElementTypes(self.__dim)[-1]

                        # Construit Connect
                        # elements, nodeTags = gmsh.model.mesh.getElementsByType(type)
                        types, elements, nodeTags = gmsh.model.mesh.getElements(self.__dim)
                        Ne = len(elements[-1])
                        connect = nodeTags[-1].reshape(Ne,-1)-1

                        # Construit la matrice coordonée
                        # noeuds, coord, parametricCoord = gmsh.model.mesh.getNodesByElementType(type)
                        noeuds, coord, parametricCoord = gmsh.model.mesh.getNodes()
                        Nn = noeuds.shape[0]                
                        coordo = coord.reshape(Nn,-1)

                assert connect.max()+1 == Nn, "Erreur dans la récupération"

                gmsh.finalize()

                tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

                return [np.array(np.array(coordo)), connect]



# TEST ==============================

import unittest
import os

class Test_ModelGmsh(unittest.TestCase):
        def setUp(self):
                pass
        
        def test_ConstructionS(self):
                
                dim = 2

                L = 120
                h = 13

                organisations=[True, False]

                for organisationMaillage in organisations:
                        # Pour chaque type d'element 2D
                        for t, type in enumerate(Element.get_Types2D()):
                                modelGmsh = ModelGmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                modelGmsh.ConstructionRectangle(L, h)

                                modelGmsh2 = ModelGmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                modelGmsh2.ConstructionRectangleAvecFissure(L, h)

        
        def test_Importation3D(self):

                import Dossier        
            
                dim = 3

                # Pour chaque type d'element 3D
                for t, type in enumerate(Element.get_Types3D()):
                        modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=t, tailleElement=120, verbosity=False)
                        path = Dossier.GetPath()
                        fichier = path + "\\models\\part.stp" 
                        modelGmsh.Importation3D(fichier)

    
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")   
                
        
        
        

