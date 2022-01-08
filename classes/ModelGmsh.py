import gmsh
import sys
import numpy as np

try:
        from classes.Element import Element
        from classes.Affichage import Affichage
        from classes.TicTac import TicTac
except:
        from Element import Element
        from Affichage import Affichage
        from TicTac import TicTac
        
        
class ModelGmsh:
        
        def __init__(self,dim: int, organisationMaillage=False, affichageGmsh=False, gmshVerbosity=False, verbosity=True, typeElement=0, tailleElement=0.0):
                
                assert tailleElement > 0.0 , "La taille de maille doit être > 0"
                
                self.__dim = dim

                self.__typeElement = Element.get_Types(dim)[typeElement]
                self.__tailleElement = tailleElement
                
                self.__organisationMaillage = organisationMaillage
                self.__affichageGmsh = affichageGmsh
                self.__verbosity = verbosity

                if verbosity:
                        Affichage.NouvelleSection("Gmsh")

                gmsh.initialize()
                if gmshVerbosity == False:
                        gmsh.option.setNumber('General.Verbosity', 0)
                gmsh.model.add("model")

        def __ConstructionMaillageGmsh(self, surface=None):

                tic = TicTac()

                type = self.__typeElement
                if self.__verbosity:
                        print("\nType d'elements: {}".format(type))

                if type in Element.get_Types(self.__dim):
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

                elif Element.get_Types(self.__dim):

                        gmsh.model.occ.synchronize()

                        gmsh.option.setNumber("Mesh.MeshSizeMin", self.__tailleElement)
                        gmsh.option.setNumber("Mesh.MeshSizeMax", self.__tailleElement)
                        gmsh.model.mesh.generate(3)
                
                # Ouvre l'interface de gmsh si necessaire
                if '-nopopup' not in sys.argv and self.__affichageGmsh:
                        gmsh.fltk.run()   
                
                tic.Tac("Construction du maillage gmsh", self.__verbosity)

        def __ConstructionCoordoConnect(self):
                
                tic = TicTac()

                # Récupère la liste d'élément correspondant a la bonne dimension
                types, elements, nodeTags = gmsh.model.mesh.getElements(dim=self.__dim)        

                # Construit la matrice connection
                Ne = len(elements[0])
                connect = []
                for e in range(Ne):
                        type, noeuds = gmsh.model.mesh.getElement(elements[0][e])
                        noeuds = list(noeuds - 1)            
                        connect.append(noeuds)                        

                # Construit la matrice coordonée
                noeuds, coord, parametricCoord = gmsh.model.mesh.getNodes()
                Nn = noeuds.shape[0]
                coordo = []
                for n in range(Nn):
                        coord, parametricCoord = gmsh.model.mesh.getNode(noeuds[n])
                        coordo.append(coord)
                # coordo = [gmsh.model.mesh.getNode(noeuds[n])[0] for n in range(Nn)]        
                
                gmsh.finalize()

                tic.Tac("Récupération du maillage gmsh", self.__verbosity)

                return [np.array(np.array(coordo)), connect]

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
                
                tic.Tac("Construction Rectangle", self.__verbosity)
                
                self.__ConstructionMaillageGmsh(surface)
                
                return self.__ConstructionCoordoConnect()

        def Importation3D(self,fichier=""):
                
                tic = TicTac()

                # Importation du fichier
                gmsh.model.occ.importShapes(fichier)

                tic.Tac("Importation du fichier step", self.__verbosity)

                self.__ConstructionMaillageGmsh()

                return self.__ConstructionCoordoConnect()

# TEST ==============================

import unittest
import os

class Test_ModelGmsh(unittest.TestCase):
        def setUp(self):
                pass
        
        def test_ConstructionRectangle(self):
                
                dim = 2

                L = 120
                h = 13

                # Pour chaque type d'element 2D
                for type in ModelGmsh.get_typesMaillage2D():
                        modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=L, verbosity=False)
                        modelGmsh.ConstructionRectangle(L, h)
        
        def test_Importation3D(self):
            
            dim = 3

            # Pour chaque type d'element 3D
            for type in ModelGmsh.get_typesMaillage3D():
                    modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=120, verbosity=False)
                    modelGmsh.Importation3D("part.stp")

    
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")   
                
        
        
        

