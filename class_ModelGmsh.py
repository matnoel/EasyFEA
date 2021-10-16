from typing import cast
import gmsh
import sys
import time
import numpy as np

from class_Materiau import Materiau

class ModelGmsh:
        
        def get_typesMaillage2D():
                return ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        
        def get_typesMaillage3D():
                return ["TETRA4"]

        def __init__(self,dim: int, organisationMaillage=False, affichageGmsh=False, gmshVerbosity=False, verbosity=True, typeElement="", tailleElement=0.0):
                
                assert typeElement in ModelGmsh.get_typesMaillage2D() or typeElement in ModelGmsh.get_typesMaillage3D(), "Le type d'element est inconnue"

                assert tailleElement > 0.0 , "La taille de maille doit être > 0"
                
                self.__dim = dim

                self.__typeElement = typeElement
                self.__tailleElement = tailleElement
                
                self.__organisationMaillage = organisationMaillage
                self.__affichageGmsh = affichageGmsh
                self.__verbosity = verbosity

                if verbosity:
                        print("==========================================================")
                        print("Gmsh : \n")

                gmsh.initialize()
                if gmshVerbosity == False:
                        gmsh.option.setNumber('General.Verbosity', 0)
                gmsh.model.add("model")

        def __ConstructionMaillageGmsh(self, surface=None):
                                
                type = self.__typeElement
                if self.__verbosity:
                        print("Type d'elements: {} \n".format(type))

                if type in ModelGmsh.get_typesMaillage2D():
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

                elif type in ModelGmsh.get_typesMaillage3D():

                        gmsh.model.occ.synchronize()

                        gmsh.option.setNumber("Mesh.MeshSizeMin", self.__tailleElement)
                        gmsh.option.setNumber("Mesh.MeshSizeMax", self.__tailleElement)
                        gmsh.model.mesh.generate(3)
                
                # Ouvre l'interface de gmsh si necessaire
                if '-nopopup' not in sys.argv and self.__affichageGmsh:
                        gmsh.fltk.run()   

        def __ConstructionCoordoConnect(self):
                
                start = time.time()
                
                # Récupère la liste d'élément correspondant a la bonne dimension
                types, elements, nodeTags = gmsh.model.mesh.getElements(dim=self.__dim)        

                # Construit la matrice connection
                Ne = len(elements[0])
                connect = []
                
                e = 0
                while e < Ne:
                        type, noeuds = gmsh.model.mesh.getElement(elements[0][e])
                        noeuds = list(noeuds - 1)            
                        connect.append(noeuds)
                        e += 1        

                # Construit la matrice coordonée
                noeuds, coord, parametricCoord = gmsh.model.mesh.getNodes()
                Nn = noeuds.shape[0]
                coordo = []
                
                n = 0
                while n < Nn:            
                        coord, parametricCoord = gmsh.model.mesh.getNode(noeuds[n])            
                        coordo.append(coord)
                        n += 1        
                
                end = start - time.time()
                if self.__verbosity:
                        print("\nConstruction Coordo et Connect ({:.3f} s)".format(np.abs(end)))

                gmsh.finalize()

                return (np.array(coordo), connect)

        def ConstructionRectangle(self, largeur, hauteur):
                
                start = time.time()

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
                
                self.__ConstructionMaillageGmsh(surface)

                end = start - time.time()
                if self.__verbosity:
                        print("\nConstruction Rectangle ({:.3f} s)".format(np.abs(end)))
                
                return self.__ConstructionCoordoConnect()

        def Importation3D(self,fichier=""):
                
                start = time.time()

                # Importation du fichier
                gmsh.model.occ.importShapes(fichier)

                self.__ConstructionMaillageGmsh()

                end = start - time.time()
                if self.__verbosity:
                        print("\nimportation du fichier step ({:.3f} s)".format(np.abs(end)))
                
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

            # Pour chaque type d'element 2D
            for type in ModelGmsh.get_typesMaillage3D():
                    modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=120, verbosity=False)
                    modelGmsh.Importation3D("part.stp")

    
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")   
                
        
        
        

