import gmsh
import sys
import numpy as np

from Element import Element
from TicTac import TicTac
from Affichage import Affichage

class Interface_Gmsh:        
        
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

        def ConstructionRectangleAvecFissure(self, largeur, hauteur, isEdgeCrack=False):
                
                tic = TicTac()

                gmsh.model.add("square with cracks")

                # surf1 = 1
                # gmsh.model.occ.addRectangle(0, 0, 0, 2, 2, surf1)

                # pt1 = gmsh.model.occ.addPoint(0, 1, 0)
                # pt2 = gmsh.model.occ.addPoint(1, 1, 0)
                # line1 = gmsh.model.occ.addLine(pt1, pt2)

                # o, m = gmsh.model.occ.fragment([(2, surf1)], [(1, line1)])
                # gmsh.model.occ.synchronize()

                # # m contains, for each input entity (surf1, line1 and line2), the child entities
                # # (if any) after the fragmentation, as lists of tuples. To apply the crack
                # # plugin we group all the intersecting lines in a physical group

                # new_surf = m[0][0][1]
                # new_lines = [item[1] for sublist in m[1:] for item in sublist]

                # gmsh.model.addPhysicalGroup(2, [new_surf], 100)
                # gmsh.model.addPhysicalGroup(1, new_lines, 101)

                # gmsh.model.mesh.generate(2)

                # gmsh.plugin.setNumber("Crack", "PhysicalGroup", 101)
                # gmsh.plugin.run("Crack")

                # if '-nopopup' not in sys.argv:
                # gmsh.fltk.run()

                # gmsh.finalize()

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
                
                # Creer la ligne de fissure
                line = gmsh.model.occ.addLine(p5, p6, 120)

                # Créer une boucle fermée reliant les lignes pour la surface
                boucle = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5])
                # boucle = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

                # Créer une surface
                surface = gmsh.model.occ.addPlaneSurface([boucle])
                
                gmsh.model.occ.synchronize()

                # Ajoute la ligne dans la surface
                gmsh.model.mesh.embed(1, [line], 2, surface)

                surface = 10101
                gmsh.model.addPhysicalGroup(2, [surface], surface)
                crack = 1000
                gmsh.model.addPhysicalGroup(1, [line], crack)
                point = 302020
                gmsh.model.addPhysicalGroup(0, [p5], point)

                        
                
                tic.Tac("Mesh","Construction Rectangle Fissuré", self.__verbosity)

                self.__organisationMaillage=False
                
                if isEdgeCrack:
                        self.__ConstructionMaillageGmsh(surface, openCrack=crack, openPoint=point)
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

        def __ConstructionMaillageGmsh(self, surface=None, openCrack=None, openPoint=None):

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

                        if openCrack != None:
                                gmsh.plugin.setNumber("Crack", "Dimension", self.__dim-1)
                                gmsh.plugin.setNumber("Crack", "PhysicalGroup", openCrack)
                                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openPoint)
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

                option=1

                # Construit Connect
                types, elements, nodeTags = gmsh.model.mesh.getElements(self.__dim)     # elements, nodeTags = gmsh.model.mesh.getElementsByType(type)
                Ne = len(elements[-1])
                connect = nodeTags[-1].reshape(Ne,-1)-1
                nPe = connect.shape[1]

                # Construit la matrice coordonée
                noeuds, coord, parametricCoord = gmsh.model.mesh.getNodes()     # noeuds, coord, parametricCoord = gmsh.model.mesh.getNodesByElementType(type)
                Nn = int(coord.size/3)
                coordo = coord.reshape(Nn,3)

                # import matplotlib.pyplot as plt

                # fig, ax = plt.subplots()

                # noeuds = np.arange(Nn)

                # ax.scatter(coordo[:,0], coordo[:,1], marker='.')

                # for n in noeuds: ax.text(coordo[n,0], coordo[n,1], str(n))

                # ax.scatter(coordo[-1,0], coordo[-1,1], marker='.', c='red')
                # ax.text(coordo[-1,0], coordo[-1,1], str(Nn))
                
                # plt.show()

                # lignes = np.where(np.max(connect, axis=1) != connect.max())                
                # connect = connect[lignes,:].reshape(-1,nPe)
                
                # coordo = coordo[range(Nn-1),:]

                Nmax = int(connect.max())
                assert Nmax == Nn-1, "Erreur dans la récupération"

                gmsh.finalize()

                tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

                return [coordo, connect]



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
                                modelGmsh = Interface_Gmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                modelGmsh.ConstructionRectangle(L, h)

                                modelGmsh2 = Interface_Gmsh(dim, organisationMaillage=organisationMaillage, typeElement=t, tailleElement=L, verbosity=False)
                                modelGmsh2.ConstructionRectangleAvecFissure(L, h)

        
        def test_Importation3D(self):

                import Dossier        
            
                dim = 3

                # Pour chaque type d'element 3D
                for t, type in enumerate(Element.get_Types3D()):
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
                
        
        
        

