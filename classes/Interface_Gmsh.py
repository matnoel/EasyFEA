
import gmsh
import sys
import numpy as np
import scipy.sparse as sp

import Dossier
from Geom import *
from GroupElem import GroupElem
from Mesh import Mesh
from TicTac import TicTac
import Affichage
import matplotlib.pyplot as plt

class Interface_Gmsh:   

        def __init__(self, affichageGmsh=False, gmshVerbosity=False, verbosity=True):                
                
                self.__affichageGmsh = affichageGmsh
                """affichage du maillage sur gmsh"""
                self.__gmshVerbosity = gmshVerbosity
                """gmsh peut ecrire dans la console"""
                self.__verbosity = verbosity
                """modelGmsh peut ecrire dans la console"""

                if verbosity:
                        Affichage.NouvelleSection("Maillage Gmsh")

        def __initGmsh(self):
                gmsh.initialize()
                if self.__gmshVerbosity == False:
                        gmsh.option.setNumber('General.Verbosity', 0)
                gmsh.model.add("model")
        
        def __CheckType(self, dim: int, elemType: str):
                if dim == 2:
                        assert elemType in GroupElem.get_Types2D()                        
                elif dim == 3:
                        assert elemType in GroupElem.get_Types3D()

        def Importation3D(self,fichier="", elemType="TETRA4", tailleElement=0.0):
                """Importe depuis un 3D

                elemTypes = ["TETRA4"]
                
                Returns:
                    Mesh: mesh
                """

                self.__initGmsh()

                assert tailleElement >= 0.0, "Doit être supérieur ou égale à 0"
                self.__CheckType(3, elemType)
                
                tic = TicTac()

                # Importation du fichier
                gmsh.model.occ.importShapes(fichier)

                gmsh.option.setNumber("Mesh.MeshSizeMin", tailleElement)
                gmsh.option.setNumber("Mesh.MeshSizeMax", tailleElement)

                tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

                self.__Construction_MaillageGmsh(3, elemType)

                return cast(Mesh, self.__Recuperation_Maillage())

        def Rectangle(self, domain: Domain, elemType="TRI3", isOrganised=False):

                """Construit un rectangle

                elemTypes = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
                
                Returns:
                    Mesh: mesh
                """

                self.__initGmsh()                
                
                self.__CheckType(2, elemType)

                tic = TicTac()

                pt1 = domain.pt1
                pt2 = domain.pt2

                assert pt1.z == 0 and pt2.z == 0

                tailleElement = domain.taille

                # Créer les points
                p1 = gmsh.model.occ.addPoint(pt1.x, pt1.y, 0, tailleElement)
                p2 = gmsh.model.occ.addPoint(pt2.x, pt1.y, 0, tailleElement)
                p3 = gmsh.model.occ.addPoint(pt2.x, pt2.y, 0, tailleElement)
                p4 = gmsh.model.occ.addPoint(pt1.x, pt2.y, 0, tailleElement)

                # Créer les lignes reliants les points
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p1)

                # Créer une boucle fermée reliant les lignes     
                boucle = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

                # Créer une surface
                surface = gmsh.model.occ.addPlaneSurface([boucle])

                surface = gmsh.model.addPhysicalGroup(2, [surface])
                
                tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
                
                self.__Construction_MaillageGmsh(2, elemType, surface=surface, isOrganised=isOrganised)
                
                return cast(Mesh, self.__Recuperation_Maillage())

        def RectangleAvecFissure(self, domain: Domain, crack: Line,
        elemType="TRI3", openCrack=False, isOrganised=False, filename=""):

                """Construit un rectangle avec une fissure ouverte ou non

                elemTypes = ["TRI3", "TRI6", "QUAD4", "QUAD8"]
                
                Returns:
                    Mesh: mesh
                """

                self.__initGmsh()                
                
                self.__CheckType(2, elemType)
                
                tic = TicTac()

                # Domain
                pt1 = domain.pt1
                pt2 = domain.pt2
                assert pt1.z == 0 and pt2.z == 0

                # Crack
                pf1 = crack.pt1
                pf2 = crack.pt2
                assert pf1.z == 0 and pf2.z == 0

                domainSize = domain.taille
                crackSize = crack.taille

                # Create the points of the rectangle
                p1 = gmsh.model.occ.addPoint(pt1.x, pt1.y, 0, domainSize)
                p2 = gmsh.model.occ.addPoint(pt2.x, pt1.y, 0, domainSize)
                p3 = gmsh.model.occ.addPoint(pt2.x, pt2.y, 0, domainSize)
                p4 = gmsh.model.occ.addPoint(pt1.x, pt2.y, 0, domainSize)

                # Create the lines connecting the points for the surface
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p1)                

                # loop for surface
                loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

                # creat surface
                surface = gmsh.model.occ.addPlaneSurface([loop])

                # Create the crack points
                p5 = gmsh.model.occ.addPoint(pf1.x, pf1.y, 0, crackSize)
                p6 = gmsh.model.occ.addPoint(pf2.x, pf2.y, 0, crackSize)

                # Create the line for the crack
                crack = gmsh.model.occ.addLine(p5, p6)

                listeOpen=[]
                if pf1.isOpen:
                        o, m = gmsh.model.occ.fragment([(0, p5), (1, crack)], [(2, surface)])
                        listeOpen.append(p5)
                if pf2.isOpen:
                        o, m = gmsh.model.occ.fragment([(0, p6), (1, crack)], [(2, surface)])
                        listeOpen.append(p6)
                gmsh.model.occ.synchronize()
                # Adds the line to the surface
                gmsh.model.mesh.embed(1, [crack], 2, surface)

                surface = gmsh.model.addPhysicalGroup(2, [surface], 100)
                crack = gmsh.model.addPhysicalGroup(1, [crack], 101)
                if len(listeOpen)==0:
                        point=None                        
                else:
                        point = gmsh.model.addPhysicalGroup(0, listeOpen, 102)
                
                tic.Tac("Mesh","Construction rectangle fissuré", self.__verbosity)
                
                if openCrack:
                        self.__Construction_MaillageGmsh(2, elemType, surface=surface, crack=crack, openBoundary=point, isOrganised=isOrganised)
                else:
                        self.__Construction_MaillageGmsh(2, elemType, surface=surface, isOrganised=isOrganised)
                
                return cast(Mesh, self.__Recuperation_Maillage(filename))

        def PlaqueTrouée(self, domain: Domain, circle: Circle, 
        elemType="TRI3", isOrganised=False, filename=""):
                
                self.__initGmsh()
                self.__CheckType(2, elemType)

                tic = TicTac()

                # Domain
                pt1 = domain.pt1
                pt2 = domain.pt2
                assert pt1.z == 0 and pt2.z == 0

                # Circle
                center = circle.center
                diam = circle.diam
                rayon = diam/2
                assert center.z == 0

                # Create the points of the rectangle
                p1 = gmsh.model.occ.addPoint(pt1.x, pt1.y, 0, domain.taille)
                p2 = gmsh.model.occ.addPoint(pt2.x, pt1.y, 0, domain.taille)
                p3 = gmsh.model.occ.addPoint(pt2.x, pt2.y, 0, domain.taille)
                p4 = gmsh.model.occ.addPoint(pt1.x, pt2.y, 0, domain.taille)

                # Créer les lignes reliants les points pour la surface
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p4)
                l4 = gmsh.model.occ.addLine(p4, p1)

                # Create a closed loop connecting the lines for the surface
                loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

                # Points cercle                
                p5 = gmsh.model.occ.addPoint(center.x, center.y, 0, circle.taille) #centre
                p6 = gmsh.model.occ.addPoint(center.x-rayon, center.y, 0, circle.taille)
                p7 = gmsh.model.occ.addPoint(center.x+rayon, center.y, 0, circle.taille)

                # Lignes cercle                
                l5 = gmsh.model.occ.addCircleArc(p6, p5, p7)
                l6 = gmsh.model.occ.addCircleArc(p7, p5, p6)
                lignecercle = gmsh.model.occ.addCurveLoop([l5,l6])

                # cercle = gmsh.model.occ.addCircle(center.x, center.y, center.z, diam/2)
                # lignecercle = gmsh.model.occ.addCurveLoop([cercle])
                # gmsh.option.setNumber("Mesh.MeshSizeMin", domain.taille)
                # gmsh.option.setNumber("Mesh.MeshSizeMax", circle.taille)

                # Create a surface
                surface = gmsh.model.occ.addPlaneSurface([loop,lignecercle])

                gmsh.model.occ.synchronize()
                gmsh.model.occ.remove([(0,p5)], False)


                tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

                self.__Construction_MaillageGmsh(2, elemType, surface=surface, isOrganised=isOrganised)

                return cast(Mesh, self.__Recuperation_Maillage(filename))
                

        def __Construction_MaillageGmsh(self, dim: int, elemType: str, isOrganised=False,
        surface=None, crack=None, openBoundary=None):

                tic = TicTac()

                match dim:
                        case 2:

                                # Impose que le maillage soit organisé                        
                                if isOrganised:
                                        # TODO Ne fonctionne plus depsuis le passage à occ
                                        # gmsh.model.geo.synchronize()
                                        # groups = gmsh.model.getPhysicalGroups()
                                        # entities = gmsh.model.getEntitiesForPhysicalGroup(2, surface)
                                        gmsh.model.geo.mesh.setTransfiniteSurface(surface)

                                # Synchronisation
                                gmsh.model.occ.synchronize()

                                if elemType in ["QUAD4","QUAD8"]:
                                        try:
                                                gmsh.model.mesh.setRecombine(2, surface)
                                        except Exception:
                                                # Récupère la surface
                                                entities = gmsh.model.getEntities()
                                                surface = entities[-1][-1]
                                                gmsh.model.mesh.setRecombine(2, surface)
                                
                                # Génère le maillage
                                gmsh.model.mesh.generate(2)

                                if elemType in ["QUAD8"]:
                                        gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

                                if elemType in ["TRI3","QUAD4"]:
                                        gmsh.model.mesh.set_order(1)
                                elif elemType in ["TRI6","QUAD8"]:
                                        gmsh.model.mesh.set_order(2)

                                if crack != None:
                                        gmsh.plugin.setNumber("Crack", "Dimension", dim-1)
                                        gmsh.plugin.setNumber("Crack", "PhysicalGroup", crack)
                                        if openBoundary != None:
                                                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openBoundary)
                                        # gmsh.plugin.setNumber("Crack", "NormalX", 0)
                                        # gmsh.plugin.setNumber("Crack", "NormalY", 0)
                                        # gmsh.plugin.setNumber("Crack", "NormalZ", 1)
                                        gmsh.plugin.run("Crack")
                                        # gmsh.write("meshhh.msh")
                                        # self.__initGmsh()
                                        # gmsh.open("meshhh.msh")
                        
                        case 3:
                                gmsh.model.occ.synchronize()                                
                                gmsh.model.mesh.generate(3)
                
                # Ouvre l'interface de gmsh si necessaire
                if '-nopopup' not in sys.argv and self.__affichageGmsh:
                        gmsh.fltk.run()   
                
                tic.Tac("Mesh","Construction du maillage gmsh", self.__verbosity)

        def __Recuperation_Maillage(self, filename=""):
                """Construction du maillage

                Parameters
                ----------
                filename : str, optional
                    nom du fichier mesh, by default ""

                Returns
                -------
                Mesh
                    Maillage crée
                """     

                # Ancienne méthode qui beugait
                # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1926
                
                tic = TicTac()

                dict_groupElem = {}
                elementTypes = gmsh.model.mesh.getElementTypes()
                nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()

                nodes = np.array(nodes-1) #numéro des noeuds
                Nn = nodes.shape[0] #Nombre de noeuds

                # Organise les noeuds du plus petits au plus grand
                sortedIndices = np.argsort(nodes)
                sortedNodes = nodes[sortedIndices]

                # Ici on va detecter les saut potententiel dans la numérotations des noeuds
                # Exemple 0 1 2 3 4 5 6 8 Ici on va detecter l'ecart 
                ecart = sortedNodes - np.arange(Nn)

                # Les noeuds a changer sont les noeuds ou l'écart est > 0
                noeudsAChanger = np.where(ecart>0)[0]

                # Construit une matrice dans laquelle on va stocker dans la première colonnes
                # les anciennes valeurs et dans la 2 eme les nouvellles
                changes = np.zeros((noeudsAChanger.shape[0],2), dtype=int)
                changes[:,0] = sortedNodes[noeudsAChanger]
                changes[:,1] = noeudsAChanger

                # On applique le changement
                nodes = np.array(sortedNodes - ecart, dtype=int)

                # On construit la matrice de coordonnées de tout les noeuds utilisé dans la maillage
                # Noeuds utilisé en 1D 2D et 3D
                coord = coord.reshape(-1,3)
                coordo = coord[sortedIndices]

                # Construit les elements
                for gmshId in elementTypes:
                                                
                        # Récupère le numéros des elements et la matrice de connection
                        elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
                        elementTags = np.array(elementTags-1, dtype=int)
                        nodeTags = np.array(nodeTags-1, dtype=int)                                

                        # Elements
                        Ne = elementTags.shape[0] #nombre d'élements
                        elements = elementTags
                        nPe = GroupElem.Get_ElemInFos(gmshId)[1] # noeuds par elements
                        
                        # Construit connect et changes les indices nécessaires
                        connect = nodeTags.reshape(Ne, nPe)
                        for indice in range(changes.shape[0]):
                                old = changes[indice,0]
                                new = changes[indice, 1]
                                l, c = np.where(connect==old)
                                connect[l, c] = new
                        
                        # TODO A tester avec l, c = np.where(connect==changes[:,0])
                        
                        # Noeuds            
                        nodes = np.unique(nodeTags)

                        # Verifie que les numéros des noeuds max est bien atteignable dans coordo
                        Nmax = nodes.max()
                        assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"
                        
                        groupElem = GroupElem(gmshId, connect, elements, coordo, nodes)
                        dict_groupElem[groupElem.dim] = groupElem
 

                gmsh.finalize()

                tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

                mesh = Mesh(dict_groupElem, self.__verbosity)

                return mesh
        
        @staticmethod
        def Construction2D(L=10, h=10, taille=3):

                interfaceGmsh = Interface_Gmsh(verbosity=False)

                list_mesh2D = []
                
                domain = Domain(Point(0,0,0), Point(L, h, 0), taille=taille)
                line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), taille=taille)
                circle = Circle(Point(x=L/2, y=h/2), L/3, taille=taille)

                # Pour chaque type d'element 2D
                for t, elemType in enumerate(GroupElem.get_Types2D()):
                        for isOrganised in [True, False]:
                                
                                mesh = interfaceGmsh.Rectangle(domain=domain, elemType=elemType, isOrganised=isOrganised)
                                mesh2 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=False)
                                mesh3 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=True)
                                mesh4 = interfaceGmsh.PlaqueTrouée(domain=domain, circle=circle, elemType=elemType, isOrganised=isOrganised)

                                for m in [mesh, mesh2, mesh3, mesh4]:
                                        list_mesh2D.append(m)
                
                return list_mesh2D

        @staticmethod
        def Construction3D():
                # Pour chaque type d'element 3D

                list_mesh3D = []
                for t, elemType in enumerate(GroupElem.get_Types3D()):
                        interfaceGmsh = Interface_Gmsh(verbosity=False)
                        path = Dossier.GetPath()
                        fichier = path + "\\models\\part.stp" 
                        mesh = interfaceGmsh.Importation3D(fichier, elemType=elemType, tailleElement=120)
                        list_mesh3D.append(mesh)
        
                return list_mesh3D
                        
        


# TEST ==============================

import unittest
import os



class Test_ModelGmsh(unittest.TestCase):


        def setUp(self):
                self.list_mesh2D = Interface_Gmsh.Construction2D()
                self.list_mesh3D = Interface_Gmsh.Construction3D()
        
        def test_Construction2D(self):
                nbMesh = len(self.list_mesh2D)
                nrows = 4
                ncols = 8
                assert nbMesh == nrows*ncols
                fig, ax = plt.subplots(nrows, ncols)
                lignes = np.repeat(np.arange(nrows), ncols)
                colonnes = np.repeat(np.arange(ncols).reshape(1,-1), nrows, axis=0).reshape(-1)
                for m, mesh2D in enumerate(self.list_mesh2D):
                        axx = ax[lignes[m],colonnes[m]]
                        Affichage.Plot_Maillage(mesh2D, ax= axx)
                        Affichage.Plot_NoeudsMaillage(mesh2D, showId=False, ax=axx, c='black')
                        axx.set_title("")
                        axx.get_xaxis().set_visible(False)
                        axx.get_yaxis().set_visible(False)
                        plt.pause(0.00005)
                
                plt.show()
        
        def test_Importation3D(self):
                for mesh3D in self.list_mesh3D:
                        Affichage.Plot_NoeudsMaillage(mesh3D, showId=True)
                        plt.pause(0.00005)
           
if __name__ == '__main__':        
    try:
        os.system("cls")
        unittest.main(verbosity=2)
    except:
        print("")   
                
        
        
        

