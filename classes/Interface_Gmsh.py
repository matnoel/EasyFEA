
from inspect import stack
from pyexpat import model
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

    def Importation3D(self,fichier="", elemType="TETRA4", tailleElement=0.0, folder=""):
        """importation du fichier 3D

        Args:
            fichier (str, optional): fichier 3D en .stp ou autres. Defaults to "".
            elemType (str, optional): type d'element utilisé. Defaults to "TETRA4" in ["TETRA4", "HEXA8", "PRISM6"].
            tailleElement (float, optional): taille d'element a utiliser. Defaults to 0.0.
            folder (str, optional): fichier de sauvegarde dans lequel on mets le .msh . Defaults to "".

        Returns:
            Mesh: maillage construit
        """
        
        # Importe depuis un 3D

        # elemTypes = 
        
        # Returns:
        #     Mesh: mesh


        assert elemType =="TETRA4", "Lorsqu'on importe une pièce on ne peut utiliser que du TETRA4"

        # TODO Permettre d'autres maillage ?

        self.__initGmsh()

        assert tailleElement >= 0.0, "Doit être supérieur ou égale à 0"
        self.__CheckType(3, elemType)
        
        tic = TicTac()

        # Importation du fichier
        gmsh.model.occ.importShapes(fichier)

        gmsh.option.setNumber("Mesh.MeshSizeMin", tailleElement)
        gmsh.option.setNumber("Mesh.MeshSizeMax", tailleElement)

        tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def Poutre3D(self, domain: Domain, extrude=[0,0,1], nCouches=1, elemType="HEXA8", isOrganised=True, folder=""):
        """Creer un 3D depuis un domaine que l'on extrude

        Args:
            domain (Domain): surface de base qui sera extrudé
            extrude (list, optional): valeurs de l'extrustion suivant x y z dans lordre. Defaults to [0,0,1].
            nCouches (int, optional): nombre de couches dans l'extrusion. Defaults to 1.
            elemType (str, optional): type delement. Defaults to "HEXA8" in ["TETRA4", "HEXA8", "PRISM6"].
            isOrganised (bool, optional): le maillage est organisé. Defaults to True.
            folder (str, optional): fichier de sauvegarde dans lequel on mets le .msh . Defaults to "".

        Returns:
            Mesh: maillage construit
        """

        self.__initGmsh()
        self.__CheckType(3, elemType)
        
        tic = TicTac()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Rectangle(domain, elemType="TRI3", isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(gmsh.model.geo, surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())
    
    
    def __Extrusion(self, factory, surfaces: list, extrude=[0,0,1], elemType="HEXA8", isOrganised=True, nCouches=1):
        
        if isinstance(gmsh.model.geo, factory):
            isOrganised = True
            factory = cast(gmsh.model.geo, factory)
        elif isinstance(gmsh.model.occ, factory):
            isOrganised = False
            factory = cast(gmsh.model.occ, factory)

        for surf in surfaces:

            if isOrganised:

                factory.synchronize()

                # points = np.array(gmsh.model.getEntities(0))[:,1]
                # factory.mesh.setTransfiniteSurface(surf, cornerTags=points)

                gmsh.model.geo.mesh.setTransfiniteSurface(surf)

            if elemType in ["HEXA8","PRISM6"]:
                # ICI si je veux faire des PRISM6 J'ai juste à l'aisser l'option activée
                numElements = [nCouches]
                combine = True
            elif elemType == "TETRA4":
                numElements = []
                combine = False
            
            # Creer les nouveaux elements pour l'extrusion
            # nCouches = np.max([np.ceil(np.abs(extrude[2] - domain.taille)), 1])
            extru = factory.extrude([(2, surf)], extrude[0], extrude[1], extrude[2], recombine=combine, numElements=numElements)


    def Rectangle(self, domain: Domain, elemType="TRI3", isOrganised=False, folder="", returnSurfaces=False):
        """Construit un rectangle et renvoie le maillage

        Args:
            domain (Domain): domaine renseigné
            elemType (str, optional): type d'element utilisé. Defaults to "TRI3" dans ["TRI3", "TRI6", "QUAD4", "QUAD8"]
            isOrganised (bool, optional): le maillage est il organisé. Defaults to False.
            folder (str, optional): fichier de sauvegarde dans lequel on mets le .msh . Defaults to "".
            returnSurfaces (bool, optional): Renvoie les surfaces crées. Defaults to False.

        Returns:
            Mesh: maillage construit
        """

        self.__initGmsh()                
        
        self.__CheckType(2, elemType)

        tic = TicTac()    

        pt1 = domain.pt1
        pt2 = domain.pt2

        # assert pt1.z == 0 and pt2.z == 0

        tailleElement = domain.taille

        # Créer les points
        p1 = gmsh.model.geo.addPoint(pt1.x, pt1.y, 0, tailleElement)
        p2 = gmsh.model.geo.addPoint(pt2.x, pt1.y, 0, tailleElement)
        p3 = gmsh.model.geo.addPoint(pt2.x, pt2.y, 0, tailleElement)
        p4 = gmsh.model.geo.addPoint(pt1.x, pt2.y, 0, tailleElement)

        # Créer les lignes reliants les points
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Créer une boucle fermée reliant les lignes     
        boucle = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

        # Créer une surface
        surface = gmsh.model.geo.addPlaneSurface([boucle])

        surface = gmsh.model.addPhysicalGroup(2, [surface])

        if returnSurfaces: return [surface]
        
        tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
        
        self.__Construction_MaillageGmsh(2, elemType, surfaces=[surface], isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    def RectangleAvecFissure(self, domain: Domain, crack: Line,
    elemType="TRI3", openCrack=False, isOrganised=False, folder=""):
        """Construit un rectangle avec une fissure dedans dans le plan 2D

        Args:
            domain (Domain): domaine renseigné qui doit etre contenue dans le plan (x, y)
            crack (Line): ligne qui carractérise la fissure
            elemType (str, optional): type d'element utilisé. Defaults to "TRI3" dans ["TRI3", "TRI6", "QUAD4", "QUAD8"]
            openCrack (bool, optional): la fissure est elle ouverte. Defaults to False.
            isOrganised (bool, optional): le maillage est il organisé. Defaults to False.
            folder (str, optional): fichier de sauvegarde dans lequel on mets le .msh . Defaults to "".

        Returns:
            Mesh: maillage construit
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
            self.__Construction_MaillageGmsh(2, elemType, surfaces=[surface], crack=crack, openBoundary=point, isOrganised=isOrganised)
        else:
            self.__Construction_MaillageGmsh(2, elemType, surfaces=[surface], isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    def PlaqueAvecCercle(self, domain: Domain, circle: Circle,
    elemType="TRI3", isOrganised=False, folder="", returnSurfaces=False):
        """Construit un rectangle un trou dedans

        Args:
            domain (Domain): domaine renseigné qui doit etre contenue dans le plan (x, y)
            circle (Line): cercle qui peut être ouvert ou fermé
            elemType (str, optional): type d'element utilisé. Defaults to "TRI3" dans ["TRI3", "TRI6", "QUAD4", "QUAD8"]
            openCrack (bool, optional): la fissure est elle ouverte. Defaults to False.
            isOrganised (bool, optional): le maillage est il organisé. Defaults to False.
            folder (str, optional): fichier de sauvegarde dans lequel on mets le .msh . Defaults to "".
            returnSurfaces (bool, optional): Renvoie les surfaces crées. Defaults to False.

        Returns:
            Mesh ou list: maillage construit ou retourne la surface si returnSurfaces=True
        """
        
            
        self.__initGmsh()
        self.__CheckType(2, elemType)

        tic = TicTac()

        # Domain
        pt1 = domain.pt1
        pt2 = domain.pt2
        if not returnSurfaces:
            assert pt1.z == 0 and pt2.z == 0

        # Circle
        center = circle.center
        diam = circle.diam
        rayon = diam/2
        if not returnSurfaces:
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
        loopDomain = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

        # Points cercle                
        p5 = gmsh.model.occ.addPoint(center.x, center.y, 0, circle.taille) #centre
        p6 = gmsh.model.occ.addPoint(center.x-rayon, center.y, 0, circle.taille)
        p7 = gmsh.model.occ.addPoint(center.x, center.y-rayon, 0, circle.taille)
        p8 = gmsh.model.occ.addPoint(center.x+rayon, center.y, 0, circle.taille)
        p9 = gmsh.model.occ.addPoint(center.x, center.y+rayon, 0, circle.taille)

        # Lignes cercle                
        l5 = gmsh.model.occ.addCircleArc(p6, p5, p7)
        l6 = gmsh.model.occ.addCircleArc(p7, p5, p8)
        l7 = gmsh.model.occ.addCircleArc(p8, p5, p9)
        l8 = gmsh.model.occ.addCircleArc(p9, p5, p6)
        lignecercle = gmsh.model.occ.addCurveLoop([l5,l6,l7,l8])

        # cercle = gmsh.model.occ.addCircle(center.x, center.y, center.z, diam/2)
        # lignecercle = gmsh.model.occ.addCurveLoop([cercle])
        # gmsh.option.setNumber("Mesh.MeshSizeMin", domain.taille)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", circle.taille)

        if circle.isCreux:
            # Create a surface avec le cyclindre creux
            surface = gmsh.model.occ.addPlaneSurface([loopDomain,lignecercle])

            # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
            gmsh.model.occ.synchronize()
            gmsh.model.occ.remove([(0,p5)], False)
            surfaces = [surface]
        else:
            # Cylindre plein
            surfaceCercle = gmsh.model.occ.addPlaneSurface([lignecercle])
            surface = gmsh.model.occ.addPlaneSurface([loopDomain, lignecercle])
            gmsh.model.occ.synchronize()
            gmsh.model.occ.remove([(0,p5)], False)

            surfaces = [surfaceCercle, surface]

            # gmsh.model.mesh.embed(1,[l5,l6],2, surface)

            # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
            # gmsh.model.occ.synchronize()
            # gmsh.model.occ.remove([(0,p6),(0,p7),(0,p8),(0,p9)], True)
        
        if returnSurfaces: return surfaces

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_MaillageGmsh(2, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def PlaqueAvecCercle3D(self, domain: Domain, circle: Circle, extrude=[0,0,1], nCouches=1,
    elemType="HEXA8", isOrganised=False, folder=""):

        self.__initGmsh()
        self.__CheckType(3, elemType)
        
        tic = TicTac()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.PlaqueAvecCercle(domain, circle, elemType="TRI3", isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(gmsh.model.occ, surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    # TODO Ici permettre la creation d'une simulation quelconques avec des points des lignes etc.   

    def __Construction_MaillageGmsh(self, dim: int, elemType: str, isOrganised=False,
    surfaces=[], crack=None, openBoundary=None, folder=""):

        tic = TicTac()
        if dim == 2:

            assert isinstance(surfaces, list)
            
            for surf in surfaces:

                # Impose que le maillage soit organisé                        
                if isOrganised:
                    # Ne fonctionne que pour une surface simple (sans trou ny fissure) et quand on construit le model avec geo et pas occ !
                    # groups = gmsh.model.getPhysicalGroups()
                    
                    # Quand geo
                    gmsh.model.geo.synchronize()
                    points = np.array(gmsh.model.getEntities(0))[:,1]
                    gmsh.model.geo.mesh.setTransfiniteSurface(surf, cornerTags=points) #Ici il faut impérativement donner les points du contour quand plus de 3 ou 4 coints
                    # gmsh.model.geo.mesh.setTransfiniteSurface(surface)

                # Synchronisation
                gmsh.model.occ.synchronize()
                gmsh.model.geo.synchronize()

                if elemType in ["QUAD4","QUAD8"]:
                    try:
                        gmsh.model.mesh.setRecombine(2, surf)
                    except Exception:
                        # Récupère la surface
                        entities = gmsh.model.getEntities()
                        surf = entities[-1][-1]
                        gmsh.model.mesh.setRecombine(2, surf)
                
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
        
        elif dim == 3:
            gmsh.model.occ.synchronize()
            gmsh.model.geo.synchronize()

            if elemType in ["HEXA8"]:

                # https://onelab.info/pipermail/gmsh/2010/005359.html               
                

                entities = gmsh.model.getEntities(2)
                surfaces = np.array(entities)[:,1]
                for surf in surfaces:
                    gmsh.model.mesh.setRecombine(2, surf)
                
                gmsh.model.mesh.setRecombine(3, 1)                

            gmsh.model.mesh.generate(3)
        
        # Ouvre l'interface de gmsh si necessaire
        if '-nopopup' not in sys.argv and self.__affichageGmsh:
            gmsh.fltk.run()
        
        tic.Tac("Mesh","Construction du maillage gmsh", self.__verbosity)

        if folder != "":
            # gmsh.write(Dossier.Join([folder, "model.geo"])) # Il semblerait que ça marche pas c'est pas grave
            gmsh.model.geo.synchronize()
            gmsh.model.occ.synchronize()
            gmsh.write(Dossier.Join([folder, "mesh.msh"]))
            tic.Tac("Mesh","Sauvegarde du .geo et du .msh", self.__verbosity)

    def __Recuperation_Maillage(self):
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
        # Le beug a été réglé car je norganisait pas bien les noeuds lors de la création 
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
            
            # A tester avec l, c = np.where(connect==changes[:,0])
            
            # Noeuds            
            nodes = np.unique(nodeTags)

            # Verifie que les numéros des noeuds max est bien atteignable dans coordo
            Nmax = nodes.max()
            assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"
            
            groupElem = GroupElem(gmshId, connect, elements, coordo, nodes)
            dict_groupElem[groupElem.dim] = groupElem
        
        tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construction2D(L=10, h=10, taille=3):

        interfaceGmsh = Interface_Gmsh(verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), taille=taille)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), taille=taille)
        circle = Circle(Point(x=L/2, y=h/2), L/3, taille=taille)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, taille=taille, isCreux=False)

        # Pour chaque type d'element 2D
        for t, elemType in enumerate(GroupElem.get_Types2D()):
            for isOrganised in [True, False]:
                    
                mesh = interfaceGmsh.Rectangle(domain=domain, elemType=elemType, isOrganised=isOrganised)
                mesh2 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=False)
                mesh3 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=True)
                mesh4 = interfaceGmsh.PlaqueAvecCercle(domain=domain, circle=circle, elemType=elemType, isOrganised=isOrganised)
                mesh5 = interfaceGmsh.PlaqueAvecCercle(domain=domain, circle=circleClose, elemType=elemType, isOrganised=isOrganised)

                for m in [mesh, mesh2, mesh3, mesh4, mesh5]:
                    list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def Construction3D(L=130, h=13, b=13, taille=130):
        # Pour chaque type d'element 3D

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
        circleCreux = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=True)
        circle = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=False)

        list_mesh3D = []
        for t, elemType in enumerate(GroupElem.get_Types3D()):
            for isOrganised in [True, False]:
                interfaceGmsh = Interface_Gmsh(verbosity=False)
                path = Dossier.GetPath()
                fichier = Dossier.Join([path,"models","part.stp"])
                if elemType == "TETRA4":
                    mesh = interfaceGmsh.Importation3D(fichier, elemType=elemType, tailleElement=taille)
                    list_mesh3D.append(mesh)
                
                mesh2 = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh2)

                mesh3 = interfaceGmsh.PlaqueAvecCercle3D(domain, circleCreux, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh3)

                mesh4 = interfaceGmsh.PlaqueAvecCercle3D(domain, circle, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh4)

        return list_mesh3D

   
                
        
        
        

