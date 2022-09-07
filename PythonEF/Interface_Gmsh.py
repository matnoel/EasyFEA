from inspect import stack
from typing import List
import gmsh
import sys
import numpy as np

import Dossier as Dossier
from Geom import *
from GroupElem import GroupElem
from Mesh import Mesh
from TicTac import Tic
import Affichage as Affichage
import matplotlib.pyplot as plt

class Interface_Gmsh:
    """Classe interface Gmsh"""

    def __init__(self, affichageGmsh=False, gmshVerbosity=False, verbosity=True):
        """Construction d'une interface qui peut intéragir avec gmsh

        Parameters
        ----------
        affichageGmsh : bool, optional
            affichage du maillage construit dans gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh peut ecrire dans le terminal, by default False
        verbosity : bool, optional
            la classe interfaceGmsh peut ecrire dans le terminal, by default True
        """
    
        self.__affichageGmsh = affichageGmsh
        """affichage du maillage sur gmsh"""
        self.__gmshVerbosity = gmshVerbosity
        """gmsh peut ecrire dans la console"""
        self.__verbosity = verbosity
        """modelGmsh peut ecrire dans la console"""

        if gmshVerbosity:
            Affichage.NouvelleSection("Maillage Gmsh")

    def __initGmsh(self):
        """Initialise gmsh"""
        gmsh.initialize()
        self.__factory = None
        if self.__gmshVerbosity == False:
            gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.model.add("model")
    
    def __CheckType(self, dim: int, elemType: str):
        """Verification si le type d'element est bien possible"""
        if dim == 2:
            assert elemType in GroupElem.get_Types2D()                        
        elif dim == 3:
            assert elemType in GroupElem.get_Types3D()

    def Importation3D(self, fichier="", tailleElement=0.0, folder=""):
        """Construis le maillage 3D depuis l'importation d'un fichier 3D et création du maillage (.stp ou .igs)

        Parameters
        ----------
        fichier : str, optional
            fichier (.stp, .igs) que gmsh va charger pour creer le maillage, by default ""
        tailleElement : float, optional
            taille de maille, by default 0.0
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """
        # Lorsqu'on importe une pièce on ne peut utiliser que du TETRA4
        elemType = "TETRA4"
        # Permettre d'autres maillage -> ça semble impossible il faut creer le maillage par gmsh pour maitriser le type d'element

        self.__initGmsh()

        assert tailleElement >= 0.0, "Doit être supérieur ou égale à 0"
        self.__CheckType(3, elemType)
        
        tic = Tic()

        factory = gmsh.model.occ # Ici ne fonctionne qu'avec occ !! ne pas changer
        self.__factory = factory

        # Importation du fichier
        factory.importShapes(fichier)

        gmsh.option.setNumber("Mesh.MeshSizeMin", tailleElement)
        gmsh.option.setNumber("Mesh.MeshSizeMax", tailleElement)

        tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def Poutre3D(self, domain: Domain, extrude=[0,0,1], nCouches=1, elemType="HEXA8", isOrganised=True, folder=""):
        """Construis le maillage 3D d'une poutre depuis une surface/domaine 2D que l'on extrude

        Parameters
        ----------
        domain : Domain
            domaine / surface que l'on va extruder
        extrude : list, optional
            directions et valeurs d'extrusion, by default [0,0,1]
        nCouches : int, optional
            nombre de couhes dans l'extrusion, by default 1
        elemType : str, optional
            type d'element utilisé, by default "HEXA8"
        isOrganised : bool, optional
            le maillage est organisé, by default True
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh 
            Maillage construit
        """

        self.__initGmsh()
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        if elemType == "TETRA4":    isOrganised=False #Il n'est pas possible d'oganiser le maillage quand on utilise des TETRA4
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Rectangle_2D(domain, elemType="TRI3", isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())
    
    
    def __Extrusion(self, surfaces: list, extrude=[0,0,1], elemType="HEXA8", isOrganised=True, nCouches=1):
        """Fonction qui effectue l'extrusion depuis plusieurs surfaces

        Parameters
        ----------
        surfaces : list[int]
            liste de surfaces
        extrude : list, optional
            directions et valeurs d'extrusion, by default [0,0,1]
        elemType : str, optional
            type d'element utilisé, by default "HEXA8"
        isOrganised : bool, optional
            le maillage est organisé, by default True
        nCouches : int, optional
            nombre de couches dans l'extrusion, by default 1
        """
        
        factory = self.__factory

        if factory == gmsh.model.geo:
            factory = cast(gmsh.model.geo, factory)
        elif factory == gmsh.model.occ:
            isOrganised = False
            factory = cast(gmsh.model.occ, factory)

        for surf in surfaces:

            if isOrganised:

                factory.synchronize()

                points = np.array(gmsh.model.getEntities(0))[:,1]
                if points.shape[0] <= 4:
                    factory.mesh.setTransfiniteSurface(surf, cornerTags=points)
                    # factory.mesh.setTransfiniteSurface(surf)

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


    def Rectangle_2D(self, domain: Domain, elemType="TRI3", isOrganised=False, folder="", returnSurfaces=False):
        """Construis le maillge d'un rectange 2D

        Parameters
        ----------
        domain : Domain
            domaine 2D qui doit être dans le plan (x,y)
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        isOrganised : bool, optional
            le maillage est organisé, by default False
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""
        returnSurfaces : bool, optional
            renvoie la surface crée, by default False

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh()                
        
        self.__CheckType(2, elemType)

        tic = Tic()    

        pt1 = domain.pt1
        pt2 = domain.pt2

        # assert pt1.z == 0 and pt2.z == 0

        tailleElement = domain.taille

        # factory=gmsh.model.occ # fonctionne mais ne permet pas d'organiser le maillage 
        factory=gmsh.model.geo

        self.__factory = factory

        # Créer les points
        p1 = factory.addPoint(pt1.x, pt1.y, 0, tailleElement)
        p2 = factory.addPoint(pt2.x, pt1.y, 0, tailleElement)
        p3 = factory.addPoint(pt2.x, pt2.y, 0, tailleElement)
        p4 = factory.addPoint(pt1.x, pt2.y, 0, tailleElement)

        # Créer les lignes reliants les points
        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)

        # Créer une boucle fermée reliant les lignes     
        boucle = factory.addCurveLoop([l1, l2, l3, l4])

        # Créer une surface
        surface = factory.addPlaneSurface([boucle])

        if isinstance(factory, gmsh.model.geo):
            surface = factory.addPhysicalGroup(2, [surface]) # obligatoire pour creer la surface organisée

        if returnSurfaces: return [surface]
        
        tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
        
        self.__Construction_MaillageGmsh(2, elemType, surfaces=[surface], isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    def RectangleAvecFissure(self, domain: Domain, crack: Line,
    elemType="TRI3", openCrack=False, isOrganised=False, folder=""):
        """Construis le maillage d'un rectangle avec une fissure dans le plan 2D

        Parameters
        ----------
        domain : Domain
            domaine 2D qui doit etre compris dans le plan (x,y)
        crack : Line
            ligne qui va construire la fissure
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        openCrack : bool, optional
            la fissure peut s'ouvrir, by default False
        isOrganised : bool, optional
            le maillage est organisé, by default False
        folder : str, optional
            dossier de sauvegarde du maillge, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh()                
        
        self.__CheckType(2, elemType)
        
        tic = Tic()

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

        factory = gmsh.model.occ # Ne fonctionne qu'avec occ
        self.__factory = factory

        # Create the points of the rectangle
        p1 = factory.addPoint(pt1.x, pt1.y, 0, domainSize)
        p2 = factory.addPoint(pt2.x, pt1.y, 0, domainSize)
        p3 = factory.addPoint(pt2.x, pt2.y, 0, domainSize)
        p4 = factory.addPoint(pt1.x, pt2.y, 0, domainSize)

        # Create the lines connecting the points for the surface
        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)                

        # loop for surface
        loop = factory.addCurveLoop([l1, l2, l3, l4])

        # creat surface
        surface = factory.addPlaneSurface([loop])

        # Create the crack points
        p5 = factory.addPoint(pf1.x, pf1.y, 0, crackSize)
        p6 = factory.addPoint(pf2.x, pf2.y, 0, crackSize)

        # Create the line for the crack
        crack = factory.addLine(p5, p6)

        listeOpen=[]
        if pf1.isOpen:
            o, m = factory.fragment([(0, p5), (1, crack)], [(2, surface)])
            listeOpen.append(p5)
        if pf2.isOpen:
            o, m = factory.fragment([(0, p6), (1, crack)], [(2, surface)])
            listeOpen.append(p6)
        factory.synchronize()
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

    def PlaqueAvecCercle2D(self, domain: Domain, circle: Circle,
    elemType="TRI3", isOrganised=False, folder="", returnSurfaces=False):
        """Construis le maillage 2D d'un rectangle un cercle (creux ou fermé)

        Parameters
        ----------
        domain : Domain
            surface qui doit etre contenu dans le plan (x,y)
        circle : Circle
            cercle creux ou plein
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        isOrganised : bool, optional
            le maillage est organisé, by default False
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""
        returnSurfaces : bool, optional
            renvoie la surface, by default False

        Returns
        -------
        Mesh 
            Maillage construit
        """
            
        self.__initGmsh()
        self.__CheckType(2, elemType)

        tic = Tic()

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

        # factory=gmsh.model.geo # fonctionne que si le cercle est remplie !
        factory=gmsh.model.occ
        self.__factory = factory

        # Create the points of the rectangle
        p1 = factory.addPoint(pt1.x, pt1.y, 0, domain.taille)
        p2 = factory.addPoint(pt2.x, pt1.y, 0, domain.taille)
        p3 = factory.addPoint(pt2.x, pt2.y, 0, domain.taille)
        p4 = factory.addPoint(pt1.x, pt2.y, 0, domain.taille)

        # Créer les lignes reliants les points pour la surface
        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)

        # Create a closed loop connecting the lines for the surface
        loopDomain = factory.addCurveLoop([l1, l2, l3, l4])

        # Points cercle                
        p5 = factory.addPoint(center.x, center.y, 0, circle.taille) #centre
        p6 = factory.addPoint(center.x-rayon, center.y, 0, circle.taille)
        p7 = factory.addPoint(center.x, center.y-rayon, 0, circle.taille)
        p8 = factory.addPoint(center.x+rayon, center.y, 0, circle.taille)
        p9 = factory.addPoint(center.x, center.y+rayon, 0, circle.taille)

        # Lignes cercle
        l5 = factory.addCircleArc(p6, p5, p7)
        l6 = factory.addCircleArc(p7, p5, p8)
        l7 = factory.addCircleArc(p8, p5, p9)
        l8 = factory.addCircleArc(p9, p5, p6)
        lignecercle = factory.addCurveLoop([l5,l6,l7,l8])

        # cercle = factory.addCircle(center.x, center.y, center.z, diam/2)
        # lignecercle = factory.addCurveLoop([cercle])
        # gmsh.option.setNumber("Mesh.MeshSizeMin", domain.taille)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", circle.taille)

        if circle.isCreux:
            # Create a surface avec le cyclindre creux
            surface = factory.addPlaneSurface([loopDomain,lignecercle])

            # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
            factory.synchronize()
            factory.remove([(0,p5)], False)
            surfaces = [surface]
        else:
            # Cylindre plein
            surfaceCercle = factory.addPlaneSurface([lignecercle])
            surface = factory.addPlaneSurface([loopDomain, lignecercle])
            factory.synchronize()
            factory.remove([(0,p5)], False)

            surfaces = [surfaceCercle, surface]

            # gmsh.model.mesh.embed(1,[l5,l6],2, surface)

            # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
            # factory.synchronize()
            # factory.remove([(0,p6),(0,p7),(0,p8),(0,p9)], True)
        
        if returnSurfaces: return surfaces

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_MaillageGmsh(2, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def PlaqueAvecCercle3D(self, domain: Domain, circle: Circle, extrude=[0,0,1], nCouches=1,
    elemType="HEXA8", isOrganised=False, folder=""):
        """Construis le maillage 3D d'un domaine avec un cylindre (creux ou fermé)

        Parameters
        ----------
        domain : Domain
            domaine / surface que l'on va extruder
        circle : Circle
            cercle creux ou plein
        extrude : list, optional
            directions et valeurs d'extrusion, by default [0,0,1]
        nCouches : int, optional
            nombre de couches dans l'extrusion, by default 1
        elemType : str, optional
            type d'element utilisé, by default "HEXA8"
        isOrganised : bool, optional
            le maillage est organisée, by default False
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh()
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.PlaqueAvecCercle2D(domain, circle, elemType="TRI3", isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    # Ici permettre la creation d'une simulation quelconques avec des points des lignes etc.
    # TODO permettre de mettre des trous ?

    def __Surfaces_From_Points(self, pointsList: List[Point], tailleElement: float, returnSurfaces: bool):
        """Construction d'une liste de surface en fonction d'une liste de points

        Parameters
        ----------
        pointsList : List[Point]
            liste de points
        tailleElement : float
            taille de maille
        returnSurfaces : bool
            renvoie la surface

        Returns
        -------
        List[int]
            liste de surfaces gmsh
        """
        
        factory = gmsh.model.occ # fonctionne toujours mais ne peut pas organiser le maillage        
        # factory = gmsh.model.geo # ne fonctionne pas toujours pour des surfaces compliquées

        # mettre en option ?
        
        self.__factory = factory

        # On creer tout les points
        points = []
        for point in pointsList:
            assert isinstance(point, Point)
            if not returnSurfaces: assert point.z == 0, "Pour une simulation 2D les points doivent être dans le plan (x, y)"

            points.append(factory.addPoint(point.x, point.y, point.z, tailleElement))

        # TODO Verifier que ça se croise pas ?

        # On creer les lignes qui relies les points
        connectLignes = np.repeat(points, 2).reshape(-1,2)
        indexForChange = np.arange(1, len(points)+1, 1)
        indexForChange[-1] = 0
        connectLignes[:,1] = connectLignes[indexForChange,1]

        lignes = []
        for pt1, pt2 in connectLignes:
            lignes.append(factory.addLine(pt1, pt2))

        # Create a closed loop connecting the lines for the surface        
        loopSurface = factory.addCurveLoop(lignes)
        
        surface = factory.addPlaneSurface([loopSurface])

        if isinstance(factory, gmsh.model.geo):
            surface = factory.addPhysicalGroup(2, [surface]) # obligatoire pour creer la surface organisée

        surfaces = [surface]

        return surfaces

    def Mesh_From_Points_2D(self, pointsList: List[Point],
    elemType="TRI3", tailleElement=0.0, isOrganised=False, folder="", returnSurfaces=False):
        """Construis le maillage 2D en créant une surface depuis une liste de points

        Parameters
        ----------
        pointsList : List[Point]
            liste de points
        elemType : str, optional
            type d'element, by default "TRI3" ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        tailleElement : float, optional
            taille d'element pour le maillage, by default 0.0
        isOrganised : bool, optional
            le maillage est organisé, by default False
        folder : str, optional
            fichier de sauvegarde du maillage, by default ""
        returnSurfaces : bool, optional
            renvoie la surface, by default False

        Returns
        -------
        Mesh
            Maillage 2D
        """

        self.__initGmsh()
        self.__CheckType(2, elemType)

        tic = Tic()

        surfaces = self.__Surfaces_From_Points(pointsList, tailleElement, returnSurfaces)
        
        if returnSurfaces: return surfaces

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_MaillageGmsh(2, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    

    def Mesh_From_Points_3D(self, pointsList: List[Point], extrude=[0,0,1], nCouches=1, 
    elemType="TETRA4", tailleElement=0.0, isOrganised=False, folder=""):
        """Construction d'un maillage 3D depuis une liste de points

        Parameters
        ----------
        pointsList : List[Point]
            liste de points
        extrude : list, optional
            extrusion, by default [0,0,1]
        nCouches : int, optional
            nombre de couches dans l'extrusion, by default 1
        elemType : str, optional
            type d'element, by default "TETRA4" ["TETRA4", "HEXA8", "PRISM6"]
        tailleElement : float, optional
            taille d'element pour le maillage, by default 0.0
        isOrganised : bool, optional
            le maillage est orgnanisé, by default False
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh
            Maillage 3D
        """

        self.__initGmsh()
        self.__CheckType(3, elemType)
        
        tic = Tic()

        if elemType == "TETRA4": isOrganised=False
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_From_Points_2D(pointsList, elemType="TRI3", tailleElement=tailleElement,
        isOrganised=isOrganised, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_MaillageGmsh(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())


    def __Construction_MaillageGmsh(self, dim: int, elemType: str, surfaces=[], 
    isOrganised=False, crack=None, openBoundary=None, folder=""):
        """Construction du maillage gmsh depuis la geométrie qui a été construit ou importée

        Parameters
        ----------
        dim : int
            dimension du maillage
        elemType : str
            type d'element
        surfaces : List[int], optional
            liste de surfaces que l'on va mailler, by default []
        isOrganised : bool, optional
            le maillage est organisé, by default False
        crack : int, optional
            fissure renseigné, by default None
        openBoundary : int, optional
            domaine qui peut s'ouvrir, by default None
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""
        """

        factory = self.__factory

        if factory == gmsh.model.occ:
            isOrganised = False
            factory = cast(gmsh.model.occ, factory)
        elif factory == gmsh.model.geo:
            factory = cast(gmsh.model.geo, factory)
        else:
            raise "factory inconnue"

        tic = Tic()
        if dim == 2:

            assert isinstance(surfaces, list)
            
            for surf in surfaces:

                # Impose que le maillage soit organisé                        
                if isOrganised:
                    # Ne fonctionne que pour une surface simple (sans trou ny fissure) et quand on construit le model avec geo et pas occ !
                    # Il n'est pas possible de creer une surface setTransfiniteSurface avec occ
                    # Dans le cas ou il faut forcemenent passé par occ, il n'est donc pas possible de creer un maillage organisé
                    
                    # Quand geo
                    gmsh.model.geo.synchronize()
                    points = np.array(gmsh.model.getEntities(0))[:,1]
                    if points.shape[0] <= 4:
                        #Ici il faut impérativement donner les points du contour quand plus de 3 ou 4 coints
                        gmsh.model.geo.mesh.setTransfiniteSurface(surf, cornerTags=points) 
                        # gmsh.model.geo.mesh.setTransfiniteSurface(surf)

                # Synchronisation
                self.__factory.synchronize()

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
            self.__factory.synchronize()

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
        """Récupération du maillage construit

        Returns
        -------
        Mesh
            Maillage construit
        """

        # Ancienne méthode qui beugait
        # Le beug a été réglé car je norganisait pas bien les noeuds lors de la création 
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1926
        
        tic = Tic()

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

        # Construit les groupes d'elements
        testDimension = False
        dimAjoute = []
        dim = 0
        for gmshId in elementTypes:
                                        
            # Récupère le numéros des elements et la matrice de connection
            elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
            elementTags = np.array(elementTags-1, dtype=int)
            nodeTags = np.array(nodeTags-1, dtype=int)

            # Elements
            Ne = elementTags.shape[0] #nombre d'élements
            elementsID = elementTags
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
            
            groupElem = GroupElem(gmshId, connect, elementsID, coordo, nodes)
            if groupElem.dim > dim: dim = groupElem.dim
            dict_groupElem[groupElem.elemType] = groupElem
            
            if groupElem.dim in dimAjoute:
                testDimension = True
            dimAjoute.append(groupElem.dim)

        if dimAjoute.count(dim) > 1:
            # TODO faire en sorte de pouvoir le faire ?
            # Peut etre compliqué surtout dans la création des matrices elementaire et assemblage
            # Pas impossible mais pas trivial
            assert not testDimension, f"Impossible car {dimAjoute.count(dim)} type d'element {dim}D"
        
        tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construction2D(L=10, h=10, taille=3):
        """Construction des maillage possibles en 2D"""

        interfaceGmsh = Interface_Gmsh(verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), taille=taille)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), taille=taille)
        circle = Circle(Point(x=L/2, y=h/2), L/3, taille=taille, isCreux=True)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, taille=taille, isCreux=False)

        aireDomain = L*h
        aireCircle = np.pi * (circleClose.diam/2)**2

        # Pour chaque type d'element 2D
        for t, elemType in enumerate(GroupElem.get_Types2D()):
            for isOrganised in [True, False]:
                    
                mesh = interfaceGmsh.Rectangle_2D(domain=domain, elemType=elemType, isOrganised=isOrganised)
                assert np.isclose(mesh.aire, aireDomain,1e-4), "Surface incorrect"
                mesh2 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=False)
                assert np.isclose(mesh2.aire, aireDomain,1e-4), "Surface incorrect"
                mesh3 = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType, isOrganised=isOrganised, openCrack=True)
                assert np.isclose(mesh3.aire, aireDomain,1e-4), "Surface incorrect"
                mesh4 = interfaceGmsh.PlaqueAvecCercle2D(domain=domain, circle=circle, elemType=elemType, isOrganised=isOrganised)
                # # assert mesh4.aire - (aireDomain-aireCircle) == 0
                # Ici on ne verifie pas car il ya trop peu delement pour bien representer le perçage
                mesh5 = interfaceGmsh.PlaqueAvecCercle2D(domain=domain, circle=circleClose, elemType=elemType, isOrganised=isOrganised)
                assert np.isclose(mesh5.aire, aireDomain,1e-4), "Surface incorrect"

                for m in [mesh, mesh2, mesh3, mesh4, mesh5]:
                    list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def Construction3D(L=130, h=13, b=13, taille=7.5):
        """Construction des maillage possibles en 3D"""
        # Pour chaque type d'element 3D

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
        circleCreux = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=True)
        circle = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=False)

        volume = L*h*b

        list_mesh3D = []
        for t, elemType in enumerate(GroupElem.get_Types3D()):
            for isOrganised in [True, False]:
                interfaceGmsh = Interface_Gmsh(verbosity=False, affichageGmsh=False)
                # path = Dossier.GetPath(__file__)
                # fichier = Dossier.Join([path,"3Dmodels","part.stp"])
                # if elemType == "TETRA4":
                #     mesh = interfaceGmsh.Importation3D(fichier, elemType=elemType, tailleElement=taille)
                #     list_mesh3D.append(mesh)
                
                mesh2 = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh2)
                assert np.isclose(mesh2.volume, volume,1e-4), "Volume incorrect"

                mesh3 = interfaceGmsh.PlaqueAvecCercle3D(domain, circleCreux, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh3)

                mesh4 = interfaceGmsh.PlaqueAvecCercle3D(domain, circle, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh4)
                assert np.isclose(mesh4.volume, volume,1e-4), "Volume incorrect"

        return list_mesh3D

   
                
        
        
        

