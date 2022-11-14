from typing import List, cast
import gmsh
import sys
import numpy as np

import Dossier as Dossier
from Geom import *
from GroupElem import GroupElem, ElemType, MatriceType, GroupElem_Factory
from Mesh import Mesh
from TicTac import Tic
import Affichage as Affichage
from Materials import Poutre_Elas_Isot

class Interface_Gmsh:
    """Classe interface Gmsh"""    

    def __init__(self, affichageGmsh=False, gmshVerbosity=False, verbosity=False):
        """Construction d'une interface qui peut intéragir avec gmsh

        Parameters
        ----------
        affichageGmsh : bool, optional
            affichage du maillage construit dans gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh peut ecrire dans le terminal, by default False
        verbosity : bool, optional
            la classe interfaceGmsh peut ecrire le résumé de la construction dans le terminale, by default False
        """
    
        self.__affichageGmsh = affichageGmsh
        """affichage du maillage sur gmsh"""
        self.__gmshVerbosity = gmshVerbosity
        """gmsh peut ecrire dans la console"""
        self.__verbosity = verbosity
        """modelGmsh peut ecrire dans la console"""

        if gmshVerbosity:
            Affichage.NouvelleSection("Maillage Gmsh")

    def __CheckType(self, dim: int, elemType: str):
        """Verification si le type d'element est bien possible"""
        if dim == 1:
            assert elemType in GroupElem.get_Types1D()
        if dim == 2:
            assert elemType in GroupElem.get_Types2D()
        elif dim == 3:
            assert elemType in GroupElem.get_Types3D()
    
    def __initGmsh(self, factory: str):
        """Initialise gmsh"""
        gmsh.initialize()
        if self.__gmshVerbosity == False:
            gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.model.add("model")
        if factory == 'occ':
            self.__factory = gmsh.model.occ
        elif factory == 'geo':
            self.__factory = gmsh.model.geo
        else:
            raise "Factory inconnue"

    def __Loop_From_Points(self, points: List[Point], taille: float) -> tuple[int, int]:
        """Création d'une boucle associée à la liste de points\n
        return loop
        """
        
        factory = self.__factory

        # On creer tout les points
        listPoint = []
        for point in points:
            assert isinstance(point, Point)
            pt = factory.addPoint(point.x, point.y, point.z, taille)
            # self.__Add_PhysicalPoint(pt)
            listPoint.append(pt)

        # On creer les lignes qui relies les points
        connectLignes = np.repeat(listPoint, 2).reshape(-1,2)
        indexForChange = np.arange(1, len(listPoint)+1, 1)
        indexForChange[-1] = 0
        connectLignes[:,1] = connectLignes[indexForChange,1]

        lignes = []
        for pt1, pt2 in connectLignes:
            lignes.append(factory.addLine(pt1, pt2))
            # self.__Add_PhysicalLine(lignes[-1])

        # Create a closed loop connecting the lines for the surface        
        loop = factory.addCurveLoop(lignes)

        return loop

    def __Loop_From_Circle(self, circle: Circle) -> tuple[int, int]:
        """Création d'une boucle associée à un cercle\n
        return loop
        """

        factory = self.__factory

        center = circle.center
        rayon = circle.diam/2

        # Points cercle                
        p0 = factory.addPoint(center.x, center.y, 0, circle.taille) #centre
        p1 = factory.addPoint(center.x-rayon, center.y, 0, circle.taille)
        p2 = factory.addPoint(center.x, center.y-rayon, 0, circle.taille)
        p3 = factory.addPoint(center.x+rayon, center.y, 0, circle.taille)
        p4 = factory.addPoint(center.x, center.y+rayon, 0, circle.taille)
        # [self.__Add_PhysicalPoint(pt) for pt in [p1, p2, p3, p4]]

        # Lignes cercle
        l1 = factory.addCircleArc(p1, p0, p2)
        l2 = factory.addCircleArc(p2, p0, p3)
        l3 = factory.addCircleArc(p3, p0, p4)
        l4 = factory.addCircleArc(p4, p0, p1)
        # Ajoute les segments dans les groupes physiques
        # [self.__Add_PhysicalLine(li) for li in [l1, l2, l3, l4]]

        # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
        factory.remove([(0,p0)], False)
            
        
        loop = factory.addCurveLoop([l1,l2,l3,l4])

        return loop

    def __Loop_From_Domain(self, domain: Domain) -> tuple[int, int]:
        """Création d'une boucle associée à un domaine\n
        return loop
        """
        pt1 = domain.pt1
        pt2 = domain.pt2

        p1 = Point(x=pt1.x, y=pt1.y, z=0)
        p2 = Point(x=pt2.x, y=pt1.y, z=0)
        p3 = Point(x=pt2.x, y=pt2.y, z=0)
        p4 = Point(x=pt1.x, y=pt2.y, z=0)

        loop = self.__Loop_From_Points([p1, p2, p3, p4], domain.taille)
        
        return loop

    def __Surface_From_Loops(self, loops: List[int]) -> tuple[int, int]:
        """Création d'une surface associée à une boucle\n
        return surface
        """
        factory = self.__factory

        surface = factory.addPlaneSurface(loops)

        return surface

    def __Crack_And_Points_From_Line(self, line: Line, surface: int) -> tuple[int, int, int]:
        """Création d'une fissure associée à une ligne\n
        crack, physicalCrack, physicalPoints
        """
        
        factory = self.__factory

        if isinstance(factory, gmsh.model.occ):
            return

        pt1 = line.pt1
        pt2 = line.pt2
        assert pt1.z == 0 and pt2.z == 0
        
        taille = line.taille
        
        # Create the crack points
        p1 = factory.addPoint(pt1.x, pt1.y, 0, taille)
        p2 = factory.addPoint(pt2.x, pt2.y, 0, taille)

        # Create the line for the crack
        crack = factory.addLine(p1, p2)
        listeOpen=[]
        if pt1.isOpen:
            o1, m1 = factory.fragment([(0, p1), (1, crack)], [(2, surface)])
            listeOpen.append(p1)
        if pt2.isOpen:
            o2, m2 = factory.fragment([(0, p2), (1, crack)], [(2, surface)])
            listeOpen.append(p2)
        factory.synchronize()
        
        # Adds the line to the surface
        gmsh.model.mesh.embed(1, [crack], 2, surface)
        if len(listeOpen)==0:
            physicalPoints = None
        else:
            physicalPoints = gmsh.model.addPhysicalGroup(0, listeOpen, name=f"P{p2}")

        return crack, physicalPoints
    
    def __Add_PhysicalPoint(self, point: int) -> int:
        """Ajoute le point dans le physical group"""
        # self.__factory.synchronize()
        pgPoint = gmsh.model.addPhysicalGroup(0, [point], name=f"P{point}")
        return pgPoint

    def __Add_PhysicalLine(self, ligne: int) -> int:
        """Ajoute la ligne dans les physical group"""
        # self.__factory.synchronize()
        pgLine = gmsh.model.addPhysicalGroup(1, [ligne], name=f"L{ligne}")
        return pgLine

    def __Add_PhysicalSurface(self, surface: int) -> int:
        """Ajoute la surface fermée ou ouverte dans les physical group"""
        # self.__factory.synchronize()
        pgSurf = gmsh.model.addPhysicalGroup(2, [surface], name=f"S{surface}")
        return pgSurf
    
    def __Add_PhysicalVolume(self, volume: int) -> int:
        """Ajoute le volume fermée ou ouverte dans les physical group"""
        # self.__factory.synchronize()
        pgVol = gmsh.model.addPhysicalGroup(3, [volume], name=f"V{volume}")
        return pgVol

    def __Add_PhyscialGroup(self, dim: int, tag: int):
        if dim == 0:
            self.__Add_PhysicalPoint(tag)
        elif dim == 1:
            self.__Add_PhysicalLine(tag)
        elif dim == 2:
            self.__Add_PhysicalSurface(tag)
        elif dim == 3:
            self.__Add_PhysicalVolume(tag)

    def __Set_PhysicalGroups(self, buildPoint=True, buildLine=True, buildSurface=True, buildVolume=True):
        """Création des groupes physiques en fonction des entités du model"""
        self.__factory.synchronize()
        entities = np.array(gmsh.model.getEntities())

        listDim = []
        if buildPoint:
            listDim.append(0)
        if buildLine:
            listDim.append(1)
        if buildSurface:
            listDim.append(2)
        if buildVolume:
            listDim.append(3)

        dims = entities[:,0]

        indexes = []
        [indexes.extend(np.where(dims == d)[0]) for d in listDim]

        entities = entities[indexes]

        [self.__Add_PhyscialGroup(dim, tag) for dim, tag in zip(entities[:,0], entities[:,1])]

    def __Extrusion(self, surfaces: list, extrude=[0,0,1], elemType=ElemType.HEXA8, isOrganised=True, nCouches=1):
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

        if factory == gmsh.model.occ:
            isOrganised = False

        for surf in surfaces:

            if isOrganised:

                factory = cast(gmsh.model.geo, factory)

                factory.synchronize()

                points = np.array(gmsh.model.getEntities(0))[:,1]
                if points.shape[0] <= 4:
                    factory.mesh.setTransfiniteSurface(surf, cornerTags=points)
                    # factory.mesh.setTransfiniteSurface(surf)

            if elemType in [ElemType.HEXA8, ElemType.PRISM6]:
                # ICI si je veux faire des PRISM6 J'ai juste à l'aisser l'option activée
                numElements = [nCouches]
                combine = True
            elif elemType == ElemType.TETRA4:
                numElements = []
                combine = False
            
            # Creer les nouveaux elements pour l'extrusion
            # nCouches = np.max([np.ceil(np.abs(extrude[2] - domain.taille)), 1])
            extru = factory.extrude([(2, surf)], extrude[0], extrude[1], extrude[2], recombine=combine, numElements=numElements)    

    # TODO générer plusieurs maillage en désactivant initGmsh et en utilisant plusieurs fonctions ?
    # mettre en place une liste de surfaces ?

    def Mesh_Importation3D(self, fichier="", tailleElement=0.0, folder=""):
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
        elemType = ElemType.TETRA4
        # Permettre d'autres maillage -> ça semble impossible il faut creer le maillage par gmsh pour maitriser le type d'element

        self.__initGmsh('occ') # Ici ne fonctionne qu'avec occ !! ne pas changer

        assert tailleElement >= 0.0, "Doit être supérieur ou égale à 0"
        self.__CheckType(3, elemType)
        
        tic = Tic()

        factory = self.__factory

        if '.stp' in fichier or '.igs' in fichier:
            factory.importShapes(fichier)
        else:
            print("Doit être un fichier .stp")

        self.__Set_PhysicalGroups(buildPoint=False, buildLine=True, buildSurface=True, buildVolume=False)

        gmsh.option.setNumber("Mesh.MeshSizeMin", tailleElement)
        gmsh.option.setNumber("Mesh.MeshSizeMax", tailleElement)

        tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

        self.__Construction_Maillage(3, elemType, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def Mesh_Poutre3D(self, domain: Domain, extrude=[0,0,1], nCouches=1, elemType=ElemType.HEXA8, isOrganised=True, folder=""):
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

        self.__initGmsh('geo')
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        if elemType == ElemType.TETRA4:    isOrganised=False #Il n'est pas possible d'oganiser le maillage quand on utilise des TETRA4
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_Rectangle_2D(domain, elemType=ElemType.TRI3, isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())    

    def Mesh_Rectangle_2D(self, domain: Domain, elemType=ElemType.TRI3, isOrganised=False, folder="", returnSurfaces=False):
        """Maillage d'un rectange 2D

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

        self.__initGmsh('geo')                
        
        self.__CheckType(2, elemType)

        tic = Tic()

        # Création de la boucle
        loop = self.__Loop_From_Domain(domain)

        # Création de la surface
        surface = self.__Surface_From_Loops([loop])

        # if isinstance(factory, gmsh.model.geo):
        #     surface = factory.addPhysicalGroup(2, [surface]) # obligatoire pour creer la surface organisée

        if returnSurfaces: return [surface]

        self.__Set_PhysicalGroups()
        
        tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
        
        self.__Construction_Maillage(2, elemType, surfaces=[surface], isOrganised=isOrganised, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())    

    def Mesh_Rectangle2DAvecFissure(self, domain: Domain, line: Line, elemType=ElemType.TRI3,folder=""):
        """Maillage d'un rectangle avec une fissure

        Parameters
        ----------
        domain : Domain
            domaine 2D qui doit etre compris dans le plan (x,y)
        crack : Line
            ligne qui va construire la fissure
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        folder : str, optional
            dossier de sauvegarde du maillge, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh('occ')                
        
        self.__CheckType(2, elemType)
        
        tic = Tic()

        # Création de la surface
        loop = self.__Loop_From_Domain(domain)

        # Création de la surface
        surface = self.__Surface_From_Loops([loop])

        # Création de la fissure
        crack, physicalPoints = self.__Crack_And_Points_From_Line(line, surface)

        # Regénération des groupes physiques
        self.__Set_PhysicalGroups(buildSurface=False)

        physicalSurface = gmsh.model.addPhysicalGroup(2, [surface])
        physicalCrack = gmsh.model.addPhysicalGroup(1, [crack])
        
        tic.Tac("Mesh","Construction rectangle fissuré", self.__verbosity)
        
        if line.isOpen:
            self.__Construction_Maillage(2, elemType, surfaces=[physicalSurface], cracks=[physicalCrack], openBoundarys=[physicalPoints], isOrganised=False)
        else:
            self.__Construction_Maillage(2, elemType, surfaces=[physicalSurface], isOrganised=False, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    def Mesh_PlaqueAvecCercle2D(self, domain: Domain, circle: Circle, elemType=ElemType.TRI3, domain2=None, folder="", returnSurfaces=False):
        """Construis le maillage 2D d'un rectangle un cercle (creux ou fermé)

        Parameters
        ----------
        domain : Domain
            surface qui doit etre contenu dans le plan (x,y)
        circle : Circle
            cercle creux ou plein
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        domain2 : str, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""
        returnSurfaces : bool, optional
            renvoie la surface, by default False

        Returns
        -------
        Mesh 
            Maillage construit
        """
        
        # factory=gmsh.model.geo # fonctionne que si le cercle est remplie !    
        self.__initGmsh('occ') 
        self.__CheckType(2, elemType)

        tic = Tic()

        factory = self.__factory

        loopDomain = self.__Loop_From_Domain(domain)

        loopCercle = self.__Loop_From_Circle(circle)
        

        if isinstance(domain2, Domain):
            # Exemple extrait de t10.py dans les tutos gmsh
            pt21 = domain2.pt1
            pt22 = domain2.pt2
            taille2 = domain2.taille

            # We could also use a `Box' field to impose a step change in element sizes
            # inside a box
            fieldDomain2 = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(fieldDomain2, "VIn", taille2)
            gmsh.model.mesh.field.setNumber(fieldDomain2, "VOut", domain.taille)
            gmsh.model.mesh.field.setNumber(fieldDomain2, "XMin", np.min([pt21.x, pt22.x]))
            gmsh.model.mesh.field.setNumber(fieldDomain2, "XMax", np.max([pt21.x, pt22.x]))
            gmsh.model.mesh.field.setNumber(fieldDomain2, "YMin", np.min([pt21.y, pt22.y]))
            gmsh.model.mesh.field.setNumber(fieldDomain2, "YMax", np.max([pt21.y, pt22.y]))
            gmsh.model.mesh.field.setNumber(fieldDomain2, "Thickness", np.abs(pt21.z - pt22.z))

            # Let's use the minimum of all the fields as the background mesh field:
            minField = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [fieldDomain2])

            gmsh.model.mesh.field.setAsBackgroundMesh(minField)

        # cercle = factory.addCircle(center.x, center.y, center.z, diam/2)
        # lignecercle = factory.addCurveLoop([cercle])
        # gmsh.option.setNumber("Mesh.MeshSizeMin", domain.taille)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", circle.taille)

        # Création d'une surface de la surface sans le cercle
        surfaceDomain = self.__Surface_From_Loops([loopDomain, loopCercle])
        factory.synchronize()

        if circle.isCreux:
            # Create a surface avec le cyclindre creux
            surfaces = [surfaceDomain]
        else:
            # Cylindre plein
            # Création d'une surface pour le cercle plein
            surfaceCercle = self.__Surface_From_Loops([loopCercle])
            p0 = factory.addPoint(circle.center.x, circle.center.y, 0, circle.taille)
            factory.synchronize()
            gmsh.model.mesh.embed(0, [p0], 2, surfaceCercle)
            factory.synchronize()
            surfaces = [surfaceCercle, surfaceDomain]
        
        if returnSurfaces: return surfaces

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(2, elemType, surfaces=surfaces, isOrganised=False, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def Mesh_PlaqueAvecCercle3D(self, domain: Domain, circle: Circle, extrude=[0,0,1], nCouches=1,  elemType=ElemType.HEXA8, folder=""):
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
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh("occ")
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_PlaqueAvecCercle2D(domain, circle, elemType=ElemType.TRI3, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=False, nCouches=nCouches)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=False, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    def Mesh_From_Lines_1D(self, listPoutres: List[Poutre_Elas_Isot], elemType=ElemType.SEG2 ,folder=""):
        """Construction d'un maillage de segment

        Parameters
        ----------
        listPoutre : List[Poutre]
            liste de Poutres
        elemType : str, optional
            type d'element, by default "SEG2" ["SEG2", "SEG3"]
        folder : str, optional
            fichier de sauvegarde du maillage, by default ""

        Returns
        -------
        Mesh
            Maillage 2D
        """

        self.__initGmsh('occ')
        self.__CheckType(1, elemType)

        tic = Tic()
        
        factory = self.__factory

        listPoints = [] 
        listeLines = []

        for poutre in listPoutres:

            line = poutre.line
            
            pt1 = line.pt1; x1 = pt1.x; y1 = pt1.y; z1 = pt1.z
            pt2 = line.pt2; x2 = pt2.x; y2 = pt2.y; z2 = pt2.z

            p1 = factory.addPoint(x1, y1, z1, line.taille)
            p2 = factory.addPoint(x2, y2, z2, line.taille)
            listPoints.append(p1)
            listPoints.append(p2)

            ligne = factory.addLine(p1, p2)
            # self.__Add_PhysicalLine(ligne)
            listeLines.append(ligne)

            factory.synchronize()
            # physicalGroup = gmsh.model.addPhysicalGroup(1, [ligne], name=f"{poutre.name}")

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(1, elemType, surfaces=[], folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def __Get_LoopsAndFilledLoops(self, geomObjectsInDomain: list) -> tuple[list, list]:
        """Création des boucles de la liste d'objets renseignés

        Parameters
        ----------
        geomObjectsInDomain : list
            Liste d'objet géométrique contenu dans le domaine

        Returns
        -------
        tuple[list, list]
            toutes les boucles créés, suivit des boucles pleines (non creuses)
        """
        loops = []
        filledLoops = []
        for objetGeom in geomObjectsInDomain:
            if isinstance(objetGeom, Circle):
                loop = self.__Loop_From_Circle(objetGeom)
            elif isinstance(objetGeom, Domain):                
                loop = self.__Loop_From_Domain(objetGeom)
            loops.append(loop)

            if not objetGeom.isCreux:
                filledLoops.append(loop)

        return loops, filledLoops

    def Mesh_From_Points_2D(self, points: List[Point], elemType=ElemType.TRI3, geomObjectsInDomain=[], tailleElement=0.0, folder="", returnSurfaces=False):
        """Construis le maillage 2D en créant une surface depuis une liste de points

        Parameters
        ----------
        points : List[Point]
            liste de points
        elemType : str, optional
            type d'element, by default "TRI3" ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        geomObjectsInDomain : List[Domain, Circle], optional
            liste d'objet à l'intérieur du domaine Creux ou non 
        tailleElement : float, optional
            taille d'element pour le maillage, by default 0.0
        folder : str, optional
            fichier de sauvegarde du maillage, by default ""
        returnSurfaces : bool, optional
            renvoie la surface, by default False

        Returns
        -------
        Mesh
            Maillage 2D
        """

        self.__initGmsh('occ')
        self.__CheckType(2, elemType)

        tic = Tic()

        factory = self.__factory

        # Création de la surface de contour
        loopSurface = self.__Loop_From_Points(points, tailleElement)

        # Création de toutes les boucles associés aux objets à l'intérieur du domaine
        loops, filledLoops = self.__Get_LoopsAndFilledLoops(geomObjectsInDomain)

        # Pour chaque objetGeom plein, il est nécessaire de créer une surface
        surfacesPleines = [factory.addPlaneSurface([loop]) for loop in filledLoops]

        # surface du domaine
        listeLoop = [loopSurface]
        listeLoop.extend(loops)

        surfaceDomaine = self.__Surface_From_Loops(listeLoop)

        # Rajoute la surface du domaine en dernier
        surfacesPleines.append(surfaceDomaine)
        
        # Création des surfaces creuses
        if returnSurfaces: return surfacesPleines

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(2, elemType, surfaces=surfacesPleines, isOrganised=False, folder=folder)

        return cast(Mesh, self.__Recuperation_Maillage())

    def Mesh_From_Points_3D(self, pointsList: List[Point], extrude=[0,0,1], nCouches=1, elemType=ElemType.TETRA4, interieursList=[], tailleElement=0.0, folder=""):
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
        folder : str, optional
            dossier de sauvegarde du maillage .msh, by default ""

        Returns
        -------
        Mesh
            Maillage 3D
        """

        self.__initGmsh('occ')
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_From_Points_2D(pointsList, elemType=ElemType.TRI3,geomObjectsInDomain=interieursList, tailleElement=tailleElement, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=False, nCouches=nCouches)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=False, folder=folder)
        
        return cast(Mesh, self.__Recuperation_Maillage())

    @staticmethod
    def __Set_order(elemType: str):
        if elemType in ["TRI3","QUAD4"]:
            gmsh.model.mesh.set_order(1)
        elif elemType in ["SEG3", "TRI6", "QUAD8"]:
            if elemType in ["QUAD8"]:
                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
            gmsh.model.mesh.set_order(2)
        elif elemType in ["SEG4", "TRI10"]:
            gmsh.model.mesh.set_order(3)
        elif elemType in ["SEG5", "TRI15"]:
            gmsh.model.mesh.set_order(4)


    def __Construction_Maillage(self, dim: int, elemType: str, surfaces=[], isOrganised=False, cracks=[], openBoundarys=[], folder=""):
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
        if dim == 1:
            self.__factory.synchronize()
            gmsh.model.mesh.generate(1)
            Interface_Gmsh.__Set_order(elemType)
        elif dim == 2:

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

                if elemType in [ElemType.QUAD4,ElemType.QUAD8]:
                    try:
                        gmsh.model.mesh.setRecombine(2, surf)
                    except Exception:
                        # Récupère la surface
                        entities = gmsh.model.getEntities()
                        surf = entities[-1][-1]
                        gmsh.model.mesh.setRecombine(2, surf)
                
                # Génère le maillage
                gmsh.model.mesh.generate(2)
                
                Interface_Gmsh.__Set_order(elemType)
        
        elif dim == 3:
            self.__factory.synchronize()

            if elemType in [ElemType.HEXA8]:

                # https://onelab.info/pipermail/gmsh/2010/005359.html

                entities = gmsh.model.getEntities(2)
                surfaces = np.array(entities)[:,1]
                for surf in surfaces:
                    gmsh.model.mesh.setRecombine(2, surf)
                
                gmsh.model.mesh.setRecombine(3, 1)

            gmsh.model.mesh.generate(3)

        if len(cracks) > 0:
            oldPG = gmsh.model.getPhysicalGroups()
            for crack, openBoundary in zip(cracks, openBoundarys):
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
            newPG = gmsh.model.getPhysicalGroups()
        
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
        
        # Construit les groupes physiques
        physicalGroups = gmsh.model.getPhysicalGroups()
        pgArray = np.array(physicalGroups)
        # A optimiser
        physicalGroupsPoint = []; namePoint = []
        physicalGroupsLine = []; nameLine = []
        physicalGroupsSurf = []; nameSurf = []
        physicalGroupsVol = []; nameVol = []

        for dim in range(pgArray[:,0].max()+1):
            indexDim = np.where(pgArray[:,0] == dim)[0]
            listTupleDim = tuple(map(tuple, pgArray[indexDim]))
            nbEnti = indexDim.size
            if dim == 0:
                namePoint.extend([f"P{n+1}" for n in range(nbEnti)])
                physicalGroupsPoint.extend(listTupleDim)
            elif dim == 1:
                nameLine.extend([f"L{n+1}" for n in range(nbEnti)])
                physicalGroupsLine.extend(listTupleDim)
            elif dim == 2:
                nameSurf.extend([f"S{n+1}" for n in range(nbEnti)])
                physicalGroupsSurf.extend(listTupleDim)
            elif dim == 3:
                nameVol.extend([f"V{n+1}" for n in range(nbEnti)])
                physicalGroupsVol.extend(listTupleDim)

        for gmshId in elementTypes:
                                        
            # Récupère le numéros des elements et la matrice de connection
            elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
            elementTags = np.array(elementTags-1, dtype=int)
            nodeTags = np.array(nodeTags-1, dtype=int)

            # Elements
            Ne = elementTags.shape[0] #nombre d'élements
            elementsID = elementTags

            nPe = GroupElem_Factory.Get_ElemInFos(gmshId)[1] # noeuds par elements
            
            # Construit connect et changes les indices nécessaires
            connect = nodeTags.reshape(Ne, nPe)
            def TriConnect(old, new):
                connect[np.where(connect==old)] = new
            [TriConnect(old, new) for old, new in zip(changes[:,0], changes[:,1])]
            # A tester avec l, c = np.where(connect==changes[:,0])
            
            # Noeuds            
            nodes = np.unique(nodeTags)

            # Verifie que les numéros des noeuds max est bien atteignable dans coordo
            Nmax = nodes.max()
            assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"

            groupElem = GroupElem_Factory.Create_GroupElem(gmshId, connect, elementsID, coordo, nodes)
            if groupElem.dim > dim: dim = groupElem.dim
            dict_groupElem[groupElem.elemType] = groupElem
            
            if groupElem.dim in dimAjoute:
                testDimension = True
            dimAjoute.append(groupElem.dim)

            # Ici on va récupérer les noeuds et elements faisant partie d'un groupe

            if groupElem.dim == 0:
                listPhysicalGroups = physicalGroupsPoint
                listName = namePoint
            elif groupElem.dim == 1:
                listPhysicalGroups = physicalGroupsLine
                listName = nameLine
            elif groupElem.dim == 2:
                listPhysicalGroups = physicalGroupsSurf
                listName = nameSurf
            elif groupElem.dim == 3:
                listPhysicalGroups = physicalGroupsVol
                listName = nameVol
            else:
                listPhysicalGroups = []

            i = -1
            for dim, tag in listPhysicalGroups:
                i += 1
                nodeTags, coord = gmsh.model.mesh.getNodesForPhysicalGroup(groupElem.dim, tag)
                if nodeTags.size == 0: continue
                nodeTags = np.array(nodeTags-1, dtype=int)
                nodes = np.unique(nodeTags)

                def TriNodes(old, new):
                    nodes[np.where(nodes==old)] = new
                [TriNodes(old, new) for old, new in zip(changes[:,0], changes[:,1])]

                # name = gmsh.model.getPhysicalName(groupElem.dim, tag)
                name = listName[i]

                groupElem.Set_Nodes_Tag(nodes, name)
                groupElem.Set_Elements_Tag(nodes, name)

        if dimAjoute.count(dim) > 1:
            assert not testDimension, f"Impossible car {dimAjoute.count(dim)} type d'element {dim}D"
            # TODO faire en sorte de pouvoir le faire ?
            # Peut etre compliqué surtout dans la création des matrices elementaire et assemblage
            # Pas impossible mais pas trivial
        
        tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construction2D(L=10, h=10, taille=3):
        """Construction des maillage possibles en 2D"""

        interfaceGmsh = Interface_Gmsh(affichageGmsh=False, verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), taille=taille)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), taille=taille, isOpen=False)
        lineOpen = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), taille=taille, isOpen=True)
        circle = Circle(Point(x=L/2, y=h/2), L/3, taille=taille, isCreux=True)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, taille=taille, isCreux=False)

        aireDomain = L*h
        aireCircle = np.pi * (circleClose.diam/2)**2

        def testAire(aire):
            assert np.abs(aireDomain-aire)/aireDomain <= 1e-6, "Surface incorrecte"

        # Pour chaque type d'element 2D
        for t, elemType in enumerate(GroupElem.get_Types2D()):

            print(elemType)

            mesh1 = interfaceGmsh.Mesh_Rectangle_2D(domain=domain,elemType=elemType, isOrganised=False)
            testAire(mesh1.aire)
            
            mesh2 = interfaceGmsh.Mesh_Rectangle_2D(domain=domain,elemType=elemType, isOrganised=True)
            testAire(mesh2.aire)

            mesh3 = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain=domain, circle=circle, elemType=elemType)
            # Ici on ne verifie pas car il ya trop peu delement pour bien representer le perçage

            mesh4 = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain=domain, circle=circleClose, elemType=elemType)
            testAire(mesh4.aire)

            mesh5 = interfaceGmsh.Mesh_Rectangle2DAvecFissure(domain=domain, line=line, elemType=elemType)
            testAire(mesh5.aire)

            mesh6 = interfaceGmsh.Mesh_Rectangle2DAvecFissure(domain=domain, line=lineOpen, elemType=elemType)
            testAire(mesh6.aire)

            for m in [mesh1, mesh2, mesh3, mesh4, mesh5, mesh6]:
                list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def Construction3D(L=130, h=13, b=13, taille=7.5, useImport3D=False):
        """Construction des maillage possibles en 3D"""
        # Pour chaque type d'element 3D

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
        circleCreux = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=True)
        circle = Circle(Point(x=L/2, y=0), h*0.7, taille=taille, isCreux=False)

        volume = L*h*b

        def testVolume(val):
            assert np.abs(volume-val)/volume <= 1e-6, "Volume incorrecte"

        folder = Dossier.GetPath()
        cpefPath = Dossier.Join([folder,"3Dmodels","CPEF.stp"])
        partPath = Dossier.Join([folder,"3Dmodels","part.stp"])

        list_mesh3D = []
        for t, elemType in enumerate(GroupElem.get_Types3D()):

            interfaceGmsh = Interface_Gmsh(verbosity=False, affichageGmsh=False)
            
            if useImport3D and elemType == "TETRA4":
                meshCpef = interfaceGmsh.Mesh_Importation3D(cpefPath, tailleElement=10)
                list_mesh3D.append(meshCpef)
                meshPart = interfaceGmsh.Mesh_Importation3D(partPath, tailleElement=taille)
                list_mesh3D.append(meshPart)

            for isOrganised in [True, False]:
                mesh1 = interfaceGmsh.Mesh_Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh1)
                testVolume(mesh1.volume)

            mesh2 = interfaceGmsh.Mesh_PlaqueAvecCercle3D(domain, circleCreux, [0,0,b], elemType=elemType)
            list_mesh3D.append(mesh2)

            mesh3 = interfaceGmsh.Mesh_PlaqueAvecCercle3D(domain, circle, [0,0,b], elemType=elemType)
            list_mesh3D.append(mesh3)
            testVolume(mesh3.volume)

        return list_mesh3D