from typing import cast
import gmsh
import sys
import os
import numpy as np
from colorama import Fore

import Folder
from Geom import *
from GroupElem import GroupElem, ElemType, MatriceType, GroupElem_Factory
from Mesh import Mesh
from TicTac import Tic
import Affichage as Affichage
from Materials import Poutre_Elas_Isot

class Interface_Gmsh:
    """Classe interface Gmsh"""    

    def __init__(self, affichageGmsh=False, gmshVerbosity=False, verbosity=False):
        """Construction d'une interface qui peut interagir avec gmsh

        Parameters
        ----------
        affichageGmsh : bool, optional
            affichage du maillage construit dans gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh peut écrire dans le terminal, by default False
        verbosity : bool, optional
            la classe interfaceGmsh peut écrire le résumé de la construction dans le terminale, by default False
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
        """Vérification si le type d'element est bien utilisable."""
        if dim == 1:
            assert elemType in GroupElem.get_Types1D(), f"Doit être dans {GroupElem.get_Types1D()}"
        if dim == 2:
            assert elemType in GroupElem.get_Types2D(), f"Doit être dans {GroupElem.get_Types2D()}"
        elif dim == 3:
            assert elemType in GroupElem.get_Types3D(), f"Doit être dans {GroupElem.get_Types3D()}"
    
    def __initGmsh(self, factory: str):
        """Initialise gmsh."""
        gmsh.initialize()
        if self.__gmshVerbosity == False:
            gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.model.add("model")
        if factory == 'occ':
            self.__factory = gmsh.model.occ
        elif factory == 'geo':
            self.__factory = gmsh.model.geo
        else:
            raise Exception("Factory inconnue")
    

    def __Loop_From_Points(self, points: list[Point], meshSize: float) -> tuple[int, int]:
        """Création d'une boucle associée à la liste de points.\n
        return loop
        """
        
        factory = self.__factory

        # On creer tout les points
        Npoints = len(points)

        # dictionnaire qui comme clé prend un objet Point et qui contient la liste d'id des points gmsh crées
        dict_point_pointsGmsh = cast(dict[Point, list[int]],{})        

        for index, point in enumerate(points):

            # pi -> id gmsh du point i
            # Pi -> coordonnées du point i           

            # on detecte si le point doit être arrondi
            if point.r == 0:
                # Sans arrondi
                p0 = factory.addPoint(point.x, point.y, point.z, meshSize)
                dict_point_pointsGmsh[point] = [p0]

            else:
                # Avec arrondi

                # Le point courant / actif est le point P0
                # Le point d'après est le point P2
                # Le point d'avant est le point P1        

                # Point / Coint dans lequel on va creer le congé
                P0 = point.coordo

                # Récupère le prochain point
                if index+1 == Npoints:
                    index_p1 = index - 1
                    index_p2 = 0
                elif index == 0:
                    index_p1 = -1
                    index_p2 = index + 1
                else:
                    index_p1 = index - 1
                    index_p2 = index + 1

                # Il faut detecter le point avant P1 et le point P2
                P1 = points[index_p1].coordo
                P2 = points[index_p2].coordo

                # vecteurs
                i = P1-P0
                j = P2-P0
                k = (i+j)/2 # vecteur entre les 2
                n = np.cross(i, j) # vecteur normal au plan formé par i, j

                # angle de i vers k            
                betha = angleBetween_a_b(i, j)/2
                
                d = point.r/np.tan(betha) # disante entre P0 et A sur i et disante entre P0 et B sur j

                d *= np.sign(betha)

                F = matriceJacobienne                

                A = F(i, n).dot(np.array([d,0,0])) + P0
                B = F(j, n).dot(np.array([d,0,0])) + P0
                C = F(i, n).dot(np.array([d, point.r,0])) + P0

                if index > 0:
                    # Récupère le dernier point gmsh crée
                    prevPoint = points[index-1]
                    factory.synchronize()
                    lastPoint = dict_point_pointsGmsh[prevPoint][-1]
                    # récupère les coordonnées du point
                    lastCoordo = gmsh.model.getValue(0, lastPoint, [])

                if index > 0 and np.linalg.norm(lastCoordo - A) <= 1e-12:
                    # si la coordonée est identique on ne recrée pas le point
                    pA = lastPoint
                else:
                    pA = factory.addPoint(A[0], A[1], A[2], meshSize) # point d'intersection entre i et le cercle
                pC = factory.addPoint(C[0], C[1], C[2], meshSize) # centre du cercle                
                pB = factory.addPoint(B[0], B[1], B[2], meshSize) # point d'intersection entre j et le cercle

                dict_point_pointsGmsh[point] = [pA, pC, pB]
            
        lignes = []        

        for index, point in enumerate(points):
            # Pour chaque point on va creer la loop associé au point et on va creer une ligne avec le prochain point
            # Par exemple si le point possède un rayon il va tout dabord falloir construire l'arc de cerlce
            # Par la suite, il est nécessaire de relié le dernier pointGmsh au premier pointGmsh du prochain noeud            

            # les points gmsh créés
            gmshPoints = dict_point_pointsGmsh[point]

            # Si le coin doit être arrondi, il est nécessaire de créer l'arc de cercle
            if point.r > 0:
                lignes.append(factory.addCircleArc(gmshPoints[0], gmshPoints[1], gmshPoints[2]))
                # Ici on supprime le point du centre du cercle TRES IMPORTANT sinon le points reste au centre du cercle
                factory.remove([(0,gmshPoints[1])], False)
                
            # Récupère l'index du prochain noeuds
            if index+1 == Npoints:
                # Si on est sur le dernier noeud on va fermer la boucle en recupérant le premier point
                indexAfter = 0
            else:
                indexAfter = index + 1

            # Récupère le prochain point gmsh pour creer la ligne entre les points
            gmshPointAfter = dict_point_pointsGmsh[points[indexAfter]][0]

            if gmshPoints[-1] != gmshPointAfter:
                # On ne crée pas le lien si les points gmsh sont identiques
                lignes.append(factory.addLine(gmshPoints[-1], gmshPointAfter))

        # Create a closed loop connecting the lines for the surface        
        loop = factory.addCurveLoop(lignes)

        return loop

    def __Loop_From_Circle(self, circle: Circle) -> tuple[int, int]:
        """Création d'une boucle associée à un cercle.\n
        return loop
        """

        factory = self.__factory

        center = circle.center
        rayon = circle.diam/2

        # Points cercle                
        p0 = factory.addPoint(center.x, center.y, center.z, circle.meshSize) #centre
        p1 = factory.addPoint(center.x-rayon, center.y, center.z, circle.meshSize)
        p2 = factory.addPoint(center.x, center.y-rayon, center.z, circle.meshSize)
        p3 = factory.addPoint(center.x+rayon, center.y, center.z, circle.meshSize)
        p4 = factory.addPoint(center.x, center.y+rayon, center.z, circle.meshSize)
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
        """Création d'une boucle associée à un domaine.\n
        return loop
        """
        pt1 = domain.pt1
        pt2 = domain.pt2

        p1 = Point(x=pt1.x, y=pt1.y, z=pt1.z)
        p2 = Point(x=pt2.x, y=pt1.y, z=pt1.z)
        p3 = Point(x=pt2.x, y=pt2.y, z=pt2.z)
        p4 = Point(x=pt1.x, y=pt2.y, z=pt2.z)
        # Ici laisser les coordonnées en z à 0

        loop = self.__Loop_From_Points([p1, p2, p3, p4], domain.meshSize)
        
        return loop

    def __Surface_From_Loops(self, loops: list[int]) -> tuple[int, int]:
        """Création d'une surface associée à une boucle.\n
        return surface
        """

        surface = self.__factory.addPlaneSurface(loops)

        return surface    
    
    def __Add_PhysicalPoint(self, point: int) -> int:
        """Ajoute le point dans le physical group"""
        pgPoint = gmsh.model.addPhysicalGroup(0, [point], name=f"P{point}")
        return pgPoint

    def __Add_PhysicalLine(self, ligne: int) -> int:
        """Ajoute la ligne dans les physical group"""
        pgLine = gmsh.model.addPhysicalGroup(1, [ligne], name=f"L{ligne}")
        return pgLine

    def __Add_PhysicalSurface(self, surface: int) -> int:
        """Ajoute la surface fermée ou ouverte dans les physical group"""
        pgSurf = gmsh.model.addPhysicalGroup(2, [surface], name=f"S{surface}")
        return pgSurf
    
    def __Add_PhysicalVolume(self, volume: int) -> int:
        """Ajoute le volume fermée ou ouverte dans les physical group."""
        pgVol = gmsh.model.addPhysicalGroup(3, [volume], name=f"V{volume}")
        return pgVol

    def __Add_PhysicalGroup(self, dim: int, tag: int):
        if dim == 0:
            self.__Add_PhysicalPoint(tag)
        elif dim == 1:
            self.__Add_PhysicalLine(tag)
        elif dim == 2:
            self.__Add_PhysicalSurface(tag)
        elif dim == 3:
            self.__Add_PhysicalVolume(tag)

    def __Set_PhysicalGroups(self, buildPoint=True, buildLine=True, buildSurface=True, buildVolume=True):
        """Création des groupes physiques en fonction des entités du modèle."""
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

        [self.__Add_PhysicalGroup(dim, tag) for dim, tag in zip(entities[:,0], entities[:,1])]

    __dict_name_dim = {
        0 : "P",
        1 : "L",
        2 : "S",
        3 : "V"
    }

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

        extruEntities = []

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
            elif elemType in [ElemType.TETRA4, ElemType.TETRA10]:
                numElements = []
                combine = False
            
            # Creer les nouveaux elements pour l'extrusion
            # nCouches = np.max([np.ceil(np.abs(extrude[2] - domain.taille)), 1])
            extru = factory.extrude([(2, surf)], extrude[0], extrude[1], extrude[2], recombine=combine, numElements=numElements)

            extruEntities.extend(extru)

        return extruEntities

    # TODO générer plusieurs maillage en désactivant initGmsh et en utilisant plusieurs fonctions ?
    # mettre en place une liste de surfaces ?

    def __Set_BackgroundMesh(self, refineGeom, tailleOut: float):
        """Renseigne un maillage de fond

        Parameters
        ----------
        refineGeom : Objet geom de fond
            Objet geometrique pour le maillage de fond
        tailleOut : float
            taille des élements en dehors du domaine
        """

        # Exemple extrait de t10.py dans les tutos gmsh

        # Regarder aussi t11.py pour faire une ligne

        if isinstance(refineGeom, Domain):

            assert not refineGeom.meshSize == 0, "Il faut définir une taille d'element pour le domaine"

            pt21 = refineGeom.pt1
            pt22 = refineGeom.pt2
            taille2 = refineGeom.meshSize

            # We could also use a `Box' field to impose a step change in element sizes
            # inside a box
            field_Box = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field_Box, "VIn", taille2)
            gmsh.model.mesh.field.setNumber(field_Box, "VOut", tailleOut)
            gmsh.model.mesh.field.setNumber(field_Box, "XMin", np.min([pt21.x, pt22.x]))
            gmsh.model.mesh.field.setNumber(field_Box, "XMax", np.max([pt21.x, pt22.x]))
            gmsh.model.mesh.field.setNumber(field_Box, "YMin", np.min([pt21.y, pt22.y]))
            gmsh.model.mesh.field.setNumber(field_Box, "YMax", np.max([pt21.y, pt22.y]))
            gmsh.model.mesh.field.setNumber(field_Box, "ZMin", np.min([pt21.z, pt22.z]))
            gmsh.model.mesh.field.setNumber(field_Box, "ZMax", np.max([pt21.z, pt22.z]))
            # gmsh.model.mesh.field.setNumber(field_Box, "Thickness", np.abs(pt21.z - pt22.z))

            # Let's use the minimum of all the fields as the background mesh field:
            minField = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [field_Box])

            gmsh.model.mesh.field.setAsBackgroundMesh(minField)

        elif isinstance(refineGeom, Circle):

            loopCercle = self.__Loop_From_Circle(refineGeom)

            field_Distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(1, "PointsList", [loopCercle])
            
            field_Thershold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(field_Thershold, "InField", field_Distance)            
            gmsh.model.mesh.field.setNumber(field_Thershold, "SizeMin", refineGeom.meshSize)
            gmsh.model.mesh.field.setNumber(field_Thershold, "SizeMax", tailleOut)
            gmsh.model.mesh.field.setNumber(field_Thershold, "DistMin", 0.15)
            gmsh.model.mesh.field.setNumber(field_Thershold, "DistMax", 0.5)

            minField = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [field_Thershold])

            gmsh.model.mesh.field.setAsBackgroundMesh(minField)

        elif isinstance(refineGeom, str):

            if not os.path.exists(refineGeom) :
                print(Fore.RED + "Le fichier .pos renseignée n'existe pas" + Fore.WHITE)
                return

            if ".pos" not in refineGeom:
                print(Fore.RED + "Doit fournir un fichier .pos" + Fore.WHITE)
                return

            gmsh.merge(refineGeom)

            # Add the post-processing view as a new size field:
            minField = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(minField, "ViewIndex", 0)

            # Apply the view as the current background mesh size field:
            gmsh.model.mesh.field.setAsBackgroundMesh(minField)

            # In order to compute the mesh sizes from the background mesh only, and
            # disregard any other size constraints, one can set:
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            

    def Mesh_Import_msh(self, fichier: str, coef=1, setPhysicalGroups=False):
        """Importation d'un fichier .msh

        Parameters
        ----------
        fichier : str
            fichier (.msh) que gmsh va charger pour creer le maillage
        coef : int, optional
            coef appliqué aux coordonnées des noeuds, by default 1
        setPhysicalGroups : bool, optional
            récupération des entités pour créer des groupes physiques d'éléments, by default False

        Returns
        -------
        Mesh
            Maillage construit
        """
        # 

        self.__initGmsh('occ')

        gmsh.open(fichier)

        if setPhysicalGroups:
            self.__Set_PhysicalGroups()

        return self.__Recuperation_Maillage(coef)

    def Mesh_Import_part3D(self, fichier: str, meshSize: float, refineGeom=None, folder=""):
        """Construis le maillage 3D depuis l'importation d'un fichier 3D et création du maillage (.stp ou .igs)

        Parameters
        ----------
        fichier : str
            fichier (.stp, .igs) que gmsh va charger pour créer le maillage
        meshSize : float
            taille de maille
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """
        # Lorsqu'on importe une pièce on ne peut utiliser que du TETRA4
        elemType = ElemType.TETRA4
        # Permettre d'autres maillage -> ça semble impossible il faut creer le maillage par gmsh pour maitriser le type d'element

        self.__initGmsh('occ') # Ici ne fonctionne qu'avec occ !! ne pas changer

        assert meshSize >= 0.0, "Doit être supérieur ou égale à 0"
        self.__CheckType(3, elemType)
        
        tic = Tic()

        factory = self.__factory

        if '.stp' in fichier or '.igs' in fichier:
            factory.importShapes(fichier)
        else:
            print("Doit être un fichier .stp")

        self.__Set_BackgroundMesh(refineGeom, meshSize)

        self.__Set_PhysicalGroups(buildPoint=False, buildLine=True, buildSurface=True, buildVolume=False)

        gmsh.option.setNumber("Mesh.MeshSizeMin", meshSize)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)

        tic.Tac("Mesh","Importation du fichier step", self.__verbosity)

        self.__Construction_Maillage(3, elemType, folder=folder)

        return self.__Recuperation_Maillage()

    def Mesh_Domain_3D(self, domain: Domain, extrude=[0,0,1], nCouches=1, elemType=ElemType.HEXA8, refineGeom=None, isOrganised=True, folder=""):
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
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        isOrganised : bool, optional
            le maillage est organisé, by default True
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

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
        surfaces = self.Mesh_Domain_2D(domain, elemType=ElemType.TRI3, refineGeom=refineGeom, isOrganised=isOrganised, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=isOrganised, nCouches=nCouches)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction Poutre3D", self.__verbosity)

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=isOrganised, folder=folder)
        
        return self.__Recuperation_Maillage()

    def Mesh_Domain_2D(self, domain: Domain, elemType=ElemType.TRI3, refineGeom=None, isOrganised=False, folder="", returnSurfaces=False):
        """Maillage d'un rectange 2D

        Parameters
        ----------
        domain : Domain
            domaine 2D qui doit être dans le plan (x,y)
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        isOrganised : bool, optional
            le maillage est organisé, by default False
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""
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

        self.__Set_BackgroundMesh(refineGeom, domain.meshSize)

        if returnSurfaces: return [surface]

        self.__Set_PhysicalGroups()
        
        tic.Tac("Mesh","Construction Rectangle", self.__verbosity)
        
        self.__Construction_Maillage(2, elemType, surfaces=[surface], isOrganised=isOrganised, folder=folder)
        
        return self.__Recuperation_Maillage()

    def __PhysicalGroups_craks(self, cracks: list, entities: list[tuple]):
        """Création des groupes physiques associés aux fissures\n
        return crackLines, crackSurfaces, openPoints, openLines
        """

        dim = entities[0][0]
        
        # listes contenants les entitées ouvertes
        openPoints = []
        openLines = []
        openSurfaces = []        

        entities0D = []
        entities1D = []
        entities2D = []

        for crack in cracks:
            if isinstance(crack, Line):
                # Création des points
                pt1 = crack.pt1
                p1 = self.__factory.addPoint(pt1.x, pt1.y, pt1.z, crack.meshSize)
                pt2 = crack.pt2
                p2 = self.__factory.addPoint(pt2.x, pt2.y, pt2.z, crack.meshSize)

                # Création de la ligne
                line = self.__factory.addLine(p1, p2)
                entities1D.append(line)
                if crack.isOpen:
                    openLines.append(line)

                if pt1.isOpen:
                    entities0D.append(p1)
                    openPoints.append(p1)
                    # o1, m1 = self.__factory.fragment([(0, p1), (1, line)], entities)
                    # openPoints.append(p1)
                    # entities0D.append(p1)

                if pt2.isOpen:
                    entities0D.append(p2)
                    openPoints.append(p2)
                    # o2, m2 = self.__factory.fragment([(0, p2), (1, line)], entities)
                    # openPoints.append(p2)
                    # entities0D.append(p2)
                
                # self.__factory.synchronize()
                # gmsh.model.mesh.embed(1, [line], dim, entities[0][1])
                
            else:
                # Récupération des boucles
                hollowLoops, filledLoops = self.__Get_hollowLoops_And_filledLoops([crack])
                loops = []; loops.extend(hollowLoops); loops.extend(filledLoops)
                
                # Consutruction des surfaces
                for loop in loops:
                    surface = self.__Surface_From_Loops([loop])
                    entities2D.append(surface)

                if crack.isCreux:
                    openSurfaces.append(surface)

                # self.__factory.synchronize()
                # gmsh.model.mesh.embed(2, [surface], dim, entities[0][1])


        newEntities = [(0, point) for point in entities0D]
        newEntities.extend([(1, line) for line in entities1D])
        newEntities.extend([(2, surf) for surf in entities2D])
        
        # o, m = gmsh.model.occ.fragment(newEntities, entities)
        o, m = gmsh.model.occ.fragment(entities, newEntities)
        self.__factory.synchronize()
        
        # if dim == 2:
        #     gmsh.model.mesh.embed(1, entities1D, 2, entities[0][1])
        # elif dim == 3:
        #     gmsh.model.mesh.embed(2, entities2D, 3, entities[0][1])        

        crackLines = gmsh.model.addPhysicalGroup(1, openLines) if len(openLines) > 0 else None
        crackSurfaces = gmsh.model.addPhysicalGroup(2, openSurfaces) if len(openSurfaces) > 0 else None

        openPoints = gmsh.model.addPhysicalGroup(0, openPoints) if len(openPoints) > 0 else None
        openLines = gmsh.model.addPhysicalGroup(1, openLines) if len(openLines) > 0 else None

        return crackLines, crackSurfaces, openPoints, openLines
        

    def Mesh_Domain_Lines_2D(self, domain: Domain, cracks: list[Line], elemType=ElemType.TRI3, refineGeom=None, folder=""):
        """Maillage d'un rectangle avec une fissure

        Parameters
        ----------
        domain : Domain
            domaine 2D qui doit etre compris dans le plan (x,y)
        cracks : list[Line]
            list de ligne pour la création de fissure
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh('occ')                
        
        self.__CheckType(2, elemType)
        
        tic = Tic()

        # Création de la surface
        loopDomain = self.__Loop_From_Domain(domain)

        # Création de la surface
        surfaceDomain = self.__Surface_From_Loops([loopDomain])

        physicalSurface = gmsh.model.addPhysicalGroup(2, [surfaceDomain])

        # Création des fissures
        crackLines, crackSurfaces, openPoints, openLines = self.__PhysicalGroups_craks(cracks, [(2, physicalSurface)])        

        # Regénération des groupes physiques
        self.__Set_PhysicalGroups(buildSurface=False)
        
        self.__Set_BackgroundMesh(refineGeom, domain.meshSize)

        tic.Tac("Mesh","Construction rectangle fissuré", self.__verbosity)

        self.__Construction_Maillage(2, elemType, surfaces=[physicalSurface], crackLines=crackLines, openPoints=openPoints, isOrganised=False)
        
        return self.__Recuperation_Maillage()

    def Mesh_Domain_Circle_2D(self, domain: Domain, circle: Circle, elemType=ElemType.TRI3, refineGeom=None, folder="", returnSurfaces=False):
        """Construis le maillage 2D d'un rectangle un cercle (creux ou fermé)

        Parameters
        ----------
        domain : Domain
            surface qui doit être contenu dans le plan (x,y)
        circle : Circle
            cercle creux ou plein
        elemType : str, optional
            type d'element utilisé, by default "TRI3"
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""
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
            p0 = factory.addPoint(circle.center.x, circle.center.y, circle.center.z, circle.meshSize)
            factory.synchronize()
            gmsh.model.mesh.embed(0, [p0], 2, surfaceCercle)
            factory.synchronize()
            surfaces = [surfaceCercle, surfaceDomain]

        self.__Set_BackgroundMesh(refineGeom, domain.meshSize)
        
        if returnSurfaces: return surfaces

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(2, elemType, surfaces=surfaces, isOrganised=False, folder=folder)

        return self.__Recuperation_Maillage()

    def Mesh_Domain_Circle_3D(self, domain: Domain, circle: Circle, extrude=[0,0,1], nCouches=1, elemType=ElemType.HEXA8, refineGeom=None, folder=""):
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
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

        Returns
        -------
        Mesh
            Maillage construit
        """

        self.__initGmsh("occ")
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_Domain_Circle_2D(domain, circle, elemType=ElemType.TRI3, refineGeom=refineGeom, folder=folder, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=False, nCouches=nCouches)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","PlaqueAvecCercle3D", self.__verbosity)

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=False, folder=folder)
        
        return self.__Recuperation_Maillage()

    def Mesh_Lines_1D(self, listPoutres: list[Poutre_Elas_Isot], elemType=ElemType.SEG2 ,folder=""):
        """Construction d'un maillage de segment

        Parameters
        ----------
        listPoutre : list[Poutre]
            liste de Poutres
        elemType : str, optional
            type d'element, by default "SEG2" ["SEG2", "SEG3"]
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

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

            p1 = factory.addPoint(x1, y1, z1, line.meshSize)
            p2 = factory.addPoint(x2, y2, z2, line.meshSize)
            listPoints.append(p1)
            listPoints.append(p2)

            ligne = factory.addLine(p1, p2)
            listeLines.append(ligne)

            factory.synchronize()
            gmsh.model.addPhysicalGroup(1, [ligne], name=f"{poutre.name}")

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(1, elemType, surfaces=[], folder=folder)

        return self.__Recuperation_Maillage()

    def __Get_hollowLoops_And_filledLoops(self, inclusions: list) -> tuple[list, list]:
        """Création des boucles les liste de boucles creuses et pleines

        Parameters
        ----------
        inclusions : list
            Liste d'objet géométrique contenu dans le domaine

        Returns
        -------
        tuple[list, list]
            toutes les boucles créés, suivit des boucles pleines (non creuses)
        """
        loops = []
        filledLoops = []
        for objetGeom in inclusions:
            if isinstance(objetGeom, Circle):
                loop = self.__Loop_From_Circle(objetGeom)
            elif isinstance(objetGeom, Domain):                
                loop = self.__Loop_From_Domain(objetGeom)
            elif isinstance(objetGeom, PointsList):                
                loop = self.__Loop_From_Points(objetGeom.points, objetGeom.meshSize)
            loops.append(loop)

            if not objetGeom.isCreux:
                filledLoops.append(loop)

        return loops, filledLoops

    def Mesh_Points_2D(self, pointsList: PointsList, elemType=ElemType.TRI3, inclusions=[], cracks=[], refineGeom=None, folder="", returnSurfaces=False):
        """Construis le maillage 2D en créant une surface depuis une liste de points

        Parameters
        ----------
        points : PointsList
            liste de points
        elemType : str, optional
            type d'element, by default "TRI3" ["TRI3", "TRI6", "QUAD4", "QUAD8"]
        inclusions : list[Domain, Circle, PointsList], optional
            liste d'objets creux ou non à l'intérieur du domaine 
        cracks : list[Line]
            liste de ligne utilisées pour la création de fissures
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""
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

        points = pointsList.points
        meshSize = pointsList.meshSize

        # Création de la surface de contour
        loopSurface = self.__Loop_From_Points(points, meshSize)

        # Création de toutes les boucles associés aux objets à l'intérieur du domaine
        hollowLoops, filledLoops = self.__Get_hollowLoops_And_filledLoops(inclusions)

        # Pour chaque objetGeom plein, il est nécessaire de créer une surface
        surfacesPleines = [factory.addPlaneSurface([loop]) for loop in filledLoops]
        
        listeLoop = [loopSurface] # surface du domaine
        listeLoop.extend(hollowLoops) # On rajoute les surfaces creuses

        surfaceDomain = self.__Surface_From_Loops(listeLoop)

        # Rajoute la surface du domaine en dernier
        surfacesPleines.insert(0, surfaceDomain)

        # Récupère l'entité 3D
        self.__factory.synchronize()
        entities2D = gmsh.model.getEntities(2)

        # Création des fissures
        crackLines, crackSurfaces, openPoints, openLines = self.__PhysicalGroups_craks(cracks, entities2D)
        
        physicalSurfaces = [gmsh.model.addPhysicalGroup(2, [surface]) for surface in surfacesPleines]
        
        # Création des surfaces creuses
        if returnSurfaces: return surfacesPleines

        self.__Set_BackgroundMesh(refineGeom, meshSize)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Construction plaque trouée", self.__verbosity)

        self.__Construction_Maillage(2, elemType, surfaces=physicalSurfaces, crackLines=crackLines, openPoints=openPoints, isOrganised=False, folder=folder)

        return self.__Recuperation_Maillage()

    def Mesh_Points_3D(self, pointsList: PointsList, extrude=[0,0,1], nCouches=1, elemType=ElemType.TETRA4, inclusions=[], cracks=[], refineGeom=None, folder=""):
        """Construction d'un maillage 3D depuis une liste de points

        Parameters
        ----------
        pointsList : PointsList
            liste de points
        extrude : list, optional
            extrusion, by default [0,0,1]
        nCouches : int, optional
            nombre de couches dans l'extrusion, by default 1
        elemType : str, optional
            type d'element, by default "TETRA4" ["TETRA4", "HEXA8", "PRISM6"]
        inclusions : list[Domain, Circle, PointsList], optional
            liste d'objets creux ou non à l'intérieur du domaine
        cracks : list[Geom]
            liste de ligne utilisées pour la création de fissures
        refineGeom : Geom, optional
            deuxième domaine pour la concentration de maillage, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""

        Returns
        -------
        Mesh
            Maillage 3D
        """

        self.__initGmsh('occ')
        self.__CheckType(3, elemType)
        
        tic = Tic()

        cracks1D = [crack for crack in cracks if isinstance(crack, Line)]
        
        # le maillage 2D de départ n'a pas d'importance
        surfaces = self.Mesh_Points_2D(pointsList, elemType=ElemType.TRI3, inclusions=inclusions, cracks=[], refineGeom=refineGeom, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, isOrganised=False, nCouches=nCouches)        

        # Récupère l'entité 3D
        self.__factory.synchronize()
        entities3D = gmsh.model.getEntities(3)

        # Création des fissures
        crackLines, crackSurfaces, openPoints, openLines = self.__PhysicalGroups_craks(cracks, entities3D)

        self.__Set_BackgroundMesh(refineGeom, pointsList.meshSize)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Mesh from points", self.__verbosity)

        surfaces = entities3D[0][1]

        self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=False, folder=folder,
        crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)

        # self.__Construction_Maillage(3, elemType, surfaces=surfaces, isOrganised=False, folder=folder)
        
        return self.__Recuperation_Maillage()
    
    def Create_posFile(self, coordo: np.ndarray, values: np.ndarray, folder: str, filename="data") -> str:

        assert isinstance(coordo, np.ndarray), "Doit être une array numpy"
        assert coordo.shape[1] == 3, "Doit être de dimension (n, 3)"

        assert values.shape[0] == coordo.shape[0], "values et coordo ne sont pas de la bonne dimension"

        data = np.append(coordo, values.reshape(-1, 1), axis=1)

        z = coordo[:,2]

        self.__initGmsh("occ")

        view = gmsh.view.add("view for new mesh")

        gmsh.view.addListData(view, "SP", coordo.shape[0], data.reshape(-1))

        path = Folder.Join([folder, f"{filename}.pos"])

        gmsh.view.write(view, path)

        return path

    @staticmethod
    def __Set_order(elemType: str):
        if elemType in ["TRI3","QUAD4"]:
            gmsh.model.mesh.set_order(1)
        elif elemType in ["SEG3", "TRI6", "QUAD8", "TETRA10"]:
            if elemType in ["QUAD8"]:
                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
            gmsh.model.mesh.set_order(2)
        elif elemType in ["SEG4", "TRI10"]:
            gmsh.model.mesh.set_order(3)
        elif elemType in ["SEG5", "TRI15"]:
            gmsh.model.mesh.set_order(4)


    def __Construction_Maillage(self, dim: int, elemType: str, surfaces=[], isOrganised=False, crackLines=None, crackSurfaces=None, openPoints=None, openLines=None, folder=""):
        """Construction du maillage gmsh depuis la geométrie qui a été construit ou importée.

        Parameters
        ----------
        dim : int
            dimension du maillage
        elemType : str
            type d'element
        surfaces : list[int], optional
            liste de surfaces que l'on va mailler, by default []
        isOrganised : bool, optional
            le maillage est organisé, by default False
        crackLines : int, optional
            groupePhysique qui regroupe toutes les fissures sur les lignes, by default None
        crackSurfaces : int, optional
            groupePhysique qui regroupe toutes les fissures sur les lignes, by default None
        openPoints : int, optional
            groupePhysique de points qui peuvent s'ouvrir, by default None
        openLines : int, optional
            groupePhysique de lignes qui peuvent s'ouvrir, by default None
        folder : str, optional
            dossier de sauvegarde du maillage mesh.msh, by default ""
        """

        factory = self.__factory

        if factory == gmsh.model.occ:
            isOrganised = False
            factory = cast(gmsh.model.occ, factory)
        elif factory == gmsh.model.geo:
            factory = cast(gmsh.model.geo, factory)
        else:
            raise Exception("factory inconnue")

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
                
                self.__factory.synchronize()
                gmsh.model.mesh.setRecombine(3, 1)
            
            gmsh.model.mesh.generate(3)

            Interface_Gmsh.__Set_order(elemType)

        # Il faut passer un seul groupe physique pour les lignes et les points

        usePluginCrack = False
        if dim == 2:
            if crackLines != None:
                gmsh.plugin.setNumber("Crack", "Dimension", 1)
                gmsh.plugin.setNumber("Crack", "PhysicalGroup", crackLines)
                usePluginCrack=True
            if openPoints != None:
                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openPoints)
        elif dim == 3:
            if crackSurfaces != None:
                gmsh.plugin.setNumber("Crack", "Dimension", 2)
                gmsh.plugin.setNumber("Crack", "PhysicalGroup", crackSurfaces)
                usePluginCrack=True
            if openLines != None:
                gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openLines)        

        if usePluginCrack:
            gmsh.plugin.run("Crack")
            
        
        # Ouvre l'interface de gmsh si necessaire
        if '-nopopup' not in sys.argv and self.__affichageGmsh:
            gmsh.fltk.run()
        
        tic.Tac("Mesh","Construction du maillage gmsh", self.__verbosity)

        if folder != "":
            # gmsh.write(Dossier.Join([folder, "model.geo"])) # Il semblerait que ça marche pas c'est pas grave          
            self.__factory.synchronize()  
            # gmsh.model.geo.synchronize()
            # gmsh.model.occ.synchronize()
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            gmsh.write(Folder.Join([folder, "mesh.msh"]))
            tic.Tac("Mesh","Sauvegarde du .msh", self.__verbosity)

    def __Recuperation_Maillage(self, coef=1):
        """Récupération du maillage construit

        Returns
        -------
        Mesh
            Maillage construit
        """

        # TODO rendre cette classe accessible

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

        # Ici on va detecter les saut dans la numérotations des noeuds
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

        # Applique le coef
        coordo = coordo * coef
        
        # Construit les groupes physiques
        physicalGroups = gmsh.model.getPhysicalGroups()
        pgArray = np.array(physicalGroups)
        # A optimiser
        physicalGroupsPoint = []; namePoint = []
        physicalGroupsLine = []; nameLine = []
        physicalGroupsSurf = []; nameSurf = []
        physicalGroupsVol = []; nameVol = []

        nbPhysicalGroup = 0

        def __name(dim: int, n: int) -> str:
            # Construit le nom de l'entitié
            index = n+nbPhysicalGroup
            tag = physicalGroups[index][1]
            name = gmsh.model.getPhysicalName(dim, tag)

            if name == "":
                name = f"{Interface_Gmsh.__dict_name_dim[dim]}{n+1}"

            return name

        for dim in range(pgArray[:,0].max()+1):
            # Pour chaque dimensions disponibles dans les groupes physiques.
            
            # On récupère les entités de la dimension
            indexDim = np.where(pgArray[:,0] == dim)[0]
            listTupleDim = tuple(map(tuple, pgArray[indexDim]))
            nbEnti = indexDim.size

            # En fonction de la dimension des entité on va leurs données des noms
            # Puis on va ajouter les entités les tuples (dim, tag) a la liste de groupePhysique associé à la dimension.
            if dim == 0:
                namePoint.extend([f"{__name(dim, n)}" for n in range(nbEnti)])
                nbEnti = len(namePoint)
                physicalGroupsPoint.extend(listTupleDim)
            elif dim == 1:
                nameLine.extend([__name(dim, n) for n in range(nbEnti)])
                nbEnti = len(nameLine)
                physicalGroupsLine.extend(listTupleDim)
            elif dim == 2:
                nameSurf.extend([f"{__name(dim, n)}" for n in range(nbEnti)])
                nbEnti = len(nameSurf)
                physicalGroupsSurf.extend(listTupleDim)
            elif dim == 3:
                nameVol.extend([f"{__name(dim, n)}" for n in range(nbEnti)])
                nbEnti = len(nameVol)
                physicalGroupsVol.extend(listTupleDim)

            nbPhysicalGroup += nbEnti

        # On verifie quon a bien tout ajouté
        assert len(physicalGroups) == nbPhysicalGroup

        # Construit les groupes d'elements        
        dimAjoute = []
        meshDim = pgArray[:,0].max()

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

            # Création du groupe d'element 
            groupElem = GroupElem_Factory.Create_GroupElem(gmshId, connect, elementsID, coordo, nodes)
            
            # On rajoute le groupe d'element au dictionnaire contenant tout les groupes
            dict_groupElem[groupElem.elemType] = groupElem
            
            # On verifie que le maillage ne possède pas un groupe d'element de cette dimension
            if groupElem.dim in dimAjoute and groupElem.dim == meshDim:
                raise Exception(f"Récupération du maillage impossible car {dimAjoute.count(meshDim)+1} type d'element {meshDim}D")
                # TODO faire en sorte de pouvoir le faire ?
                # Peut etre compliqué surtout dans la création des matrices elementaire et assemblage
                # Pas impossible mais pas trivial
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

            # Pour chaque groupe physique je vais venir récupéré les noeuds
            # et associé les tags
            i = -1
            for dim, tag in listPhysicalGroups:
                i += 1

                # name = gmsh.model.getPhysicalName(groupElem.dim, tag)
                name = listName[i]

                nodeTags, coord = gmsh.model.mesh.getNodesForPhysicalGroup(groupElem.dim, tag)
                # Si aucun noeud à été récupéré passe au prochain groupePhysique
                if nodeTags.size == 0: continue

                # Récupération de la liste de noeud unique
                nodeTags = np.array(nodeTags-1, dtype=int)
                nodes = np.unique(nodeTags)

                def TriNodes(old, new):
                    nodes[np.where(nodes==old)] = new
                [TriNodes(old, new) for old, new in zip(changes[:,0], changes[:,1])]

                groupElem.Set_Nodes_Tag(nodes, name)
                groupElem.Set_Elements_Tag(nodes, name)
        
        tic.Tac("Mesh","Récupération du maillage gmsh", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construction2D(L=10, h=10, taille=3):
        """Construction des maillage possibles en 2D"""

        interfaceGmsh = Interface_Gmsh(affichageGmsh=False, verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), meshSize=taille)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=taille, isOpen=False)
        lineOpen = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=taille, isOpen=True)
        circle = Circle(Point(x=L/2, y=h/2), L/3, meshSize=taille, isCreux=True)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, meshSize=taille, isCreux=False)

        aireDomain = L*h
        aireCircle = np.pi * (circleClose.diam/2)**2

        def testAire(aire):
            assert np.abs(aireDomain-aire)/aireDomain <= 1e-6, "Surface incorrecte"

        # Pour chaque type d'element 2D
        for t, elemType in enumerate(GroupElem.get_Types2D()):

            print(elemType)

            mesh1 = interfaceGmsh.Mesh_Domain_2D(domain=domain,elemType=elemType, isOrganised=False)
            testAire(mesh1.aire)
            
            mesh2 = interfaceGmsh.Mesh_Domain_2D(domain=domain,elemType=elemType, isOrganised=True)
            testAire(mesh2.aire)

            mesh3 = interfaceGmsh.Mesh_Domain_Circle_2D(domain=domain, circle=circle, elemType=elemType)
            # Ici on ne verifie pas car il ya trop peu delement pour bien representer le perçage

            mesh4 = interfaceGmsh.Mesh_Domain_Circle_2D(domain=domain, circle=circleClose, elemType=elemType)
            testAire(mesh4.aire)

            mesh5 = interfaceGmsh.Mesh_Domain_Lines_2D(domain=domain, cracks=[line], elemType=elemType)
            testAire(mesh5.aire)

            mesh6 = interfaceGmsh.Mesh_Domain_Lines_2D(domain=domain, cracks=[lineOpen], elemType=elemType)
            testAire(mesh6.aire)

            for m in [mesh1, mesh2, mesh3, mesh4, mesh5, mesh6]:
                list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def Construction3D(L=130, h=13, b=13, taille=7.5, useImport3D=False):
        """Construction des maillage possibles en 3D"""
        # Pour chaque type d'element 3D

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=taille)
        circleCreux = Circle(Point(x=L/2, y=0,z=-b/2), h*0.7, meshSize=taille, isCreux=True)
        circle = Circle(Point(x=L/2, y=0 ,z=-b/2), h*0.7, meshSize=taille, isCreux=False)

        volume = L*h*b

        def testVolume(val):
            assert np.abs(volume-val)/volume <= 1e-6, "Volume incorrecte"

        folder = Folder.Get_Path()
        cpefPath = Folder.Join([folder,"3Dmodels","CPEF.stp"])
        partPath = Folder.Join([folder,"3Dmodels","part.stp"])

        list_mesh3D = []
        for t, elemType in enumerate(GroupElem.get_Types3D()):

            interfaceGmsh = Interface_Gmsh(verbosity=False, affichageGmsh=False)
            
            if useImport3D and elemType == "TETRA4":
                meshCpef = interfaceGmsh.Mesh_Import_part3D(cpefPath, meshSize=10)
                list_mesh3D.append(meshCpef)
                meshPart = interfaceGmsh.Mesh_Import_part3D(partPath, meshSize=taille)
                list_mesh3D.append(meshPart)

            for isOrganised in [True, False]:
                mesh1 = interfaceGmsh.Mesh_Domain_3D(domain, [0,0,b], elemType=elemType, isOrganised=isOrganised)
                list_mesh3D.append(mesh1)
                testVolume(mesh1.volume)

            mesh2 = interfaceGmsh.Mesh_Domain_Circle_3D(domain, circleCreux, [0,0,b], elemType=elemType)
            list_mesh3D.append(mesh2)

            mesh3 = interfaceGmsh.Mesh_Domain_Circle_3D(domain, circle, [0,0,b], elemType=elemType)
            list_mesh3D.append(mesh3)
            testVolume(mesh3.volume)

        return list_mesh3D