from typing import cast
import gmsh
import sys
import os
import numpy as np
from colorama import Fore

import Folder
from Geom import *
from GroupElem import GroupElem, ElemType, MatrixType, GroupElem_Factory
from Mesh import Mesh
from TicTac import Tic
import Display as Display
from Materials import Beam_Elas_Isot

class Interface_Gmsh:

    def __init__(self, affichageGmsh=False, gmshVerbosity=False, verbosity=False):
        """Building an interface that can interact with gmsh

        Parameters
        ----------
        affichageGmsh : bool, optional
            display mesh built in gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh can write to terminal, by default False
        verbosity : bool, optional
            interfaceGmsh class can write construction summary to terminal, by default False
        """
    
        self.__affichageGmsh = affichageGmsh
        """gmsh can display the mesh"""
        self.__gmshVerbosity = gmshVerbosity
        """gmsh can write to the console"""
        self.__verbosity = verbosity
        """the interface can write to the console"""

        if gmshVerbosity:
            Display.Section("New interface with gmsh")

    def __CheckType(self, dim: int, elemType: str):
        """Check that the element type is usable."""
        if dim == 1:
            assert elemType in GroupElem.get_Types1D(), f"Must be in {GroupElem.get_Types1D()}"
        if dim == 2:
            assert elemType in GroupElem.get_Types2D(), f"Must be in {GroupElem.get_Types2D()}"
        elif dim == 3:
            assert elemType in GroupElem.get_Types3D(), f"Must be in {GroupElem.get_Types3D()}"
    
    def __initGmsh(self, factory: str):
        """Initialize gmsh."""
        gmsh.initialize()
        if self.__gmshVerbosity == False:
            gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.model.add("model")
        if factory == 'occ':
            self.__factory = gmsh.model.occ
        elif factory == 'geo':
            self.__factory = gmsh.model.geo
        else:
            raise Exception("Unknow factory")
        
    def __Loop_From_Geom(self, geom: Geom) -> int:
        """Creation of a loop based on the geometric object."""

        if isinstance(geom, Circle):
            loop = self.__Loop_From_Circle(geom)
        elif isinstance(geom, Domain):                
            loop = self.__Loop_From_Domain(geom)
        elif isinstance(geom, PointsList):                
            loop = self.__Loop_From_Points(geom.points, geom.meshSize)[0]
        elif isinstance(geom, Contour):
            loop = self.__Loop_From_Contour(geom)[0]
        else:
            raise Exception("Must be a circle, a domain, a list of points or a contour.")
        
        return loop

    def __Loop_From_Points(self, points: list[Point], meshSize: float) -> tuple[int, list[int], list[int]]:
        """Creation of a loop associated with the list of points.
        return loop, lines, openPoints
        """
        
        factory = self.__factory

        # We create all the points
        Npoints = len(points)

        # dictionary, which takes a Point object as key and contains the id list of gmsh points created
        dict_point_pointsGmsh = cast(dict[Point, list[int]],{})
        
        openPoints = []

        for index, point in enumerate(points):

            # pi -> gmsh id of point i
            # Pi -> coordinates of point i
            if index > 0:
                # Retrieves the last gmsh point created
                prevPoint = points[index-1]
                factory.synchronize()
                lastPoint = dict_point_pointsGmsh[prevPoint][-1]
                # retrieves point coordinates
                lastCoordo = gmsh.model.getValue(0, lastPoint, [])          

            # detects whether the point needs to be rounded
            if point.r == 0:
                # No rounding
                coordP=np.array([point.x, point.y, point.z])

                if index > 0 and np.linalg.norm(lastCoordo - coordP) <= 1e-12:
                    p0 = lastPoint
                else:
                    p0 = factory.addPoint(point.x, point.y, point.z, meshSize)

                    if point.isOpen:
                        openPoints.append(p0)

                dict_point_pointsGmsh[point] = [p0]                

            else:
                # With rounding

                # The current / active point is P0
                # The next point is P2
                # The point before is point P1        

                # Point / Coint in which to create the fillet
                P0 = point.coordo

                # Recovers next point
                if index+1 == Npoints:
                    index_p1 = index - 1
                    index_p2 = 0
                elif index == 0:
                    index_p1 = -1
                    index_p2 = index + 1
                else:
                    index_p1 = index - 1
                    index_p2 = index + 1

                # Detect the point before P1 and the point P2
                P1 = points[index_p1].coordo
                P2 = points[index_p2].coordo

                A, B, C = Points_Rayon(P0, P1, P2, point.r)                

                if index > 0 and np.linalg.norm(lastCoordo - A) <= 1e-12:
                    # if the coordinate is identical, the point is not recreated
                    pA = lastPoint
                else:
                    pA = factory.addPoint(A[0], A[1], A[2], meshSize) # point of intersection between i and the circle

                    if point.isOpen:
                        openPoints.append(pA)
                    
                pC = factory.addPoint(C[0], C[1], C[2], meshSize) # circle center
                
                if index > 0 and (np.linalg.norm(B - firstCoordo) <= 1e-12):
                    pB = firstPoint
                else:
                    pB = factory.addPoint(B[0], B[1], B[2], meshSize) # point of intersection between j and the circle

                    if point.isOpen:
                        openPoints.append(pB)

                dict_point_pointsGmsh[point] = [pA, pC, pB]

            if index == 0:
                self.__factory.synchronize()
                firstPoint = dict_point_pointsGmsh[point][0]
                firstCoordo = gmsh.model.getValue(0, firstPoint, [])

            
        lines = []        
        self.__factory.synchronize()
        for index, point in enumerate(points):
            # For each point we'll create the loop associated with the point and we'll create a line with the next point.
            # For example, if the point has a radius, you'll first need to build the arc of the circle.
            # Then we need to connect the lastGmsh point to the firstGmsh point of the next node.

            # gmsh points created
            gmshPoints = dict_point_pointsGmsh[point]

            # If the corner is rounded, it is necessary to create the circular arc
            if point.r != 0:
                lines.append(factory.addCircleArc(gmshPoints[0], gmshPoints[1], gmshPoints[2]))
                # Here we remove the point from the center of the circle VERY IMPORTANT otherwise the point remains at the center of the circle.
                factory.remove([(0,gmshPoints[1])], False)
                
            # Retrieves the index of the next node
            if index+1 == Npoints:
                # If we are on the last node, we will close the loop by recovering the first point.
                indexAfter = 0
            else:
                indexAfter = index + 1

            # Gets the next gmsh point to create the line between the points
            gmshPointAfter = dict_point_pointsGmsh[points[indexAfter]][0]
            
            if gmshPoints[-1] != gmshPointAfter:
                c1 = gmsh.model.getValue(0, gmshPoints[-1], [])
                c2 = gmsh.model.getValue(0, gmshPointAfter, [])
                if np.linalg.norm(c1-c2) >= 1e-12:
                    # The link is not created if the gmsh points are identical and have the same coordinates.
                    lines.append(factory.addLine(gmshPoints[-1], gmshPointAfter))

        # Create a closed loop connecting the lines for the surface
        loop = factory.addCurveLoop(lines)

        return loop, lines, openPoints
    
    def __Loop_From_Contour(self, contour: Contour) -> tuple[int, list[int], list[int]]:
        """Create a loop associated with a list of 1D objects.
        return loop, openLines, openPoints
        """

        factory = self.__factory

        lines = []        

        nGeom = len(contour.geoms)

        openPoints = []
        openLines = []

        for i, geom in enumerate(contour.geoms):

            assert isinstance(geom, (Line, CircleArc)), "Must be a line or a CircleArc"

            if i == 0:
                p0 = factory.addPoint(geom.pt1.x, geom.pt1.y, geom.pt1.z, geom.meshSize)
                if geom.pt1.isOpen: openPoints.append(p0)
                p1 = factory.addPoint(geom.pt2.x, geom.pt2.y, geom.pt2.z, geom.meshSize)
                if geom.pt2.isOpen: openPoints.append(p1)
            elif i > 0 and i+1 < nGeom:
                p0 = p1
                p1 = factory.addPoint(geom.pt2.x, geom.pt2.y, geom.pt2.z, geom.meshSize)
                if geom.pt2.isOpen: openPoints.append(p1)
            else:
                p0 = p1
                p1 = firstPoint           

            if isinstance(geom, Line):

                line = factory.addLine(p0, p1)

                if geom.isOpen:
                    openLines.append(line)

                lines.append(line)

            elif isinstance(geom, CircleArc):                

                pC =  factory.addPoint(geom.center.x, geom.center.y, geom.center.z, geom.meshSize)

                p3 = factory.addPoint(geom.pt3.x, geom.pt3.y, geom.pt3.z, geom.meshSize)

                if geom.pt3.isOpen: openPoints.append(p3)

                line1 = factory.addCircleArc(p0, pC, p3)
                line2 = factory.addCircleArc(p3, pC, p1)

                lines.extend([line1, line2])

                if geom.isOpen:
                    openLines.extend([line1, line2])

                factory.synchronize()
                factory.remove([(0,pC)], False)

            if i == 0:
                firstPoint = p0

        loop = factory.addCurveLoop(lines)

        return loop, openLines, openPoints

    def __Loop_From_Circle(self, circle: Circle) -> int:
        """Creation of a loop associated with a circle.
        return loop
        """

        factory = self.__factory

        center = circle.center
        rayon = circle.diam/2

        # Circle points                
        p0 = factory.addPoint(center.x, center.y, center.z, circle.meshSize) # center
        p1 = factory.addPoint(center.x-rayon, center.y, center.z, circle.meshSize)
        p2 = factory.addPoint(center.x, center.y-rayon, center.z, circle.meshSize)
        p3 = factory.addPoint(center.x+rayon, center.y, center.z, circle.meshSize)
        p4 = factory.addPoint(center.x, center.y+rayon, center.z, circle.meshSize)        

        # Circle arcs
        l1 = factory.addCircleArc(p1, p0, p2)
        l2 = factory.addCircleArc(p2, p0, p3)
        l3 = factory.addCircleArc(p3, p0, p4)
        l4 = factory.addCircleArc(p4, p0, p1)

        # Here we remove the point from the center of the circle VERY IMPORTANT otherwise the point remains at the center of the circle.
        factory.remove([(0,p0)], False)
        
        loop = factory.addCurveLoop([l1,l2,l3,l4])

        return loop

    def __Loop_From_Domain(self, domain: Domain) -> tuple[int, int]:
        """Create a loop associated with a domain.
        return loop
        """
        pt1 = domain.pt1
        pt2 = domain.pt2

        p1 = Point(x=pt1.x, y=pt1.y, z=pt1.z)
        p2 = Point(x=pt2.x, y=pt1.y, z=pt1.z)
        p3 = Point(x=pt2.x, y=pt2.y, z=pt2.z)
        p4 = Point(x=pt1.x, y=pt2.y, z=pt2.z)

        loop = self.__Loop_From_Points([p1, p2, p3, p4], domain.meshSize)[0]
        
        return loop

    def __Surface_From_Loops(self, loops: list[int]) -> tuple[int, int]:
        """Create a surface associated with a loop.
        return surface
        """

        surface = self.__factory.addPlaneSurface(loops)

        return surface
    
    def __Add_PhysicalPoint(self, point: int) -> int:
        """Adds the point to the physical group."""
        pgPoint = gmsh.model.addPhysicalGroup(0, [point])
        return pgPoint
    
    def __Add_PhysicalLine(self, ligne: int) -> int:
        """Adds the line to the physical group."""
        pgLine = gmsh.model.addPhysicalGroup(1, [ligne])
        return pgLine
    
    def __Add_PhysicalSurface(self, surface: int) -> int:
        """Adds closed or open surface to physical groups."""
        pgSurf = gmsh.model.addPhysicalGroup(2, [surface])
        return pgSurf    
    
    def __Add_PhysicalVolume(self, volume: int) -> int:
        """Adds closed or open volume to physical groups."""
        pgVol = gmsh.model.addPhysicalGroup(3, [volume])
        return pgVol

    def __Add_PhysicalGroup(self, dim: int, tag: int) -> None:
        """Adds a physical group based on its dimension."""
        if dim == 0:
            self.__Add_PhysicalPoint(tag)
        elif dim == 1:
            self.__Add_PhysicalLine(tag)
        elif dim == 2:
            self.__Add_PhysicalSurface(tag)
        elif dim == 3:
            self.__Add_PhysicalVolume(tag)

    def __Set_PhysicalGroups(self, buildPoint=True, buildLine=True, buildSurface=True, buildVolume=True) -> None:
        """Create physical groups based on model entities."""
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

    def __Extrusion(self, surfaces: list, extrude=[0,0,1], elemType=ElemType.HEXA8, nCouches=1):
        """Function that extrudes multiple surfaces

        Parameters
        ----------
        surfaces : list[int]
            list of surfaces
        extrude : list, optional
            extrusion directions and values, by default [0,0,1]
        elemType : str, optional
            element type used, by default "HEXA8        
        nLayers: int, optional
            number of layers in extrusion, by default 1
        """
        
        factory = self.__factory

        extruEntities = []

        if "TETRA" in elemType:
            numElements = []
            combine = False
        else:
            numElements = [nCouches]
            combine = True

        for surf in surfaces:

            if elemType in [ElemType.HEXA8, ElemType.HEXA20]:
                # https://onelab.info/pipermail/gmsh/2010/005359.html

                factory.synchronize()
                gmsh.model.mesh.setRecombine(2, surf)
            
            # Create new elements for extrusion
            extru = factory.extrude([(2, surf)], extrude[0], extrude[1], extrude[2], recombine=combine, numElements=numElements)

            extruEntities.extend(extru)

        return extruEntities

    # TODO generate multiple meshes by disabling initGmsh and using multiple functions?
    # set up a list of surfaces?

    def __Set_BackgroundMesh(self, refineGeom: Geom, tailleOut: float):
        """Sets a background mesh

        Parameters
        ----------
        refineGeom : Geom
            Geometric object for background mesh
        tailleOut : float
            size of elements outside the domain
        """

        # Example extracted from t10.py in the gmsh tutorials

        # See also t11.py to make a line

        if isinstance(refineGeom, Domain):

            assert not refineGeom.meshSize == 0, "Domain mesh size is required."

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

            pt = refineGeom.center

            # point = self.__factory.addPoint(pt.x, pt.y, pt.z, refineGeom.meshSize)

            # field_Distance = gmsh.model.mesh.field.add("Distance")
            # gmsh.model.mesh.field.setNumbers(field_Distance, "PointsList", [point])            

            # minField = gmsh.model.mesh.field.add("Min")
            # gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [field_Distance])

            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field, "F", f"{pt.x} + {pt.y}")

            minField = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [field])

            # self.__factory.remove([(0,point)])

            # loopCercle = self.__Loop_From_Circle(refineGeom)

            # field_Distance = gmsh.model.mesh.field.add("Distance")
            # gmsh.model.mesh.field.setNumbers(field_Distance, "PointsList", [loopCercle])
            
            # field_Thershold = gmsh.model.mesh.field.add("Threshold")
            # gmsh.model.mesh.field.setNumber(field_Thershold, "InField", field_Distance)            
            # gmsh.model.mesh.field.setNumber(field_Thershold, "SizeMin", refineGeom.meshSize)
            # gmsh.model.mesh.field.setNumber(field_Thershold, "SizeMax", tailleOut)
            # gmsh.model.mesh.field.setNumber(field_Thershold, "DistMin", 0.15)
            # gmsh.model.mesh.field.setNumber(field_Thershold, "DistMax", 0.5)

            # minField = gmsh.model.mesh.field.add("Min")
            # gmsh.model.mesh.field.setNumbers(minField, "FieldsList", [field_Thershold])

            gmsh.model.mesh.field.setAsBackgroundMesh(minField)            

        elif isinstance(refineGeom, str):

            if not os.path.exists(refineGeom) :
                print(Fore.RED + "The .pos file does not exist." + Fore.WHITE)
                return

            if ".pos" not in refineGeom:
                print(Fore.RED + "Must provide a .pos file" + Fore.WHITE)
                return

            gmsh.merge(refineGeom)

            # Add the post-processing view as a new size field:
            minField = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(minField, "ViewIndex", 0)

            # Apply the view as the current background mesh size field:
            gmsh.model.mesh.field.setAsBackgroundMesh(minField)

            # # In order to compute the mesh sizes from the background mesh only, and
            # # disregard any other size constraints, one can set:
            # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0) 

    def Mesh_Import_mesh(self, mesh: str, setPhysicalGroups=False, coef=1.0):
        """Importing an .msh file. Must be an gmsh file.

        Parameters
        ----------
        mesh : str
            file (.msh) that gmsh will load to create the mesh        
        setPhysicalGroups : bool, optional
            retrieve entities to create physical groups of elements, by default False
        coef : float, optional
            coef applied to node coordinates, by default 1.0

        Returns
        -------
        Mesh
            Built mesh
        """

        self.__initGmsh('occ')

        tic = Tic()

        gmsh.open(mesh)
        
        tic.Tac("Mesh","Mesh import", self.__verbosity)

        if setPhysicalGroups:
            self.__Set_PhysicalGroups()

        return self.__Construct_Mesh(coef)

    def Mesh_Import_part(self, fichier: str, meshSize=0.0, elemType=ElemType.TETRA4, refineGeom=None, folder=""):
        """Build mesh from imported file (.stp or .igs)

        Parameters
        ----------
        file : str
            file (.stp, .igs) that gmsh will load to create the mesh.\n
            Note that for igs files, entities cannot be recovered.
        meshSize : float, optional
            mesh size, by default 0.0
        elemType : str, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10"]
        refineGeom : Geom, optional
            second domain for mesh concentration, by default None
        folder : str, optional
            mesh save folder mesh.msh, by default ""

        Returns
        -------
        Mesh
            Built mesh
        """
        # When importing a part, only TETRA4 or TETRA10 can be used.        
        # Allow other meshes -> this seems impossible - you have to create the mesh using gmsh to control the type of element.

        self.__initGmsh('occ') # Ici ne fonctionne qu'avec occ !! ne pas changer

        assert meshSize >= 0.0, "Must be greater than or equal to 0."
        self.__CheckType(3, elemType)
        
        tic = Tic()

        factory = self.__factory

        if '.stp' in fichier or '.igs' in fichier:
            factory.importShapes(fichier)
        else:
            print("Must be a .stp or .igs file")

        self.__Set_BackgroundMesh(refineGeom, meshSize)

        self.__Set_PhysicalGroups(buildPoint=False, buildLine=True, buildSurface=True, buildVolume=False)

        gmsh.option.setNumber("Mesh.MeshSizeMin", meshSize)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)

        tic.Tac("Mesh","File import", self.__verbosity)

        self.__Meshing(3, elemType, folder=folder)

        return self.__Construct_Mesh()

    def __PhysicalGroups_cracks(self, cracks: list, entities: list[tuple]) -> tuple[int, int, int, int]:
        """Creation of physical groups associated with cracks.\n
        return crackLines, crackSurfaces, openPoints, openLines
        """

        if len(cracks) == 0:
            return None, None, None, None
        
        # lists containing open entities
        openPoints = []
        openLines = []
        openSurfaces = []        

        entities0D = []
        entities1D = []
        entities2D = []

        for crack in cracks:
            if isinstance(crack, Line):
                # Creating points
                pt1 = crack.pt1
                p1 = self.__factory.addPoint(pt1.x, pt1.y, pt1.z, crack.meshSize)
                pt2 = crack.pt2
                p2 = self.__factory.addPoint(pt2.x, pt2.y, pt2.z, crack.meshSize)

                # Line creation
                line = self.__factory.addLine(p1, p2)
                entities1D.append(line)
                if crack.isOpen:
                    openLines.append(line)

                if pt1.isOpen:
                    entities0D.append(p1)
                    openPoints.append(p1)

                if pt2.isOpen:
                    entities0D.append(p2)
                    openPoints.append(p2)

            elif isinstance(crack, PointsList):

                loop, lines, openPts = self.__Loop_From_Points(crack.points, crack.meshSize)
                
                entities0D.extend(openPts)
                openPoints.extend(openPts)
                entities1D.extend(lines)

                surface = self.__Surface_From_Loops([loop])                
                entities2D.append(surface)

                if crack.isCreux:
                    openLines.extend(lines)
                    openSurfaces.append(surface)

            elif isinstance(crack, Contour):

                loop, openLns, openPts = self.__Loop_From_Contour(crack)
                
                entities0D.extend(openPts)
                openPoints.extend(openPts)
                entities1D.extend(openLns)

                surface = self.__Surface_From_Loops([loop])                
                entities2D.append(surface)

                if crack.isCreux:
                    openLines.extend(openLns)
                    openSurfaces.append(surface)
                
            else:
                # Loop recovery
                hollowLoops, filledLoops = self.__Get_hollow_And_filled_Loops([crack])
                loops = []; loops.extend(hollowLoops); loops.extend(filledLoops)
                
                # Surface construction
                for loop in loops:
                    surface = self.__Surface_From_Loops([loop])
                    entities2D.append(surface)

                    if crack.isCreux:
                        openSurfaces.append(surface)

        newEntities = [(0, point) for point in entities0D]
        newEntities.extend([(1, line) for line in entities1D])
        newEntities.extend([(2, surf) for surf in entities2D])

        o, m = gmsh.model.occ.fragment(entities, newEntities)
        self.__factory.synchronize()

        crackLines = gmsh.model.addPhysicalGroup(1, openLines) if len(openLines) > 0 else None
        crackSurfaces = gmsh.model.addPhysicalGroup(2, openSurfaces) if len(openSurfaces) > 0 else None

        openPoints = gmsh.model.addPhysicalGroup(0, openPoints) if len(openPoints) > 0 else None
        openLines = gmsh.model.addPhysicalGroup(1, openLines) if len(openLines) > 0 else None

        return crackLines, crackSurfaces, openPoints, openLines

    def Mesh_Beams(self, listPoutres: list[Beam_Elas_Isot], elemType=ElemType.SEG2, folder=""):
        """Construction of a segment mesh

        Parameters
        ----------
        listBeam : list[Beam]
            list of Beams
        elemType : str, optional
            element type, by default "SEG2" ["SEG2", "SEG3", "SEG4"]
        folder : str, optional
            mesh save folder mesh.msh, by default ""

        Returns
        -------
        Mesh
            construct mesh
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
        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Beam mesh construction", self.__verbosity)

        self.__Meshing(1, elemType, surfaces=[], folder=folder)

        mesh = self.__Construct_Mesh()

        def FuncAddTags(poutre: Beam_Elas_Isot):
            nodes = mesh.Nodes_Line(poutre.line)
            for grp in mesh.Get_list_groupElem():
                grp.Set_Nodes_Tag(nodes, poutre.name)
                grp.Set_Elements_Tag(nodes, poutre.name)

        [FuncAddTags(poutre) for poutre in listPoutres]

        return mesh

    def __Get_hollow_And_filled_Loops(self, inclusions: list) -> tuple[list, list]:
        """Creation of hollow and filled loops

        Parameters
        ----------
        inclusions : list
            List of geometric objects contained in the domain

        Returns
        -------
        tuple[list, list]
            all loops created, followed by full (non-hollow) loops
        """
        hollowLoops = []
        filledLoops = []
        for objetGeom in inclusions:
            
            loop = self.__Loop_From_Geom(objetGeom)

            if objetGeom.isCreux:
                hollowLoops.append(loop)
            else:                
                filledLoops.append(loop)

        return hollowLoops, filledLoops

    # TODO make the revolution around an axis

    def Mesh_2D(self, contour: Geom, inclusions=[], elemType=ElemType.TRI3, cracks=[], isOrganised=False, refineGeom=None, folder="", returnSurfaces=False):
        """Build the 2D mesh by creating a surface from a Geom object

        Parameters
        ----------
        contour : Geom
            geometry that builds the contour
        inclusions : list[Domain, Circle, PointsList, Contour], optional
            list of hollow and non-hollow objects inside the domain 
        elemType : str, optional
            element type, by default "TRI3" ["TRI3", "TRI6", "TRI10", "QUAD4", "QUAD8"]
        cracks : list[Line | PointsList | Countour]
            list of object used to create cracks
        isOrganised : bool, optional
            mesh is organized, by default False
        refineGeom : Geom, optional
            second domain for mesh concentration, by default None
        folder : str, optional
            mesh save folder mesh.msh, by default ""
        returnSurfaces : bool, optional
            returns surface, by default False

        Returns
        -------
        Mesh
            2D mesh
        """
        Point
        if isOrganised and isinstance(contour, Domain) and len(inclusions)==0 and len(cracks)==0:
            self.__initGmsh('geo')
        else:
            self.__initGmsh('occ')
            isOrganised = False
        self.__CheckType(2, elemType)

        tic = Tic()

        factory = self.__factory
        
        meshSize = contour.meshSize

        # Create contour surface
        loopContour = self.__Loop_From_Geom(contour)

        # Creation of all loops associated with objects within the domain
        hollowLoops, filledLoops = self.__Get_hollow_And_filled_Loops(inclusions)
        
        listeLoop = [loopContour] # domain surface
        listeLoop.extend(hollowLoops) # Hollow contours are added
        listeLoop.extend(filledLoops) # Filled contours are added

        surfaceDomain = self.__Surface_From_Loops(listeLoop)

        # For each filled Geom object, it is necessary to create a surface
        surfacesPleines = [surfaceDomain]
        [surfacesPleines.append(factory.addPlaneSurface([loop])) for loop in filledLoops]
        
        self.__factory.synchronize()
        if returnSurfaces: return surfacesPleines
        
        # Recovers 2D entities
        entities2D = gmsh.model.getEntities(2)

        # Crack creation
        crackLines, crackSurfaces, openPoints, openLines = self.__PhysicalGroups_cracks(cracks, entities2D)            

        self.__Set_BackgroundMesh(refineGeom, meshSize)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self.__Meshing(2, elemType, surfacesPleines, isOrganised, crackLines=crackLines, openPoints=openPoints, folder=folder)

        return self.__Construct_Mesh()

    def Mesh_3D(self, contour: Geom, inclusions=[], extrude=[0,0,1], nCouches=1, elemType=ElemType.TETRA4, cracks=[], refineGeom=None, folder=""):
        """Build the 3D mesh by creating a surface from a Geom object

        Parameters
        ----------
        contour : Geom
            geometry that builds the contour
        inclusions : list[Domain, Circle, PointsList, Contour], optional
            list of hollow and non-hollow objects inside the domain
        extrude : list, optional
            extrusion, by default [0,0,1]
        nLayers : int, optional
            number of layers in extrusion, by default 1
        elemType : str, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"]
        cracks : list[Line | PointsList | Countour]
            list of object used to create cracks
        refineGeom : Geom, optional
            second domain for mesh concentration, by default None
        folder : str, optional
            mesh.msh backup folder, by default ""

        Returns
        -------
        Mesh
            3D mesh
        """
        
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # the starting 2D mesh is irrelevant
        surfaces = self.Mesh_2D(contour, inclusions, ElemType.TRI3, [], False, refineGeom, returnSurfaces=True)

        self.__Extrusion(surfaces=surfaces, extrude=extrude, elemType=elemType, nCouches=nCouches)        

        # Recovers 3D entity
        self.__factory.synchronize()
        entities3D = gmsh.model.getEntities(3)

        # Crack creation
        crackLines, crackSurfaces, openPoints, openLines = self.__PhysicalGroups_cracks(cracks, entities3D)

        self.__Set_BackgroundMesh(refineGeom, contour.meshSize)

        self.__Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)        

        self.__Meshing(3, elemType, folder=folder, crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)
        
        return self.__Construct_Mesh()
    
    def Create_posFile(self, coordo: np.ndarray, values: np.ndarray, folder: str, filename="data") -> str:
        """Creation of a .pos file that can be used to refine a mesh in a zone.

        Parameters
        ----------
        coordo : np.ndarray
            coordinates of values
        values : np.ndarray
            values at coordinates
        folder : str
            backup file
        filename : str, optional
            pos file name, by default "data".

        Returns
        -------
        str
            Returns path to .pos file
        """

        assert isinstance(coordo, np.ndarray), "Must be a numpy array"
        assert coordo.shape[1] == 3, "Must be of dimension (n, 3)"

        assert values.shape[0] == coordo.shape[0], "values and coordo are the wrong size"

        data = np.append(coordo, values.reshape(-1, 1), axis=1)

        self.__initGmsh("occ")

        view = gmsh.view.add("scalar points")

        gmsh.view.addListData(view, "SP", coordo.shape[0], data.reshape(-1))

        path = Folder.New_File(f"{filename}.pos", folder)

        gmsh.view.write(view, path)

        return path

    @staticmethod
    def __Set_order(elemType: str):
        """Set mesh order"""
        if elemType in ["TRI3","QUAD4"]:
            gmsh.model.mesh.set_order(1)
        elif elemType in ["SEG3", "TRI6", "QUAD8", "TETRA10", "HEXA20", "PRISM15"]:
            if elemType in ["QUAD8", "HEXA20", "PRISM15"]:
                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
            gmsh.model.mesh.set_order(2)
        elif elemType in ["SEG4", "TRI10"]:
            gmsh.model.mesh.set_order(3)
        elif elemType in ["SEG5", "TRI15"]:
            gmsh.model.mesh.set_order(4)

    def __Meshing(self, dim: int, elemType: str, surfaces=[], isOrganised=False, crackLines=None, crackSurfaces=None, openPoints=None, openLines=None, folder=""):
        """Construction of gmsh mesh from geometry that has been built or imported.

        Parameters
        ----------
        dim : int
            mesh size
        elemType : str
            element type
        surfaces : list[int], optional
            list of surfaces to be meshed, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        crackLines : int, optional
            PhysicalGroup that groups all cracks on lines, by default None
        crackSurfaces : int, optional
            physical grouping of all cracks on lines, by default None
        openPoints: int, optional
            physical group of points that can open, by default None
        openLines : int, optional
            physical group of lines that can open, by default None
        folder : str, optional
            mesh save folder mesh.msh, by default ""
        """

        factory = self.__factory

        if factory == gmsh.model.occ:
            isOrganised = False
            factory = cast(gmsh.model.occ, factory)
        elif factory == gmsh.model.geo:
            factory = cast(gmsh.model.geo, factory)
        else:
            raise Exception("Unknow factory")

        tic = Tic()
        if dim == 1:
            self.__factory.synchronize()
            gmsh.model.mesh.generate(1)
            Interface_Gmsh.__Set_order(elemType)
        elif dim == 2:

            assert isinstance(surfaces, list)
            
            for surf in surfaces:
                
                if isOrganised:
                    # Only works for a simple surface (no holes or cracks) and when the model is built with geo and not occ!
                    # It's not possible to create a setTransfiniteSurface with occ
                    # If you have to use occ, it's not possible to create an organized mesh.
                                        
                    gmsh.model.geo.synchronize()
                    points = np.array(gmsh.model.getEntities(0))[:,1]
                    if points.shape[0] <= 4:
                        #It is imperative to give the contour points when more than 3 or 4 points are used.
                        gmsh.model.geo.mesh.setTransfiniteSurface(surf, cornerTags=points)

                # Synchronisation
                self.__factory.synchronize()

                if elemType in [ElemType.QUAD4,ElemType.QUAD8]:
                    try:
                        gmsh.model.mesh.setRecombine(2, surf)
                    except Exception:
                        # Recover surface
                        entities = gmsh.model.getEntities()
                        surf = entities[-1][-1]
                        gmsh.model.mesh.setRecombine(2, surf)
                
            # Generates mesh
            gmsh.model.mesh.generate(2)
            
            Interface_Gmsh.__Set_order(elemType)
        
        elif dim == 3:
            self.__factory.synchronize()

            entities = gmsh.model.getEntities(2)
            surfaces = np.array(entities)[:,1]

            # for surf in surfaces:
                    
            #     if isOrganised:

            #         factory = cast(gmsh.model.geo, factory)

            #         factory.synchronize()

            #         points = np.array(gmsh.model.getEntities(0))[:,1]
            #         factory.mesh.setTransfiniteSurface(surf)
            #         # if points.shape[0] <= 4:
            #         #     factory.mesh.setTransfiniteSurface(surf, cornerTags=points)
                
            # factory.synchronize()
            # gmsh.model.mesh.setRecombine(3, 1)            
            
            gmsh.model.mesh.generate(3)

            Interface_Gmsh.__Set_order(elemType)

        # A single physical group is required for lines and points
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
        
        # Open gmsh interface if necessary
        if '-nopopup' not in sys.argv and self.__affichageGmsh:
            gmsh.fltk.run()
        
        tic.Tac("Mesh","Meshing", self.__verbosity)

        if folder != "":
            # gmsh.write(Dossier.Join([folder, "model.geo"])) # Il semblerait que ça marche pas c'est pas grave
            self.__factory.synchronize()
            # gmsh.model.geo.synchronize()
            # gmsh.model.occ.synchronize()
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            gmsh.write(Folder.Join([folder, "mesh.msh"]))
            tic.Tac("Mesh","Saving .msh", self.__verbosity)

    def __Construct_Mesh(self, coef=1):
        """Recovering the built mesh"""

        # TODO make this class accessible?

        # Old method was boggling
        # The bugs have been fixed because I didn't properly organize the nodes when I created them
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1926
        
        tic = Tic()

        dict_groupElem = {}
        elementTypes = gmsh.model.mesh.getElementTypes()
        nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()

        nodes = np.array(nodes-1) #node number
        Nn = nodes.shape[0] #Number of nodes

        # Organize nodes from smallest to largest
        sortedIndices = np.argsort(nodes)
        sortedNodes = nodes[sortedIndices]

        # Here we will detect jumps in node numbering
        # Example 0 1 2 3 4 5 6 8 Here we will detect the jump between 6 and 8.
        ecart = sortedNodes - np.arange(Nn)

        # Nodes to be changed are those where the deviation is > 0
        noeudsAChanger = np.where(ecart>0)[0]

        # Builds a matrix in which we will store in the first column
        # the old values and in the 2nd the new ones
        changes = np.zeros((noeudsAChanger.shape[0],2), dtype=int)
        changes[:,0] = sortedNodes[noeudsAChanger]
        changes[:,1] = noeudsAChanger

        # Apply the change
        nodes = np.array(sortedNodes - ecart, dtype=int)

        # The coordinate matrix of all nodes used in the mesh is constructed
        # Nodes used in 1D 2D and 3D
        coord = coord.reshape(-1,3)
        coordo = coord[sortedIndices]

        # Apply coef to scale coordinates
        coordo = coordo * coef
        
        # Builds physical groups
        physicalGroups = gmsh.model.getPhysicalGroups()
        pgArray = np.array(physicalGroups)
        # To be optimized
        physicalGroupsPoint = []; namePoint = []
        physicalGroupsLine = []; nameLine = []
        physicalGroupsSurf = []; nameSurf = []
        physicalGroupsVol = []; nameVol = []

        nbPhysicalGroup = 0

        def __name(dim: int, n: int) -> str:
            # Builds entity name
            name = f"{Interface_Gmsh.__dict_name_dim[dim]}{n}"
            return name

        for dim in range(pgArray[:,0].max()+1):
            # For each dimension available in the physical groups.
            
            # We retrieve the entities of the dimension
            indexDim = np.where(pgArray[:,0] == dim)[0]
            listTupleDim = tuple(map(tuple, pgArray[indexDim]))
            nbEnti = indexDim.size

            # Depending on the dimension of the entities, we'll give them names
            # Then we'll add the entity tuples (dim, tag) to the PhysicalGroup list associated with the dimension.
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

        # Check that everything has been added correctly
        assert len(physicalGroups) == nbPhysicalGroup

        # Builds element groups
        dimAjoute = []
        meshDim = pgArray[:,0].max()

        for gmshId in elementTypes:
            # For each element type
                                        
            # Retrieves element numbers and connection matrix
            elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
            elementTags = np.array(elementTags-1, dtype=int)
            nodeTags = np.array(nodeTags-1, dtype=int)

            # Elements
            Ne = elementTags.shape[0] #number of elements
            elementsID = elementTags            
            nPe = GroupElem_Factory.Get_ElemInFos(gmshId)[1] # nodes per element
            
            # Builds connect and changes the necessary nodes
            connect = nodeTags.reshape(Ne, nPe)
            def TriConnect(old, new):
                connect[np.where(connect==old)] = new
            [TriConnect(old, new) for old, new in zip(changes[:,0], changes[:,1])]
            # A tester avec l, c = np.where(connect==changes[:,0])
            
            # Nodes
            nodes = np.unique(nodeTags)

            # Check that max node numbers can be reached in coordo
            Nmax = nodes.max()
            assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"

            # Element group creation
            groupElem = GroupElem_Factory.Create_GroupElem(gmshId, connect, coordo, nodes)
            
            # We add the element group to the dictionary containing all groups
            dict_groupElem[groupElem.elemType] = groupElem
            
            # Check that the mesh does not have a group of elements of this dimension
            if groupElem.dim in dimAjoute and groupElem.dim == meshDim:
                recoElement = 'Triangular' if meshDim == 2 else 'Tetrahedron'
                raise Exception(f"Importing the mesh is impossible because several {meshDim}D elements have been detected. Try out {recoElement} elements.")
                # TODO make it work ?
                # Can be complicated especially in the creation of elemental matrices and assembly
                # Not impossible but not trivial
                # Relaunch the procedure if it doesn't work?
            dimAjoute.append(groupElem.dim)

            # Here we'll retrieve the nodes and elements belonging to a group
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

            # For each physical group, I'll retrieve the nodes
            # and associate the tags
            i = -1

            for dim, tag in listPhysicalGroups:
                i += 1

                name = listName[i]

                nodeTags, coord = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
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
        
        tic.Tac("Mesh","Mesh construction", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construction_2D(L=10, h=10, taille=3) -> list[Mesh]:
        """2D mesh generation"""

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
            assert np.abs(aireDomain-aire)/aireDomain <= 1e-6, "Incorrect surface"

        # For each type of 2D element
        for t, elemType in enumerate(GroupElem.get_Types2D()):

            print(elemType)

            mesh1 = interfaceGmsh.Mesh_2D(domain, elemType=elemType, isOrganised=False)
            testAire(mesh1.area)
            
            mesh2 = interfaceGmsh.Mesh_2D(domain, elemType=elemType, isOrganised=True)
            testAire(mesh2.area)

            mesh3 = interfaceGmsh.Mesh_2D(domain, [circle], elemType)
            # Here we don't check because there are too few elements to properly represent the hole

            mesh4 = interfaceGmsh.Mesh_2D(domain, [circleClose], elemType)
            testAire(mesh4.area)

            mesh5 = interfaceGmsh.Mesh_2D(domain, cracks=[line], elemType=elemType)
            testAire(mesh5.area)

            mesh6 = interfaceGmsh.Mesh_2D(domain, cracks=[lineOpen], elemType=elemType)
            testAire(mesh6.area)

            for m in [mesh1, mesh2, mesh3, mesh4, mesh5, mesh6]:
                list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def Construction_3D(L=130, h=13, b=13, taille=7.5, useImport3D=False):
        """3D mesh generation"""        

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=taille)
        circleCreux = Circle(Point(x=L/2, y=0,z=-b/2), h*0.7, meshSize=taille, isCreux=True)
        circle = Circle(Point(x=L/2, y=0 ,z=-b/2), h*0.7, meshSize=taille, isCreux=False)

        volume = L*h*b

        def testVolume(val):
            assert np.abs(volume-val)/volume <= 1e-6, "Incorrect volume"

        folder = Folder.Get_Path()
        cpefPath = Folder.Join([folder,"3Dmodels","CPEF.stp"])
        partPath = Folder.Join([folder,"3Dmodels","part.stp"])

        interfaceGmsh = Interface_Gmsh(verbosity=False, affichageGmsh=False)

        list_mesh3D = []
        # For each type of 3D element
        for t, elemType in enumerate(GroupElem.get_Types3D()):            
            
            if useImport3D and elemType in ["TETRA4","TETRA10"]:
                meshCpef = interfaceGmsh.Mesh_Import_part(cpefPath, meshSize=10, elemType=elemType)
                list_mesh3D.append(meshCpef)
                meshPart = interfaceGmsh.Mesh_Import_part(partPath, meshSize=taille, elemType=elemType)
                list_mesh3D.append(meshPart)

            mesh1 = interfaceGmsh.Mesh_3D(domain, [], [0,0,b], 3, elemType=elemType)
            list_mesh3D.append(mesh1)
            testVolume(mesh1.volume)                

            mesh2 = interfaceGmsh.Mesh_3D(domain, [circleCreux], [0,0,b], 3, elemType)
            list_mesh3D.append(mesh2)            

            mesh3 = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,b], 3, elemType)
            list_mesh3D.append(mesh3)
            testVolume(mesh3.volume)


        return list_mesh3D