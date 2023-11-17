"""Interface module with gmsh (https://gmsh.info/).
This module lets you manipulate Geom objects to create meshes."""

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
from Materials import _Beam_Model

class Interface_Gmsh:

    def __init__(self, openGmsh=False, gmshVerbosity=False, verbosity=False):
        """Building an interface that can interact with gmsh.

        Parameters
        ----------
        openGmsh : bool, optional
            display mesh built in gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh can write to terminal, by default False
        verbosity : bool, optional
            interfaceGmsh class can write construction summary to terminal, by default False
        """
    
        self.__openGmsh = openGmsh
        """gmsh can display the mesh"""
        self.__gmshVerbosity = gmshVerbosity
        """gmsh can write to the console"""
        self.__verbosity = verbosity
        """the interface can write to the console"""

        self._init_gmsh_factory()

        if gmshVerbosity:
            Display.Section("New interface with gmsh.")

    def __CheckType(self, dim: int, elemType: str):
        """Check that the element type is usable."""
        if dim == 1:
            assert elemType in GroupElem.get_Types1D(), f"Must be in {GroupElem.get_Types1D()}"
        if dim == 2:
            assert elemType in GroupElem.get_Types2D(), f"Must be in {GroupElem.get_Types2D()}"
        elif dim == 3:
            assert elemType in GroupElem.get_Types3D(), f"Must be in {GroupElem.get_Types3D()}"
    
    def _init_gmsh_factory(self, factory: str= 'occ'):
        """Initialize gmsh factory."""        
        if not gmsh.isInitialized():
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
        return self.__factory
        
    def _Loop_From_Geom(self, geom: Geom) -> int:
        """Creation of a loop based on the geometric object."""        

        if isinstance(geom, Circle):
            loop = self._Loop_From_Circle(geom)[0]
        elif isinstance(geom, Domain):                
            loop = self._Loop_From_Domain(geom)
        elif isinstance(geom, PointsList):                
            loop = self._Loop_From_Points(geom.points, geom.meshSize)[0]
        elif isinstance(geom, Contour):
            loop = self._Loop_From_Contour(geom)[0]
        else:
            raise Exception("Must be a circle, a domain, a list of points or a contour.")
        
        return loop

    def _Loop_From_Points(self, points: list[Point], meshSize: float) -> tuple[int, list[int], list[int]]:
        """Creation of a loop associated with the list of points.\n
        return loop, lines, openPoints
        """
        
        factory = self.__factory

        # We create all the points
        Npoints = len(points)

        # dictionary, which takes a Point object as key and contains the id list of gmsh points created
        dict_point_pointsGmsh = cast(dict[Point, list[int]],{})
        
        openPoints = []

        # create the points to make the loops
        # this loop create a dictionnary of gmsh points for each point in points
        for index, point in enumerate(points):
            # pi -> gmsh id of point i
            # Pi -> coordinates of point i
            if index > 0:
                # Retrieves the last gmsh point created
                prevPoint = points[index-1]
                factory.synchronize()
                lastPoint = dict_point_pointsGmsh[prevPoint][-1]
                # retrieves last point coordinates
                lastCoordo = gmsh.model.getValue(0, lastPoint, [])          

            # detects whether the point needs to be rounded
            if point.r == 0:
                # No rounding
                coordP=np.array([point.x, point.y, point.z])

                if index > 0 and np.linalg.norm(lastCoordo - coordP) <= 1e-12:
                    # check if the current point location
                    # if they have the same coordinates 
                    # we dont create the point
                    p0 = lastPoint
                else:
                    # we create a new point
                    p0 = factory.addPoint(point.x, point.y, point.z, meshSize)

                    if point.isOpen:
                        openPoints.append(p0)

                dict_point_pointsGmsh[point] = [p0]                

            else:
                # With rounding
                # Construct a radius in the corner

                # The current / active point is P0 (corner point)
                # The next point is P2 (next point in the loop)
                # The point before is point P1 (previous point in the loop)

                # Point / Coint in which to create the fillet
                P0 = point.coordo
                # Recovers points 1 and 2 indices in points
                if index+1 == Npoints:
                    # here we are on the last point of points
                    index_p1 = index - 1
                    index_p2 = 0
                elif index == 0:
                    # here we are on the first point of points
                    index_p1 = -1
                    index_p2 = index + 1
                else:
                    index_p1 = index - 1
                    index_p2 = index + 1

                # Get the points P1 & P2 coordinates
                P1 = points[index_p1].coordo
                P2 = points[index_p2].coordo
                
                # Get the points A, B and C to construct the circle arc
                A, B, C = Points_Rayon(P0, P1, P2, point.r)
                # A is the point on the line form by P0 P1
                # B is the point on the line form by P0 P2
                # C is the point used for the circular arc
                # C will be delete after the creation of the circle arc

                # A
                if index > 0 and np.linalg.norm(lastCoordo - A) <= 1e-12:
                    # if the coordinate is identical, the point is not recreated
                    pA = lastPoint
                else:
                    pA = factory.addPoint(A[0], A[1], A[2], meshSize)

                    if point.isOpen:
                        openPoints.append(pA)
                # C 
                pC = factory.addPoint(C[0], C[1], C[2], meshSize) # circle center
                
                # B
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

        factory.synchronize()

        return loop, lines, openPoints
    
    def _Loop_From_Contour(self, contour: Contour) -> tuple[int, list[int], list[int]]:
        """Create a loop associated with a list of 1D objects (Line, CircleArc).\n
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

        factory.synchronize()

        return loop, openLines, openPoints

    def _Loop_From_Circle(self, circle: Circle) -> tuple[int, list[int], list[int]]:
        """Creation of a loop associated with a circle.\n
        return loop, lines, points
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
        points = [p1, p2, p3, p4]

        # Circle arcs
        l1 = factory.addCircleArc(p1, p0, p2)
        l2 = factory.addCircleArc(p2, p0, p3)
        l3 = factory.addCircleArc(p3, p0, p4)
        l4 = factory.addCircleArc(p4, p0, p1)
        lines = [l1, l2, l3, l4]

        # Here we remove the point from the center of the circle
        # VERY IMPORTANT otherwise the point remains at the center of the circle.
        # We don't want any points not attached to the mesh
        factory.remove([(0,p0)], False)
        
        loop = factory.addCurveLoop([l1,l2,l3,l4])

        factory.synchronize()

        return loop, lines, points

    def _Loop_From_Domain(self, domain: Domain) -> int:
        """Create a loop associated with a domain.\n
        return loop
        """
        pt1 = domain.pt1
        pt2 = domain.pt2

        p1 = Point(x=pt1.x, y=pt1.y, z=pt1.z)
        p2 = Point(x=pt2.x, y=pt1.y, z=pt1.z)
        p3 = Point(x=pt2.x, y=pt2.y, z=pt2.z)
        p4 = Point(x=pt1.x, y=pt2.y, z=pt2.z)

        loop = self._Loop_From_Points([p1, p2, p3, p4], domain.meshSize)[0]

        self.__factory.synchronize()
        
        return loop

    def _Surface_From_Loops(self, loops: list[int]) -> tuple[int, int]:
        """Create a surface associated with a loop.\n
        return surface
        """
        
        surface = self.__factory.addPlaneSurface(loops)

        self.__factory.synchronize()

        return surface
    
    def _Surfaces(self, contour: Geom, inclusions: list[Geom]) -> list[int]:
        """Create surfaces.\n
        return filled surfaces

        Parameters
        ----------
        contour : Geom
            the object that creates the surface area
        inclusions : list[Geom]
            objects that create hollow or filled surfaces in the first surface.\n
            CAUTION : all inclusions must be contained within the contour and do not cross
        """

        factory = self.__factory

        # Create contour surface
        loopContour = self._Loop_From_Geom(contour)

        # Creation of all loops associated with objects within the domain
        hollowLoops, filledLoops = self.__Get_hollow_And_filled_Loops(inclusions)
        
        listeLoop = [loopContour] # domain surface
        listeLoop.extend(hollowLoops) # Hollow contours are added
        listeLoop.extend(filledLoops) # Filled contours are added

        surfaceContour = self._Surface_From_Loops(listeLoop) # first filled surface

        # For each filled Geom object, it is necessary to create a surface
        surfaces = [surfaceContour]
        [surfaces.append(factory.addPlaneSurface([loop])) for loop in filledLoops]

        factory.synchronize()

        return surfaces   
    
    __dict_name_dim = {
        0 : "P",
        1 : "L",
        2 : "S",
        3 : "V"
    }

    def _Set_PhysicalGroups(self, setPoints=True, setLines=True, setSurfaces=True, setVolumes=True) -> None:
        """Create physical groups based on model entities."""
        self.__factory.synchronize()
        entities = np.array(gmsh.model.getEntities())
        ents = gmsh.model.getEntities()

        if entities.size == 0: return
        
        listDim = []
        if setPoints: listDim.append(0)            
        if setLines: listDim.append(1)            
        if setSurfaces: listDim.append(2)            
        if setVolumes: listDim.append(3)

        def _addPhysicalGroup(dim: int, tag: int, t:int) -> None:
            name = f"{self.__dict_name_dim[dim]}{t}"
            gmsh.model.addPhysicalGroup(dim, [tag], name=name)

        for dim in listDim:
            idx = entities[:,0]==dim
            tags = entities[idx, 1]
            [_addPhysicalGroup(dim, tag, t) for t, tag in enumerate(tags)]

    def _Extrude(self, surfaces: list[int], extrude=[0,0,1], elemType=ElemType.HEXA8, nLayers=1, isOrganised=False):
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
        isOrganised : bool, optional
            mesh is organized, by default False
        """
        
        factory = self.__factory

        extruEntities = []

        if "TETRA" in elemType:
            recombine = False
            numElements = [nLayers] if nLayers > 1 else []
        else:            
            recombine = True
            numElements = [nLayers]

        surfaces = [entity2D[1] for entity2D in gmsh.model.getEntities(2)]

        factory.synchronize()

        for surf in surfaces:

            if isOrganised:
                # only works if the surface is formed by 4 lines
                lines = gmsh.model.getBoundary([(2, surf)])
                if len(lines) == 4:
                    gmsh.model.mesh.setTransfiniteSurface(surf, cornerTags=[])
            
            if elemType in [ElemType.HEXA8, ElemType.HEXA20]:
                # https://onelab.info/pipermail/gmsh/2010/005359.html
                gmsh.model.mesh.setRecombine(2, surf)
            
            # Create new elements for extrusion
            extru = factory.extrude([(2, surf)], extrude[0], extrude[1], extrude[2], recombine=recombine, numElements=numElements)

            extruEntities.extend(extru)

        factory.synchronize()

        return extruEntities
    
    def _Revolve(self, surfaces: list[int], axis: Line, angle: float, elemType: ElemType, nLayers=360, isOrganised=False):
        """Function that revolves multiple surfaces.

        Parameters
        ----------
        surfaces : list[int]
            list of surfaces
        axis : Line
            revolution axis
        angle: float
            revolution angle
        elemType : str
            element type used
        nLayers: int, optional
            number of layers in extrusion, by default 360
        isOrganised : bool, optional
            mesh is organized, by default False
        """
        
        factory = self.__factory

        angleIs2PI = np.abs(angle) == 2 * np.pi

        if angleIs2PI:
            angle = angle / 2
            nLayers = nLayers // 2 if nLayers > 1 else 1

        revolEntities = []
        if "TETRA" in elemType:
            recombine = False
            numElements = [nLayers] if nLayers > 1 else []
        else:            
            recombine = True
            numElements = [nLayers]

        factory.synchronize()

        for surf in surfaces:

            if isOrganised:
                # only works if the surface is formed by 4 lines
                lines = gmsh.model.getBoundary([(2, surf)])
                if len(lines) == 4:
                    gmsh.model.mesh.setTransfiniteSurface(surf, cornerTags=[])

            if elemType in [ElemType.HEXA8, ElemType.HEXA20]:
                # https://onelab.info/pipermail/gmsh/2010/005359.html
                gmsh.model.mesh.setRecombine(2, surf)

        entities = gmsh.model.getEntities(2)

        # Create new entites for revolution
        revol = factory.revolve(entities, axis.pt1.x, axis.pt1.y, axis.pt1.z, axis.pt2.x, axis.pt2.y, axis.pt2.z, angle, numElements, recombine=recombine)
        revolEntities.extend(revol)

        if angleIs2PI:
            revol = factory.revolve(entities, axis.pt1.x, axis.pt1.y, axis.pt1.z, axis.pt2.x, axis.pt2.y, axis.pt2.z, -angle, numElements, recombine=recombine)
            revolEntities.extend(revol)

        factory.synchronize()

        return revolEntities

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

        self._init_gmsh_factory()

        tic = Tic()

        gmsh.open(mesh)
        
        tic.Tac("Mesh","Mesh import", self.__verbosity)

        if setPhysicalGroups:
            self._Set_PhysicalGroups()

        return self._Construct_Mesh(coef)

    def Mesh_Import_part(self, file: str, dim: int, meshSize=0.0, elemType: ElemType=None, refineGeom=None, folder=""):
        """Build mesh from imported file (.stp or .igs).\n
        You can only use triangles or tetrahedrons.

        Parameters
        ----------
        file : str
            file (.stp, .igs) that gmsh will load to create the mesh.\n
            Note that for igs files, entities cannot be recovered.
        meshSize : float, optional
            mesh size, by default 0.0
        elemType : ElemType, optional
            element type, by default "TRI3" or "TETRA4" depending on dim.
        refineGeom : Geom, optional
            second domain for mesh concentration, by default None
        folder : str, optional
            mesh save folder mesh.msh, by default ""

        Returns
        -------
        Mesh
            Built mesh
        """
        
        # Allow other meshes -> this seems impossible - you have to create the mesh using gmsh to control the type of element.

        if elemType is None:
            elemType = ElemType.TRI3 if dim == 2 else ElemType.TETRA4

        self._init_gmsh_factory('occ') # Only work with occ !! Do not change

        assert meshSize >= 0.0, "Must be greater than or equal to 0."
        self.__CheckType(dim, elemType)
        
        tic = Tic()

        factory = self.__factory

        if '.stp' in file or '.igs' in file:
            factory.importShapes(file)
        else:
            print("Must be a .stp or .igs file")

        self._RefineMesh(refineGeom, meshSize)

        self._Set_PhysicalGroups(setPoints=False, setLines=True, setSurfaces=True, setVolumes=False)

        gmsh.option.setNumber("Mesh.MeshSizeMin", meshSize)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)

        tic.Tac("Mesh","File import", self.__verbosity)

        self._Meshing(dim, elemType, folder=folder)

        return self._Construct_Mesh()

    def _PhysicalGroups_cracks(self, cracks: list, entities: list[tuple]) -> tuple[int, int, int, int]:
        """Creation of physical groups associated with cracks embeded in entities.\n
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

                if pt1.isOpen: entities0D.append(p1); openPoints.append(p1)
                if pt2.isOpen: entities0D.append(p2); openPoints.append(p2)

            elif isinstance(crack, PointsList):

                loop, lines, openPts = self._Loop_From_Points(crack.points, crack.meshSize)
                
                entities0D.extend(openPts)
                openPoints.extend(openPts)
                entities1D.extend(lines)

                surface = self._Surface_From_Loops([loop])
                entities2D.append(surface)

                if crack.isHollow:
                    openLines.extend(lines)
                    openSurfaces.append(surface)

            elif isinstance(crack, Contour):

                loop, openLns, openPts = self._Loop_From_Contour(crack)
                
                entities0D.extend(openPts)
                openPoints.extend(openPts)
                entities1D.extend(openLns)

                surface = self._Surface_From_Loops([loop])
                entities2D.append(surface)

                if crack.isHollow:
                    openLines.extend(openLns)
                    openSurfaces.append(surface)

            elif isinstance(crack, CircleArc):
                
                pC =  self.__factory.addPoint(crack.center.x, crack.center.y, crack.center.z, crack.meshSize)
                p1 = self.__factory.addPoint(crack.pt1.x, crack.pt1.y, crack.pt1.z, crack.meshSize)
                p2 = self.__factory.addPoint(crack.pt2.x, crack.pt2.y, crack.pt2.z, crack.meshSize)
                p3 = self.__factory.addPoint(crack.pt3.x, crack.pt3.y, crack.pt3.z, crack.meshSize)

                if crack.pt1.isOpen: entities0D.append(p1); openPoints.append(p1)
                if crack.pt2.isOpen: entities0D.append(p2); openPoints.append(p2)
                if crack.pt3.isOpen: entities0D.append(p3); openPoints.append(p3)

                line1 = self.__factory.addCircleArc(p1, pC, p3)
                line2 = self.__factory.addCircleArc(p3, pC, p2)
                lines = [line1, line2]
                entities1D.extend(lines)
                if crack.isOpen:
                    openLines.extend(lines)

                self.__factory.synchronize()
                self.__factory.remove([(0,pC)], False)
                
            else:
                # Loop recovery
                hollowLoops, filledLoops = self.__Get_hollow_And_filled_Loops([crack])
                loops = []; loops.extend(hollowLoops); loops.extend(filledLoops)
                
                # Surface construction
                for loop in loops:
                    surface = self._Surface_From_Loops([loop])
                    entities2D.append(surface)

                    if crack.isHollow:
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

    def Mesh_Beams(self, beams: list[_Beam_Model], elemType=ElemType.SEG2, folder=""):
        """Construction of a segment mesh

        Parameters
        beams
        listBeam : list[_Beam_Model]
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

        self._init_gmsh_factory()
        self.__CheckType(1, elemType)

        tic = Tic()
        
        factory = self.__factory

        points = [] 
        lines = []

        for beam in beams:
            line = beam.line
            
            pt1 = line.pt1; x1 = pt1.x; y1 = pt1.y; z1 = pt1.z
            pt2 = line.pt2; x2 = pt2.x; y2 = pt2.y; z2 = pt2.z

            p1 = factory.addPoint(x1, y1, z1, line.meshSize)
            p2 = factory.addPoint(x2, y2, z2, line.meshSize)
            points.append(p1)
            points.append(p2)

            line = factory.addLine(p1, p2)
            lines.append(line)
        
        factory.synchronize()
        self._Set_PhysicalGroups(setLines=False)

        tic.Tac("Mesh","Beam mesh construction", self.__verbosity)

        self._Meshing(1, elemType, folder=folder)

        mesh = self._Construct_Mesh()

        def FuncAddTags(beam: _Beam_Model):
            nodes = mesh.Nodes_Line(beam.line)
            for grp in mesh.Get_list_groupElem():
                grp.Set_Nodes_Tag(nodes, beam.name)
                grp.Set_Elements_Tag(nodes, beam.name)

        [FuncAddTags(beam) for beam in beams]

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
            
            loop = self._Loop_From_Geom(objetGeom)

            if objetGeom.isHollow:
                hollowLoops.append(loop)
            else:                
                filledLoops.append(loop)

        return hollowLoops, filledLoops    

    def Mesh_2D(self, contour: Geom, inclusions: list[Geom]=[], elemType=ElemType.TRI3,
                cracks:list[Geom]=[], refineGeoms: list[Union[Geom,str]]=[], isOrganised=False, folder=""):
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
        refineGeoms : list[Domain|Circle|str], optional
            geometric objects for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        folder : str, optional
            mesh save folder mesh.msh, by default ""

        Returns
        -------
        Mesh
            2D mesh
        """
        
        self._init_gmsh_factory('occ')
        self.__CheckType(2, elemType)

        tic = Tic()

        factory = self.__factory
        
        meshSize = contour.meshSize
        
        self._Surfaces(contour, inclusions)

        # Recovers 2D entities
        entities2D = gmsh.model.getEntities(2)

        # Crack creation
        crackLines, crackSurfaces, openPoints, openLines = self._PhysicalGroups_cracks(cracks, entities2D)

        self._RefineMesh(refineGeoms, meshSize)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)                

        self._Meshing(2, elemType, isOrganised, crackLines=crackLines, openPoints=openPoints, folder=folder)

        return self._Construct_Mesh()

    def Mesh_3D(self, contour: Geom, inclusions: list[Geom]=[],
                extrude=[0,0,1], nLayers=1, elemType=ElemType.TETRA4,
                cracks: list[Geom]=[], refineGeoms: list[Union[Geom,str]]=[], isOrganised=False, folder="") -> Mesh:
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
        refineGeoms : list[Domain|Circle|str], optional
            geometric objects for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        folder : str, optional
            mesh.msh backup folder, by default ""

        Returns
        -------
        Mesh
            3D mesh
        """
        
        self._init_gmsh_factory()
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # the starting 2D mesh is irrelevant
        surfaces = self._Surfaces(contour, inclusions)

        self._Extrude(surfaces=surfaces, extrude=extrude, elemType=elemType, nLayers=nLayers, isOrganised=isOrganised)        

        # Recovers 3D entities
        entities3D = gmsh.model.getEntities(3)

        # Crack creation
        crackLines, crackSurfaces, openPoints, openLines = self._PhysicalGroups_cracks(cracks, entities3D)

        self._RefineMesh(refineGeoms, contour.meshSize)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self._Meshing(3, elemType, folder=folder, crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)
        
        return self._Construct_Mesh()
    
    def Mesh_Revolve(self, contour: Geom, inclusions: list[Geom]=[],
                     axis: Line=Line(Point(), Point(0,1)), angle=2*np.pi, nLayers=180, elemType=ElemType.TETRA4,
                     cracks: list[Geom]=[], refineGeoms: list[Union[Geom,str]]=[],
                     folder="") -> Mesh:
        """Builds a 3D mesh by rotating a surface along an axis.

        Parameters
        ----------
        contour : Geom
            geometry that builds the contour
        inclusions : list[Domain, Circle, PointsList, Contour], optional
            list of hollow and non-hollow objects inside the domain
        axis : Line, optional
            revolution axis, by default Line(Point(), Point(0,1))
        angle : _type_, optional
            revolution angle, by default 2*np.pi
        nLayers : int, optional
            number of layers in revolution, by default 180
        elemType : ElemType, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"]
        cracks : list[Line | PointsList | Countour]
            list of object used to create cracks
        refineGeoms : list[Domain|Circle|str], optional
            geometric objects for mesh refinement, by default []
        folder : str, optional
            mesh.msh backup folder, by default ""

        Returns
        -------
        Mesh
            3D mesh
        """

        self._init_gmsh_factory()
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        # the starting 2D mesh is irrelevant
        surfaces = self._Surfaces(contour, inclusions)        
        
        self._Revolve(surfaces=surfaces, axis=axis, angle=angle, elemType=elemType, nLayers=nLayers)           

        # Recovers 3D entities
        entities3D = gmsh.model.getEntities(3)

        # Crack creation
        crackLines, crackSurfaces, openPoints, openLines = self._PhysicalGroups_cracks(cracks, entities3D)

        self._RefineMesh(refineGeoms, contour.meshSize)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self._Meshing(3, elemType, folder=folder, crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)

        return self._Construct_Mesh()
    
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

        self._init_gmsh_factory()

        view = gmsh.view.add("scalar points")

        gmsh.view.addListData(view, "SP", coordo.shape[0], data.reshape(-1))

        path = Folder.New_File(f"{filename}.pos", folder)

        gmsh.view.write(view, path)

        return path
    
    def _RefineMesh(self, refineGeoms: list[Union[Domain,Circle,str]], meshSize: float):
        """Sets a background mesh

        Parameters
        ----------
        refineGeoms : list[Domain|Circle|str]
            Geometric objects to refine de background mesh
        meshSize : float
            size of elements outside the domain
        """

        # See t10.py in the gmsh tutorials
        # https://gitlab.onelab.info/gmsh/gmsh/blob/master/tutorials/python/t10.py

        if refineGeoms is None or len(refineGeoms) == 0: return

        fields = []

        for geom in refineGeoms:

            if isinstance(geom, Domain):

                coordo = np.array([point.coordo  for point in geom.points])

                field = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(field, "VIn", geom.meshSize)
                gmsh.model.mesh.field.setNumber(field, "VOut", meshSize)
                gmsh.model.mesh.field.setNumber(field, "XMin", coordo[:,0].min())
                gmsh.model.mesh.field.setNumber(field, "XMax", coordo[:,0].max())
                gmsh.model.mesh.field.setNumber(field, "YMin", coordo[:,1].min())
                gmsh.model.mesh.field.setNumber(field, "YMax", coordo[:,1].max())
                gmsh.model.mesh.field.setNumber(field, "ZMin", coordo[:,2].min())
                gmsh.model.mesh.field.setNumber(field, "ZMax", coordo[:,2].max())

            elif isinstance(geom, Circle):

                pC = geom.center
                field = gmsh.model.mesh.field.add("Cylinder")
                gmsh.model.mesh.field.setNumber(field, "VIn", geom.meshSize)
                gmsh.model.mesh.field.setNumber(field, "VOut", meshSize)
                gmsh.model.mesh.field.setNumber(field, "Radius", geom.diam/2)
                gmsh.model.mesh.field.setNumber(field, "XCenter", pC.x)
                gmsh.model.mesh.field.setNumber(field, "YCenter", pC.y)
                gmsh.model.mesh.field.setNumber(field, "ZCenter", pC.z)

            elif isinstance(geom, str):

                if not Folder.Exists(geom) :
                    print(Fore.RED + "The .pos file does not exist." + Fore.WHITE)
                    continue

                if ".pos" not in geom:
                    print(Fore.RED + "Must provide a .pos file" + Fore.WHITE)
                    continue

                gmsh.merge(geom)

                # Add the post-processing view as a new size field:
                field = gmsh.model.mesh.field.add("PostView")
                # gmsh.model.mesh.field.setNumber(field, "ViewIndex", 0)
                # gmsh.model.mesh.field.setNumber(field, "UseClosest", 0)

            elif geom is None:
                continue

            else:
                print(Fore.RED + "refineGeoms must be of type Domain, Circle, str(.pos file)" + Fore.WHITE)
            
            fields.append(field)

        # Let's use the minimum of all the fields as the mesh size field:
        minField = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minField, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minField)

        # Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
        # (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
        # "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size fields
        # better - in particular size fields with large element size gradients:
        gmsh.option.setNumber("Mesh.Algorithm", 5)

    @staticmethod
    def _Set_mesh_order(elemType: str):
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

    def _Set_algorithm(self, elemType: ElemType) -> None:
        """Set the mesh algorithm.\n
        Mesh.Algorithm
            2D mesh algorithm (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad)
            Default value: 6

        Mesh.Algorithm3D
            3D mesh algorithm (1: Delaunay, 3: Initial mesh only, 4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT)
            Default value: 1            

        Mesh.RecombinationAlgorithm
            Mesh recombination algorithm (0: simple, 1: blossom, 2: simple full-quad, 3: blossom full-quad)
            Default value: 1        

        Mesh.SubdivisionAlgorithm
            Mesh subdivision algorithm (0: none, 1: all quadrangles, 2: all hexahedra, 3: barycentric)
            Default value: 0
        """

        if elemType in GroupElem.get_Types1D() or elemType in GroupElem.get_Types2D():
            meshAlgorithm = 6 # 6: Frontal-Delaunay
        elif elemType in GroupElem.get_Types3D():
            meshAlgorithm = 1 # 1: Delaunay
        gmsh.option.setNumber("Mesh.Algorithm", meshAlgorithm)

        recombineAlgorithm = 1
        if elemType in [ElemType.QUAD4, ElemType.QUAD8]:
            subdivisionAlgorithm = 1
        else:
            subdivisionAlgorithm = 0        

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recombineAlgorithm)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", subdivisionAlgorithm)

    def _Meshing(self, dim: int, elemType: str, isOrganised=False, crackLines=None, crackSurfaces=None, openPoints=None, openLines=None, folder="", filename="mesh"):
        """Construction of gmsh mesh from geometry that has been built or imported.

        Parameters
        ----------
        dim : int
            mesh size
        elemType : str
            element type
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
        filename : str, optional
            mesh save folder mesh.msh, by default mesh
        """

        factory = self.__factory
        
        self._Set_algorithm(elemType)
        factory.synchronize()

        tic = Tic()
        if dim == 1:
            gmsh.model.mesh.generate(1)

        elif dim == 2:
            surfaces = [entity2D[1] for entity2D in gmsh.model.getEntities(2)]            
            for surface in surfaces:
                if isOrganised:
                    # only works if the surface is formed by 4 lines
                    lines = gmsh.model.getBoundary([(2, surface)])
                    if len(lines) == 4:
                        gmsh.model.mesh.setTransfiniteSurface(surface, cornerTags=[])

                if elemType in [ElemType.QUAD4,ElemType.QUAD8]:
                    gmsh.model.mesh.setRecombine(2, surface)
            
            # Generates mesh
            gmsh.model.mesh.generate(2)
        
        elif dim == 3:
            self.__factory.synchronize()            

            gmsh.model.mesh.generate(3)

        Interface_Gmsh._Set_mesh_order(elemType)

        gmsh.model.mesh.removeDuplicateNodes()

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
        if '-nopopup' not in sys.argv and self.__openGmsh:
            gmsh.fltk.run()
        
        tic.Tac("Mesh","Meshing", self.__verbosity)

        if folder != "":
            # gmsh.write(Dossier.Join([folder, "model.geo"])) # It doesn't seem to work, but that's okay
            self.__factory.synchronize()

            if not os.path.exists(folder):
                os.makedirs(folder)
            msh = Folder.Join([folder, f"{filename}.msh"])
            gmsh.write(msh)
            tic.Tac("Mesh","Saving .msh", self.__verbosity)

    def _Construct_Mesh(self, coef=1) -> Mesh:
        """Recovering the built mesh"""

        # Old method was boggling
        # The bugs have been fixed because I didn't properly organize the nodes when I created them
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1926
        
        tic = Tic()

        dict_groupElem = {}
        meshDim = gmsh.model.getDimension()
        elementTypes = gmsh.model.mesh.getElementTypes()
        nodes, coord, parametricCoord = gmsh.model.mesh.getNodes()
        
        nodes = np.array(nodes, dtype=int) - 1 # node numbers
        Nn = nodes.shape[0] # Number of nodes

        # Organize nodes from smallest to largest
        sortedIdx = np.argsort(nodes)
        sortedNodes = nodes[sortedIdx]

        # Here we will detect jumps in node numbering
        # Example nodes = [0 1 2 3 4 5 6 8]
        
        # Here we will detect the jump between 6 and 8.
        # diff = [0 0 0 0 0 0 0 1]
        diff = sortedNodes - np.arange(Nn)
        jumpInNodes = np.max(diff) > 0 # detect if there is a jump in the nodes

        # Array that stores the changes        
        # For example below -> Changes = [0 1 2 3 4 5 6 0 7]
        # changes is used such correctedNodes = changes[nodes]
        changes = np.zeros(nodes.max()+1, dtype=int)        
        changes[sortedNodes] = sortedNodes - diff

        # The coordinate matrix of all nodes used in the mesh is constructed        
        coordo: np.ndarray = coord.reshape(-1,3)[sortedIdx,:]

        # Apply coef to scale coordinates
        coordo = coordo * coef

        knownDims = [] # known dimensions in the mesh
        # For each element type
        for gmshId in elementTypes:
                                        
            # Retrieves element numbers and connection matrix
            elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
            elementTags = np.array(elementTags, dtype=int) - 1 # tags for each elements
            nodeTags = np.array(nodeTags, dtype=int) - 1 # connection matrix in shape (e * nPe)

            nodeTags: np.ndarray = changes[nodeTags] # Apply changes to correct jumps in nodes
            
            # Elements
            Ne = elementTags.shape[0] # number of elements
            nPe = GroupElem_Factory.Get_ElemInFos(gmshId)[1] # nodes per element
            connect: np.ndarray = nodeTags.reshape(Ne, nPe) # Builds connect matrix

            # Nodes            
            nodes = np.unique(nodeTags) 
            Nmax = nodes.max() # Check that max node numbers can be reached in coordo
            assert Nmax <= (coordo.shape[0]-1), f"Nodes {Nmax} doesn't exist in coordo"

            # Element group creation
            groupElem = GroupElem_Factory.Create_GroupElem(gmshId, connect, coordo, nodes)
            
            # We add the element group to the dictionary containing all groups
            dict_groupElem[groupElem.elemType] = groupElem
            
            # Check that the mesh does not have a group of elements of this dimension
            if groupElem.dim in knownDims and groupElem.dim == meshDim:
                recoElement = 'Triangular' if meshDim == 2 else 'Tetrahedron'
                raise Exception(f"Importing the mesh is impossible because several {meshDim}D elements have been detected. Try out {recoElement} elements.\n You can also try standardizing the mesh size")
                # TODO make it work ?
                # Can be complicated especially in the creation of elemental matrices and assembly
                # Not impossible but not trivial
                # Relaunch the procedure if it doesn't work?
            knownDims.append(groupElem.dim)

            # Here we'll retrieve the nodes and elements belonging to a group
            physicalGroups = gmsh.model.getPhysicalGroups(groupElem.dim)
            # add nodes and elements associated with physical groups
            def __addPysicalGroup(group: tuple[int, int]):

                dim = group[0]
                tag = group[1]
                name = gmsh.model.getPhysicalName(dim, tag)

                nodeTags, __ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
                    
                # If no node has been retrieved, move on to the nextPhysics group.
                if nodeTags.size == 0: return
                nodeTags = np.array(nodeTags, dtype=int) - 1

                # nodes associated with the group
                nodesGroup = changes[nodeTags] # Apply change

                # add the group for notes and elements
                groupElem.Set_Nodes_Tag(nodesGroup, name)
                groupElem.Set_Elements_Tag(nodesGroup, name)

            [__addPysicalGroup(group) for group in physicalGroups]
        
        tic.Tac("Mesh","Construct mesh object", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def Construct_2D_meshes(L=10, h=10, taille=3) -> list[Mesh]:
        """2D mesh generation."""

        interfaceGmsh = Interface_Gmsh(openGmsh=False, verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), meshSize=taille)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=taille, isOpen=False)
        lineOpen = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=taille, isOpen=True)
        circle = Circle(Point(x=L/2, y=h/2), L/3, meshSize=taille, isHollow=True)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, meshSize=taille, isHollow=False)

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
    def Construct_3D_meshes(L=130, h=13, b=13, taille=7.5, useImport3D=False):
        """3D mesh generation."""        

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=taille)
        circleCreux = Circle(Point(x=L/2, y=0,z=-b/2), h*0.7, meshSize=taille, isHollow=True)
        circle = Circle(Point(x=L/2, y=0 ,z=-b/2), h*0.7, meshSize=taille, isHollow=False)
        axis = Line(domain.pt1+[-1,0], domain.pt1+[-1,h])

        volume = L*h*b

        def testVolume(val):
            assert np.abs(volume-val)/volume <= 1e-6, "Incorrect volume"

        folder = Folder.Get_Path()        
        partPath = Folder.Join([folder,"3Dmodels","beam.stp"])

        interfaceGmsh = Interface_Gmsh()

        list_mesh3D = []
        # For each type of 3D element
        for t, elemType in enumerate(GroupElem.get_Types3D()):
            
            if useImport3D and elemType in ["TETRA4","TETRA10"]:
                meshPart = interfaceGmsh.Mesh_Import_part(partPath, 3, meshSize=taille, elemType=elemType)
                list_mesh3D.append(meshPart)

            mesh1 = interfaceGmsh.Mesh_3D(domain, [], [0,0,-b], 3, elemType=elemType)
            list_mesh3D.append(mesh1)
            testVolume(mesh1.volume)                

            mesh2 = interfaceGmsh.Mesh_3D(domain, [circleCreux], [0,0,-b], 3, elemType)
            list_mesh3D.append(mesh2)            

            mesh3 = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,-b], 3, elemType)
            list_mesh3D.append(mesh3)
            testVolume(mesh3.volume)

        return list_mesh3D