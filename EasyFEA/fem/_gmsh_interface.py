# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module providing an interface with Gmsh (https://gmsh.info/).\n
This module handles geometric objects (_Geom) to facilitate the creation of meshes on Gmsh."""

import sys, os, matplotlib, gmsh
import numpy as np
from typing import Union
from collections.abc import Iterable

# utilities
from ..utilities import Display, Folder, Tic
# geom
from ..Geoms import _Geom, Point, Points, Line, Circle, CircleArc, Domain, Contour
from ..geoms import Normalize
# fem
from ._mesh import Mesh, ElemType
from ._group_elems import _GroupElem, GroupElemFactory

# types
LoopCompatible = Union[Circle, Domain, Points, Contour]
ContourCompatible = Union[Line, CircleArc, Points]
CrackCompatible = Union[Line, Points, Contour, CircleArc]

class Mesher:
    """Mesher class used to construct and generate the mesh via gmsh."""

    def __init__(self, openGmsh=False, gmshVerbosity=False, verbosity=False):
        """Creates a gmsh interface that handles _Geom objects.

        Parameters
        ----------
        openGmsh : bool, optional
            displays the mesh built in gmsh, by default False
        gmshVerbosity : bool, optional
            gmsh can write in the terminal, by default False
        verbosity : bool, optional
            the mesher can write in terminal, by default False
        """
    
        self.__openGmsh = openGmsh
        """gmsh can display the mesh"""
        self.__gmshVerbosity = gmshVerbosity
        """gmsh can write in the terminal"""
        self.__verbosity = verbosity
        """the mesher can write in terminal"""

        # TODO Add debug config to print exceptions and errors 
        # use geom.Plot() to display the error
        # catch the exception and print it
        # create a Mesher Exception ?

        self._Init_gmsh()

        if verbosity:
            Display.Section("Init GMSH interface")

    def __CheckType(self, dim: int, elemType: str) -> None:
        """Checks that the element type is available."""
        if dim == 1:
            assert elemType in ElemType.Get_1D(), f"Must be in {ElemType.Get_1D()}"
        if dim == 2:
            assert elemType in ElemType.Get_2D(), f"Must be in {ElemType.Get_2D()}"
        elif dim == 3:
            assert elemType in ElemType.Get_3D(), f"Must be in {ElemType.Get_3D()}"
    
    def _Init_gmsh(self, factory: str= 'occ') -> None:
        """Initializes gmsh."""
        if not gmsh.isInitialized():
            gmsh.initialize()
        if self.__gmshVerbosity == False:
            gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.model.add("model")
        if factory == 'occ':
            self._factory = gmsh.model.occ
        elif factory == 'geo':
            self._factory = gmsh.model.geo
        else:
            raise Exception("Unknow factory")
        
    def _Synchronize(self) -> None:
        """Synchronizes geometric entities created on gmsh."""

        factory = self._factory

        if factory == gmsh.model.occ:
            # If occ is used, checks whether objects have already been synchronized.
            ents1 = factory.getEntities()
            ents2 = gmsh.model.getEntities()    
            if len(ents1) != len(ents2):
                # Entities are not up to date
                factory.synchronize()
        else:
            factory.synchronize()
        
    def _Loop_From_Geom(self, geom: LoopCompatible) -> tuple[int, list[int], list[int]]:
        """Creates a loop based on the geometric object.\n
        returns loop, lines, points"""

        if isinstance(geom, Circle):
            loop, lines, points = self._Loop_From_Circle(geom)[:3]
        elif isinstance(geom, Domain):
            loop, lines, points = self._Loop_From_Domain(geom)[:3]
        elif isinstance(geom, Points):
            contour = geom.Get_Contour()    
            loop, lines, points = self._Loop_From_Contour(contour)[:3]
        elif isinstance(geom, Contour):
            loop, lines, points = self._Loop_From_Contour(geom)[:3]
        else:
            raise Exception("Must be a circle, a domain, points or a contour.")
        
        return loop, lines, points
    
    def _Loop_From_Contour(self, contour: Contour) -> tuple[int, list[int], list[int], list[int], list[int]]:
        """Creates a loop with a contour object (list of Line, CircleArc, Points).\n
        returns loop, lines, points, openLines, openPoints
        """

        factory = self._factory

        points: list[int] = []
        lines: list[int] = []

        nGeom = len(contour.geoms)

        openPoints = []
        openLines = []

        for i, geom in enumerate(contour.geoms):

            assert isinstance(geom, ContourCompatible), "Must be a Line, CircleArc or Points"

            if i == 0:
                p1 = factory.addPoint(*geom.pt1.coord, geom.meshSize)
                if geom.pt1.isOpen: openPoints.append(p1)
                p2 = factory.addPoint(*geom.pt2.coord, geom.meshSize)
                if geom.pt2.isOpen: openPoints.append(p2)
                points.extend([p1,p2])
            elif i > 0 and i+1 < nGeom:
                p1 = p2
                p2 = factory.addPoint(*geom.pt2.coord, geom.meshSize)
                if geom.pt2.isOpen: openPoints.append(p2)
                points.append(p2)
            else:
                p1 = p2
                p2 = firstPoint

            if isinstance(geom, Line):

                line = factory.addLine(p1, p2)

                if geom.isOpen:
                    openLines.append(line)

                lines.append(line)

            elif isinstance(geom, CircleArc):                

                pC =  factory.addPoint(*geom.center.coord, geom.meshSize)
                p3 = factory.addPoint(*geom.pt3.coord)

                if np.abs(geom.angle) > np.pi:
                    line1 = factory.addCircleArc(p1, pC, p3)
                    line2 = factory.addCircleArc(p3, pC, p2)
                    lines.extend([line1, line2])
                    if geom.isOpen:
                        openLines.extend([line1, line2])
                else:                    
                    if factory == gmsh.model.occ:                        
                        line = factory.addCircleArc(p1, p3, p2, center=False)
                    else:
                        n = geom.n
                        line = factory.addCircleArc(p1, pC, p2, nx=n[0], ny=n[1], nz=n[2])
                    lines.append(line)
                    if geom.isOpen:
                        openLines.append(line)
                
                factory.remove([(0,pC)])
                factory.remove([(0,p3)])

            elif isinstance(geom, Points):
                
                # get points to construct the spline
                splinePoints = [factory.addPoint(*p.coord, geom.meshSize) for p in geom.points[1:-1]]        
                splinePoints.insert(0,p1)
                splinePoints.append(p2)
                
                line = factory.addSpline(splinePoints)
                lines.append(line)
                if geom.isOpen:
                    openLines.append(line)

                factory.remove([(0,p) for p in splinePoints[1:-1]])

            if i == 0:
                firstPoint = p1
        
        loop = factory.addCurveLoop(lines)

        return loop, lines, points, openLines, openPoints

    def _Loop_From_Circle(self, circle: Circle) -> tuple[int, list[int], list[int]]:
        """Creates a loop with a circle object.\n
        returns loop, lines, points
        """

        factory = self._factory

        center = circle.center

        # Circle points                
        p0 = factory.addPoint(*center.coord, circle.meshSize) # center
        p1 = factory.addPoint(*circle.pt1.coord, circle.meshSize)
        p2 = factory.addPoint(*circle.pt2.coord, circle.meshSize)
        p3 = factory.addPoint(*circle.pt3.coord, circle.meshSize)
        p4 = factory.addPoint(*circle.pt4.coord, circle.meshSize)
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

        return loop, lines, points

    def _Loop_From_Domain(self, domain: Domain) -> tuple[int, list[int], list[int]]:
        """Creates a loop with a domain object.\n
        returns loop, lines, points
        """
        pt1 = domain.pt1
        pt2 = domain.pt2
        mS = domain.meshSize

        factory = self._factory

        p1 = factory.addPoint(pt1.x, pt1.y, pt1.z, mS)
        p2 = factory.addPoint(pt2.x, pt1.y, pt1.z, mS)
        p3 = factory.addPoint(pt2.x, pt2.y, pt1.z, mS)
        p4 = factory.addPoint(pt1.x, pt2.y, pt1.z, mS)
        points = [p1,p2,p3,p4]

        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)
        lines = [l1,l2,l3,l4]

        loop = factory.addCurveLoop(lines)
        
        return loop, lines, points

    def _Surface_From_Loops(self, loops: list[int]) -> int:
        """Creates a gmsh surface with a gmsh loop (must be a plane surface).\n
        returns surface
        """
        # must form a plane surface
        surface = self._factory.addPlaneSurface(loops)

        return surface
    
    def _Surfaces(self, contour: _Geom, inclusions: list[_Geom]=[],
                  elemType=ElemType.TRI3, isOrganised=False) -> tuple[list[int],list[int],list[int]]:
        """Creates gmsh surfaces.\n
        They must be plane surfaces otherwise you must use 'factory.addSurfaceFilling' function.\n
        returns surfaces, lines, points

        Parameters
        ----------
        contour : _Geom
            the object that creates the surface area
        inclusions : list[_Geom]
            hollow or filled objects contained in the contour surface.\n
            CAUTION: all inclusions must be contained within the contour and must not intersect.
        elemType : ElemType, optional
            element type used, by default TRI3
        isOrganised : bool, optional
            mesh is organized, by default False
        """

        assert isinstance(contour, _Geom), "The contour must be a geometric object."
        assert isinstance(inclusions, Iterable), "inclusions must be a list of geometric objects."

        factory = self._factory

        # Create a contour surface
        loopContour, lines, points = self._Loop_From_Geom(contour)

        # Create all hollow and filled loops associated with inclusions
        hollowLoops, filledLoops = self.__Get_hollow_And_filled_Loops(inclusions)
        
        loops = [loopContour] # contour loop
        loops.extend(hollowLoops) # Hollow loops
        loops.extend(filledLoops) # Filled loops

        surfaces = [self._Surface_From_Loops(loops)] # first surface
        [surfaces.append(factory.addPlaneSurface([loop])) for loop in filledLoops] # Adds filled surfaces

        # The number of elements per line is calculated here to organize the surface if it can be.
        if not isOrganised or isinstance(contour, Circle) or len(inclusions) > 0:
            # Cannot be organized if there are inclusions.
            # It is not necessary to impose a number of elements for circles and domains!
            numElems = []
        else:
            # geom objects from Domain, contour or points
            if isinstance(contour, Domain):
                # construct points
                p1 = contour.pt1
                p3 = contour.pt2
                p2 = p1.copy(); p2.x = p3.x
                p4 = p1.copy(); p4.y = p3.y
                # construct points
                l1 = Line(p1, p2, contour.meshSize)
                l2 = Line(p2, p3, contour.meshSize)
                l3 = Line(p3, p4, contour.meshSize)
                l4 = Line(p4, p1, contour.meshSize)
                # construct geoms
                geoms = Contour([l1, l2, l3, l4]).geoms
            elif isinstance(contour, Contour):
                geoms = contour.geoms
            elif isinstance(contour, Points):
                geoms = contour.Get_Contour().geoms
            
            N = len(geoms) # number of geom in contour
            def get_numElem(geom: ContourCompatible):
                meshSize = geom.length if geom.meshSize == 0 else geom.meshSize                
                return geom.length/meshSize            
            if N % 2 == 0: # N is odd
                numElems = [get_numElem(geom) for geom in geoms[:N//2]]
                numElems = numElems*2
            else:
                numElems = [get_numElem(geom) for geom in geoms]
                
        self._Surfaces_Organize(surfaces, elemType, isOrganised, numElems)

        return surfaces, lines, points
    
    def _Surfaces_Organize(self, surfaces: list[int], elemType: ElemType,
                           isOrganised=False, numElems:list[int]=[]) -> None:
        """Organizes surfaces.

        Parameters
        ----------
        surfaces : list[int]
            list of gmsh surfaces
        elemType : ElemType
            element type
        isOrganised : bool, optional
            the mesh is organized, by default False
        numElems : list[int], optional
            number of elements per line, by default []
        """

        self._Synchronize() # mandatory

        setRecombine = elemType.startswith(("QUAD","HEXA"))
        
        for surf in surfaces:

            lines = gmsh.model.getBoundary([(2, surf)])
            if len(lines) == len(numElems):
                [gmsh.model.mesh.setTransfiniteCurve(l[1], int(n+1))
                    for l, n in zip(lines, numElems)]

            if isOrganised:                
                if len(lines) in [3, 4]:
                    # only works if the surface is formed by 3 or 4 lines
                    gmsh.model.mesh.setTransfiniteSurface(surf)

            if setRecombine:
                # see https://onelab.info/pipermail/gmsh/2010/005359.html
                gmsh.model.mesh.setRecombine(2, surf)
    
    def _Additional_Surfaces(self, dim: int, surfaces:list[_Geom], elemType, isOrganised) -> None:
        """Adds surfaces to existing dim entities. Tip: if the mesh is not well generated, you can also give the inclusions.

        Parameters
        ----------
        dim : int
            dimension (dim >= 2)
        surfaces : list[_Geom]
            surfaces
        elemType : ElemType
            element type used
        isOrganised : bool
            mesh is organized
        """

        assert isinstance(surfaces, Iterable), "surfaces must be a list of geometric objects."
        
        assert dim >= 2

        factory = self._factory
                
        list_surface: list[_Geom] = []
        for surface in surfaces:
            assert isinstance(surface, _Geom)
            if not surface.isHollow:
                # first create an hollow surface with cut
                # then add the surface
                newSurf = surface.copy()
                newSurf.isHollow = True
                list_surface.append(newSurf)
            list_surface.append(surface)

        for surface in list_surface:            
            # get old entities
            oldEntities = factory.getEntities(dim)

            # Create new surfaces
            newSurfaces = self._Surfaces(surface, [], elemType, isOrganised)[0]

            # Delete or add created entities to the current geometry.
            newEntities = [(2, surf) for surf in newSurfaces]

            if surface.isHollow:
                factory.cut(oldEntities, newEntities)
            else:
                factory.fragment(oldEntities, newEntities, False, True)


    def _Additional_Lines(self, dim: int, lines:list[Union[Line, CircleArc]]) -> None:
        """Adds lines to existing dim entities. WARNING: lines must be within the domain.

        Parameters
        ----------
        dim : int
            dimension (dim >= 1)
        lines : list[Union[Line, CircleArc]]
            lines
        """
        
        assert dim >= 1

        factory = self._factory

        for line in lines:
            oldEntities = factory.getEntities(dim)
            if isinstance(line, Line):
                p1 = factory.addPoint(*line.pt1.coord, line.meshSize)
                p2 = factory.addPoint(*line.pt2.coord, line.meshSize)
                geom_line = factory.addLine(p1, p2)
            elif isinstance(line, CircleArc):
                p1 = factory.addPoint(*line.pt1.coord, line.meshSize) # start
                p2 = factory.addPoint(*line.pt2.coord, line.meshSize) # end
                p3 = factory.addPoint(*line.pt3.coord, line.meshSize) # mid
                geom_line = factory.addCircleArc(p1, p3, p2, center=False)
            else:
                raise Exception("You need to give lines or arcs.")
            factory.fragment(oldEntities, [(1, geom_line)], False, True)
            
    def _Additional_Points(self, dim: int, points:list[Point], meshSize: float=0.0) -> None:
        """Adds points to existing dim entities. WARNING points must be within the domain.

        Parameters
        ----------
        dim : int
            dimension (dim >= 1)
        points : list[Point]
            points
        meshSize : float
            meshSize
        """
        
        assert dim >= 1

        factory = self._factory

        oldEntities = factory.getEntities(dim)
        
        list_point: list[int] = []
        for point in points:
            if isinstance(point, Point):
                p = factory.addPoint(*point.coord, meshSize)
            else:
                raise Exception("You need to give a list of point.")
            list_point.append(p)

        self._Synchronize()
        for entity in oldEntities:
            gmsh.model.mesh.embed(0, list_point, entity[0], entity[1])


    def _Spline_From_Points(self, points: Points) -> tuple[int, list[int]]:
        """Creates a gmsh spline from points.\n
        returns spline, points"""

        meshSize = points.meshSize
        gmshPoints = [self._factory.addPoint(*p.coord, meshSize) for p in points.points]

        spline = self._factory.addSpline(gmshPoints)
        # remove all points except the first and the last points
        self._factory.remove([(0,p) for p in gmshPoints[1:-1]])

        # get first and last points
        points = [gmshPoints[0], gmshPoints[-1]]

        return spline, points
    
    __dict_name_dim = {
        0 : "P",
        1 : "L",
        2 : "S",
        3 : "V"
    }

    def _Set_PhysicalGroups(self, setPoints=True, setLines=True, setSurfaces=True, setVolumes=True) -> None:
        """Creates physical groups based on created entities."""
        
        self._Synchronize() # mandatory

        entities = np.asarray(self._factory.getEntities(), dtype=int)
        
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

    def _Extrude(self, surfaces: list[int], extrude=[0,0,1], elemType=ElemType.TETRA4, layers:list[int]=[]) -> list[tuple[int, int]]:
        """Extrudes gmsh surfaces and returns extruded entities.

        Parameters
        ----------
        surfaces : list[int]
            gmsh surfaces
        extrude : list, optional
            extrusion vector, by default [0,0,1]
        elemType : ElemType, optional
            element type used, by default "TETRA4"        
        layers: list[int], optional
            layers in the extrusion, by default []
        """
        
        factory = self._factory

        extruEntities = []

        if isinstance(layers, int):
            layers = [layers]

        if "TETRA" in elemType:
            recombine = False
        else:
            recombine = True
            if len(layers) == 0:
                layers = [1]

        entites = [(2, surf) for surf in surfaces]
        extru = factory.extrude(entites, *extrude, recombine=recombine, numElements=layers)
        extruEntities.extend(extru)

        return extruEntities
    
    def _Revolve(self, surfaces: list[int], axis: Line, angle: float=360.0, elemType: ElemType=ElemType.TETRA4, layers:list[int]=[30]) -> list[tuple[int, int]]:
        """Revolves gmsh surfaces and returns revolved entities.

        Parameters
        ----------
        surfaces : list[int]
            gmsh surfaces
        axis : Line
            rotation axis
        angle: float, optional
            rotation angle in deg, by default 360.0
        elemType : ElemType, optional
            element type used
        layers: list[int], optional
            layers in the rotation, by default [30]
        """
        
        factory = self._factory

        angle = angle * np.pi/180

        angleIs2PI = np.abs(angle) == 2 * np.pi

        if angleIs2PI:
            angle = angle / 2
            layers = [l//2 for l in layers]

        revolEntities = []

        if "TETRA" in elemType:
            recombine = False
        else:
            recombine = True
            if len(layers) == 0:
                layers = [3]

        entities = [(2,s) for s in surfaces]

        p0 = axis.pt1.coord
        a0 = Normalize(axis.pt2.coord - p0)

        # Create new entites
        revol = factory.revolve(entities, *p0, *a0, angle, layers, recombine=recombine)
        revolEntities.extend(revol)

        if angleIs2PI:
            revol = factory.revolve(entities, *p0, *a0, -angle, layers, recombine=recombine)
            revolEntities.extend(revol)

        return revolEntities
    
    def _Linking_Surfaces(self,
                           lines1: list[int], points1: list[int],
                           lines2: list[int], points2: list[int],
                           linkingLines: list[int],
                           elemType: ElemType, nLayers:int=0,
                           listPoints: list[list[Point]]=[]):
        """Creates linking surfaces based on lines and linkingLines.\n
        return linkingSurfaces.
        """

        # check that the given entities are linkable
        assert len(lines1) == len(lines2), "Must provide same number of lines."
        nP, nL = len(points1), len(lines1)
        assert nP == nL, "Must provide the same number of points as lines."

        nLayers = int(nLayers)

        factory = self._factory        
        
        linkingSurfaces: list[int] = []
        list_corners: list[tuple[int, int, int, int]] = []

        for i in range(nP):
            j = i+1 if i+1 < nP else 0
            
            # get the lines to construct the surfaces
            l1 = lines1[i]
            l2 = linkingLines[j]
            l3 = lines2[i]
            l4 = linkingLines[i]
            # get the points of the surface
            p1, p2 = points1[i], points1[j]
            p3, p4 = points2[i], points2[j]
            list_corners.append((p1,p2,p3,p4))
            # loop to create the surface (- are optionnal)
            loop = factory.addCurveLoop([l1,l2,-l3,-l4])
            # create the surface and add it to linking surfaces
            if len(listPoints) == 0:
                surf = factory.addSurfaceFilling(loop)
            else:
                pointTags = [factory.addPoint(*p.coord) for p in listPoints[i]]
                surf = factory.addSurfaceFilling(loop, pointTags=pointTags)
                factory.remove([(0, p) for p in pointTags]) 
            linkingSurfaces.append(surf)
            
        assert len(list_corners) == len(linkingSurfaces)

        if nLayers > 0:

            self._Synchronize()

            useRecombine = 'HEXA' in elemType or 'PRISM' in elemType
            
            # organize the transfinite lines
            [gmsh.model.mesh.setTransfiniteCurve(l, nLayers+1) for l in linkingLines]
            
            # surf must be transfinite to have a strucutred surfaces during the extrusion
            for surf, corners in zip(linkingSurfaces, list_corners):
                gmsh.model.mesh.setTransfiniteSurface(surf, cornerTags=corners)    

                if useRecombine:
                    # must recombine the surface in case we use PRISM or HEXA elements
                    gmsh.model.mesh.setRecombine(2, surf)

        return linkingSurfaces
    
    def _Link_Contours(self, contour1: Contour, contour2: Contour, elemType: ElemType,
                      nLayers:int=0, numElems:list[int]=[]) -> list[tuple[int, int]]:
        """Links 2 contours and create a volume.\n
        Contours must be connectable, i.e. they must have the same number of points and lines.

        Parameters
        ----------
        contour1 : Contour
            the first contour
        contour2 : Contour
            the second contour
        elemType : ElemType
            element type used
        nLayers : int, optional
            number of layers joining the contours, by default 0
        numElems : list[int], optional
            number of elements for each lines in contour, by default []

        Returns
        -------
        list[tuple[int, int]]
            created entities
        """

        tic = Tic()

        factory = self._factory
        
        # specify whether contour surfaces can be organized
        canBeOrganised = len(contour1.geoms) == 4
        if not canBeOrganised:
            Display.MyPrintError("Caution! We recommend handling surfaces with 3 or 4 corners.")
        # specify if it is necessary to recombine bonding surfaces
        useTransfinite = ('HEXA' in elemType or 'PRISM' in elemType) and len(contour1.geoms) in [3, 4]
        
        # construct loops, lines and points for contour1 and contour2
        loop1, lines1, points1 = self._Loop_From_Geom(contour1)
        loop2, lines2, points2 = self._Loop_From_Geom(contour2)

        surf1 = factory.addSurfaceFilling(loop1) # here we dont use self._Surfaces()
        surf2 = factory.addSurfaceFilling(loop2)
        
        # organize the mesh generation
        if useTransfinite:
            if len(numElems) == 0:                
                numElems = [int(geom.length / geom.meshSize) for geom in contour1.geoms]    
                assert len(numElems) == len(lines1)
        # Here, the following function will synchronize the created entities
        self._Surfaces_Organize([surf1, surf2], elemType, canBeOrganised, numElems)

        # create link between every points belonging to points1 and points2
        linkingLines: list[int] = [factory.addLine(pi, pj) for pi, pj in zip(points1, points2)]

        linkingSurfaces = self._Linking_Surfaces(lines1, points1,
                                                  lines2, points2,
                                                  linkingLines,
                                                  elemType, nLayers)

        # append entities together
        points = [*points1, *points2]
        lines = [*lines1, *lines2, *linkingLines]
        surfaces = [surf1, surf2, *linkingSurfaces]
        
        shell = factory.addSurfaceLoop(surfaces)
        factory.addVolume([shell])

        if useTransfinite:
            self._Synchronize()
            gmsh.model.mesh.setTransfiniteVolume(shell, points)

        tic.Tac("Mesh","Link contours", self.__verbosity)

        # get entities
        entities = self.Get_Entities(points, lines, surfaces, [shell])

        return entities
    
    @staticmethod
    def Get_Entities(points=[], lines=[], surfaces=[], volumes=[]) -> list[tuple[int, int]]:
        """Get entities from from points, lines, surfaces and volumes tags"""
        entities = []
        entities.extend([(0,p) for p in points])
        entities.extend([(1,l) for l in lines])
        entities.extend([(2,s) for s in surfaces])
        entities.extend([(3,v) for v in volumes])
        return entities

    def Mesh_Import_mesh(self, mesh: str, setPhysicalGroups=False, coef=1.0) -> Mesh:
        """Creates the mesh from an *.msh file.

        Parameters
        ----------
        mesh : str
            *.msh file
        setPhysicalGroups : bool, optional
            creates physical groups, by default False
        coef : float, optional
            coef applied to the node coordinates, by default 1.0

        Returns
        -------
        Mesh
            Created mesh
        """

        self._Init_gmsh()

        tic = Tic()

        gmsh.open(mesh)
        
        tic.Tac("Mesh","Mesh import", self.__verbosity)

        if setPhysicalGroups:
            self._Set_PhysicalGroups()

        return self._Mesh_Get_Mesh(coef)

    def Mesh_Import_part(self, file: str, dim: int, meshSize=0.0, elemType: ElemType=None, refineGeoms=[None], folder="") -> Mesh:
        """Creates the mesh from *.stp or *.igs files.\n
        You can only use triangles or tetrahedrons.

        Parameters
        ----------
        file : str
            *.stp or *.igs files.\n
            Note that for igs files, entities cannot be recovered.
        meshSize : float, optional
            mesh size, by default 0.0
        elemType : ElemType, optional
            element type, by default "TRI3" or "TETRA4" depending on dim.
        refineGeoms : list[Domain|Circle|str]
            geometric objects to refine the mesh
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """
        
        # Allow other meshes -> this seems impossible - you have to create the mesh using gmsh to control the type of element.

        __doesNotWork = [ElemType.HEXA8.name, ElemType.HEXA20.name,
                         ElemType.PRISM6.name, ElemType.PRISM15.name] # keep .name to improve error display
        if elemType is None:
            elemType = ElemType.TRI3 if dim == 2 else ElemType.TETRA4
        elif elemType in __doesNotWork:
            from ..utilities.Display import MyPrintError
            MyPrintError(f"It is not possible to mesh an imported part with the following elements: {__doesNotWork}")
            elemType = ElemType.TETRA4 if elemType in [ElemType.HEXA8, ElemType.HEXA20] else ElemType.TETRA10
            MyPrintError(f"\nThe part will be meshed with {elemType} elements.")

        self._Init_gmsh() # Only work with occ !! Do not change

        assert meshSize >= 0.0, "Must be greater than or equal to 0."
        self.__CheckType(dim, elemType)
        
        tic = Tic()

        factory = self._factory

        if '.stp' in file or '.igs' in file:
            factory.importShapes(file)
        else:
            print("Must be a .stp or .igs file")

        if meshSize > 0:
            self.Set_meshSize(meshSize)

        self._Mesh_Refine(refineGeoms, meshSize)

        self._Set_PhysicalGroups(setPoints=False, setLines=True, setSurfaces=True, setVolumes=False)

        gmsh.option.setNumber("Mesh.MeshSizeMin", meshSize)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)

        tic.Tac("Mesh","File import", self.__verbosity)

        self._Mesh_Generate(dim, elemType, folder=folder)

        return self._Mesh_Get_Mesh()

    def _Cracks_SetPhysicalGroups(self, cracks: list[CrackCompatible], entities: list[tuple]) -> tuple[int, int, int, int]:
        """Creates physical groups for cracks embeded in existing gmsh entities.\n
        returns crackLines, crackSurfaces, openPoints, openLines
        """

        assert isinstance(cracks, Iterable), "cracks must be a list of geometric objects."

        factory = self._factory

        if len(cracks) == 0:
            return None, None, None, None
        
        # lists containing open entities
        cracks_1D = []; openPoints = []
        cracks_2D = []; openLines = []

        entities_0D = []
        entities_1D = []
        entities_2D = []

        for crack in cracks:
            if isinstance(crack, Line): # 1D CRACK
                # Create points
                pt1 = crack.pt1
                p1 = factory.addPoint(pt1.x, pt1.y, pt1.z, crack.meshSize)
                pt2 = crack.pt2
                p2 = factory.addPoint(pt2.x, pt2.y, pt2.z, crack.meshSize)
                entities_0D.extend([p1,p2])

                # Create line
                line = factory.addLine(p1, p2)
                entities_1D.append(line)

                if crack.isOpen:
                    cracks_1D.append(line)
                    if pt1.isOpen: openPoints.append(p1)
                    if pt2.isOpen: openPoints.append(p2)

            elif isinstance(crack, Points):  # 1D CRACK

                loop, lines, points, openLns, openPts = self._Loop_From_Contour(crack.Get_Contour())

                self._factory.remove([(1,loop),(1,lines[-1])])
                
                entities_0D.extend(points)
                entities_1D.extend(lines[:-1])

                if crack.isOpen:
                    cracks_1D.extend(lines[:-1])
                    openLines.extend(openLns)    
                    openPoints.extend(openPts)

            elif isinstance(crack, Contour):  # 2D CRACK

                loop, lines, points, openLns, openPts = self._Loop_From_Contour(crack)
                
                try:
                    surf = self._Surface_From_Loops([loop])
                except Exception:
                    surf = self._factory.addSurfaceFilling(loop)
                
                entities_0D.extend(points)
                entities_1D.extend(lines)
                entities_2D.append(surf)

                if crack.isOpen:
                    cracks_2D.append(surf)
                    openLines.extend(openLns)    
                    openPoints.extend(openPts)

            elif isinstance(crack, CircleArc): # 1D CRACK
                
                # add points
                pC =  factory.addPoint(*crack.center.coord, crack.meshSize)
                p1 = factory.addPoint(*crack.pt1.coord, crack.meshSize)
                p2 = factory.addPoint(*crack.pt2.coord, crack.meshSize)
                p3 = factory.addPoint(*crack.pt3.coord, crack.meshSize)
                entities_0D.extend([p1,p2,p3])

                # add lines
                line1 = factory.addCircleArc(p1, pC, p3)
                line2 = factory.addCircleArc(p3, pC, p2)
                lines = [line1, line2]
                entities_1D.extend(lines)

                if crack.isOpen:
                    cracks_1D.extend(lines)
                    if crack.pt1.isOpen: openPoints.append(p1)
                    if crack.pt2.isOpen: openPoints.append(p2)
                    if crack.pt3.isOpen: openPoints.append(p3)
                
                factory.remove([(0,pC)], False)

            else:

                raise Exception("Cracks must be Line, Points, Contour or CircleArc")    

        newEntities = [(0, point) for point in entities_0D]
        newEntities.extend([(1, line) for line in entities_1D])
        newEntities.extend([(2, surf) for surf in entities_2D])

        if factory == gmsh.model.occ:
            o, m = gmsh.model.occ.fragment(entities, newEntities, removeTool=False)

        self._Synchronize() # mandatory

        crackLines = gmsh.model.addPhysicalGroup(1, cracks_1D) if len(cracks_1D) > 0 else None
        crackSurfaces = gmsh.model.addPhysicalGroup(2, cracks_2D) if len(cracks_2D) > 0 else None

        openPoints = gmsh.model.addPhysicalGroup(0, openPoints) if len(openPoints) > 0 else None
        openLines = gmsh.model.addPhysicalGroup(1, openLines) if len(openLines) > 0 else None

        return crackLines, crackSurfaces, openPoints, openLines

    def Mesh_Beams(self, beams: list, elemType=ElemType.SEG2, additionalPoints:list[Point]=[], folder="") -> Mesh:
        """Creates a beam mesh.

        Parameters
        ----------
        beams : list[_Beam]
            list of Beams
        elemType : str, optional
            element type, by default "SEG2" ["SEG2", "SEG3", "SEG4"]
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh
        additionalPoints : list[Point]
            additional points that will be added to the mesh. WARNING: points must be within the domain.

        Returns
        -------
        Mesh
            Created mesh
        """

        # materials
        from ..materials._beam import _Beam
        beams: list[_Beam] = beams

        assert isinstance(beams, Iterable), "beams must be a list of beams."

        self._Init_gmsh()
        self.__CheckType(1, elemType)

        tic = Tic()
        
        factory = self._factory

        points = [] 
        lines = []
        list_meshSize = []

        for beam in beams:
            line = beam.line
            list_meshSize.append(line.meshSize)

            pt1 = line.pt1; x1 = pt1.x; y1 = pt1.y; z1 = pt1.z
            pt2 = line.pt2; x2 = pt2.x; y2 = pt2.y; z2 = pt2.z

            p1 = factory.addPoint(x1, y1, z1, line.meshSize)
            p2 = factory.addPoint(x2, y2, z2, line.meshSize)
            points.append(p1)
            points.append(p2)

            line = factory.addLine(p1, p2)
            lines.append(line)

        # remove meshSize = 0
        list_meshSize = [m for m in list_meshSize if m>0]
        mS = np.min(list_meshSize) if len(list_meshSize) > 0 else 0
        self._Additional_Points(1, additionalPoints, mS)
        
        self._Set_PhysicalGroups(setLines=False)

        tic.Tac("Mesh","Beam mesh construction", self.__verbosity)

        self._Mesh_Generate(1, elemType, folder=folder)

        mesh = self._Mesh_Get_Mesh()

        def FuncAddTags(beam: _Beam):
            nodes = mesh.Nodes_Line(beam.line)
            for grp in mesh.Get_list_groupElem():
                grp._Set_Nodes_Tag(nodes, beam.name)
                grp._Set_Elements_Tag(nodes, beam.name)

        [FuncAddTags(beam) for beam in beams]

        return mesh

    def __Get_hollow_And_filled_Loops(self, inclusions: list[_Geom]) -> tuple[list[int], list[int]]:
        """Creates hollow and filled loops.

        Parameters
        ----------
        inclusions : Circle | Domain | Points | Contour
            geometric objects contained in the domain.

        Returns
        -------
        tuple[list[int], list[int]]
            created hollow and filled loops
        """
        hollowLoops = []
        filledLoops = []
        for objetGeom in inclusions:
            
            loop = self._Loop_From_Geom(objetGeom)[0]

            if objetGeom.isHollow:
                hollowLoops.append(loop)
            else:                
                filledLoops.append(loop)

        return hollowLoops, filledLoops

    def Mesh_2D(self, contour: _Geom, inclusions: list[_Geom]=[], elemType=ElemType.TRI3,
                cracks:list[_Geom]=[], refineGeoms: list[Union[_Geom,str]]=[],
                isOrganised=False, additionalSurfaces:list[_Geom]=[],
                additionalLines:list[Union[Line,CircleArc]]=[], additionalPoints: list[Point]=[], folder="") -> Mesh:
        """Creates a 2D mesh from a contour and inclusions that must form a closed plane surface.

        Parameters
        ----------
        contour : _Geom
            geom object
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        elemType : str, optional
            element type, by default "TRI3" ["TRI3", "TRI6", "TRI10", "QUAD4", "QUAD8"]
        cracks : list[Line | Points | Countour]
            list of geom object used to create open or closed cracks 
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[_Geom]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """

        # this function only work for occ factory (# .fragment() & .getEntities())
        self._Init_gmsh('occ')
        self.__CheckType(2, elemType)

        tic = Tic()

        factory = self._factory

        self._Surfaces(contour, inclusions, elemType, isOrganised)
        self._Additional_Surfaces(2, additionalSurfaces, elemType, isOrganised)
        self._Additional_Lines(2, additionalLines)
        self._Additional_Points(2, additionalPoints)
        # TODO: add contour to refineGeoms by default when adding surfaces, lines or points
        # additionalSurfaces, additionalLines, additionalPoints
        # adding these lines, points or surfaces will probably break the old mesh size conditions.
        # adding contour to refineGeoms ensures that the mesh size is correct.

        # Recover 2D entities
        entities_2D = factory.getEntities(2)

        # Crack creation
        crackLines, __, openPoints, __ = self._Cracks_SetPhysicalGroups(cracks, entities_2D)

        # get created surfaces
        surfaces = [entity[1] for entity in factory.getEntities(2)]
        self._Surfaces_Organize(surfaces, elemType, isOrganised)

        self._Mesh_Refine(refineGeoms, contour.meshSize)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self._Mesh_Generate(2, elemType, crackLines=crackLines, openPoints=openPoints, folder=folder)

        return self._Mesh_Get_Mesh()

    def Mesh_Extrude(self, contour: _Geom, inclusions: list[_Geom]=[],
                extrude=[0,0,1], layers:list[int]=[], elemType=ElemType.TETRA4,
                cracks: list[_Geom]=[], refineGeoms: list[Union[_Geom,str]]=[],
                isOrganised=False, additionalSurfaces:list[_Geom]=[],
                additionalLines:list[Union[Line,CircleArc]]=[], additionalPoints: list[Point]=[], folder="") -> Mesh:
        """Creates a 3D mesh by extruding a surface constructed from a contour and inclusions.

        Parameters
        ----------
        contour : _Geom
            geom object
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        extrude : list, optional
            extrusion vector, by default [0,0,1]
        layers: list[int], optional
            layers in the extrusion, by default []
        elemType : str, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"]
        cracks : list[Line | Points | Countour]
            list of geom object used to create open or closed cracks 
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[_Geom]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """
        
        # this function only work for occ factory (# .fragment() & .getEntities())
        self._Init_gmsh()
        self.__CheckType(3, elemType)
        
        tic = Tic()

        factory = self._factory

        self._Surfaces(contour, inclusions, elemType, isOrganised)
        self._Additional_Surfaces(2, additionalSurfaces, elemType, isOrganised)
        self._Additional_Lines(2, additionalLines)
        self._Additional_Points(2, additionalPoints)

        # get created surfaces        
        surfaces = [entity[1] for entity in factory.getEntities(2)]
        self._Surfaces_Organize(surfaces, elemType, isOrganised)

        self._Extrude(surfaces=surfaces, extrude=extrude, elemType=elemType, layers=layers)

        # get 3D entities
        entities_3D = factory.getEntities(3)

        # create cracks
        crackLines, crackSurfaces, openPoints, openLines = self._Cracks_SetPhysicalGroups(cracks, entities_3D)

        self._Mesh_Refine(refineGeoms, contour.meshSize, extrude=extrude)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self._Mesh_Generate(3, elemType, folder=folder, crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)
        
        return self._Mesh_Get_Mesh()
    
    def Mesh_Revolve(self, contour: _Geom, inclusions: list[_Geom]=[],
                     axis: Line=Line(Point(), Point(0,1)), angle=360, layers:list[int]=[30], elemType=ElemType.TETRA4,
                     cracks: list[_Geom]=[], refineGeoms: list[Union[_Geom,str]]=[],
                     isOrganised=False, additionalSurfaces:list[_Geom]=[],
                     additionalLines:list[Union[Line,CircleArc]]=[], additionalPoints: list[Point]=[], folder="") -> Mesh:
        """Creates a 3D mesh by rotating a surface along an axis.

        Parameters
        ----------
        contour : _Geom
            geometry that builds the contour
        inclusions : list[Domain, Circle, Points, Contour], optional
            list of hollow and filled geom objects inside the domain
        axis : Line, optional
            revolution axis, by default Line(Point(), Point(0,1))
        angle : float|int, optional
            revolution angle in [deg], by default 360
        layers: list[int], optional
            layers in extrusion, by default [30]
        elemType : ElemType, optional
            element type, by default "TETRA4" ["TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"]
        cracks : list[Line | Points | Countour]
            list of geom object used to create open or closed cracks 
        refineGeoms : list[Domain|Circle|str], optional
            list of geom object for mesh refinement, by default []
        isOrganised : bool, optional
            mesh is organized, by default False
        additionalSurfaces : list[_Geom]
            additional surfaces that will be added to or removed from the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). Tip: if the mesh is not well generated, you can also give the inclusions.
        additionalLines : list[Union[Line,CircleArc]]
            additional lines that will be added to the surfaces created by the contour and the inclusions. (e.g Domain, Circle, Contour, Points). WARNING: lines must be within the domain.
        additionalPoints : list[Point]
            additional points that will be added to the surfaces created by the contour and the inclusions. WARNING: points must be within the domain.
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh

        Returns
        -------
        Mesh
            Created mesh
        """

        # this function only work for occ factory (# .fragment() & .getEntities())
        self._Init_gmsh()
        self.__CheckType(3, elemType)
        
        tic = Tic()
        
        factory = self._factory

        self._Surfaces(contour, inclusions, elemType, isOrganised)
        self._Additional_Surfaces(2, additionalSurfaces, elemType, isOrganised)
        self._Additional_Lines(2, additionalLines)
        self._Additional_Points(2, additionalPoints)

        # get created surfaces
        surfaces = [entity[1] for entity in factory.getEntities(2)]
        self._Surfaces_Organize(surfaces, elemType, isOrganised)
        
        self._Revolve(surfaces=surfaces, axis=axis, angle=angle, elemType=elemType, layers=layers)

        # get 3D entities
        entities_3D = factory.getEntities(3)

        # create crack
        crackLines, crackSurfaces, openPoints, openLines = self._Cracks_SetPhysicalGroups(cracks, entities_3D)

        self._Mesh_Refine(refineGeoms, contour.meshSize)

        self._Set_PhysicalGroups()

        tic.Tac("Mesh","Geometry", self.__verbosity)

        self._Mesh_Generate(3, elemType, folder=folder, crackLines=crackLines, crackSurfaces=crackSurfaces, openPoints=openPoints, openLines=openLines)

        return self._Mesh_Get_Mesh()
    
    def Create_posFile(self, coord: np.ndarray, values: np.ndarray, folder: str, filename="data") -> str:
        """Creates of a *.pos file that can be used to refine a mesh in a zone.

        Parameters
        ----------
        coord : np.ndarray
            coordinates of values
        values : np.ndarray
            scalar nodes values
        folder : str
            save folder
        filename : str, optional
            *.pos file name, by default "data".

        Returns
        -------
        str
            Returns the path to the created *.pos file
        """

        assert isinstance(coord, np.ndarray), "Must be a numpy array"
        assert coord.shape[1] == 3, "Must be of dimension (n, 3)"

        assert values.shape[0] == coord.shape[0], "values and coordo must be get the same number of lines"

        data = np.append(coord, values.reshape(-1, 1), axis=1)

        self._Init_gmsh()

        view = gmsh.view.add("scalar points")

        gmsh.view.addListData(view, "SP", coord.shape[0], data.ravel())

        path = Folder.Join(folder, f"{filename}.pos", mkdir=True)

        gmsh.view.write(view, path)

        return path
    
    def Set_meshSize(self, meshSize:float) -> None:
        """Sets the mesh size"""
        self._Synchronize() # mandatory
        gmsh.model.mesh.setSize(self._factory.getEntities(0), meshSize)
    
    def _Mesh_Refine(self, refineGeoms: list[Union[Domain,Circle,str]], meshSize: float, extrude=[0,0,1]) -> None:
        """Sets a background mesh

        Parameters
        ----------
        refineGeoms : list[Domain|Circle|str]
            Geometric objects to refine de background mesh
        meshSize : float
            size of elements outside the domain
        extrude : list[float]
            extrusion vector, by default [0,0,1]
        """

        # See t10.py in the gmsh tutorials
        # https://gitlab.onelab.info/gmsh/gmsh/blob/master/tutorials/python/t10.py

        assert isinstance(refineGeoms, Iterable), "refineGeoms must be a list of geometric objects."

        if refineGeoms is None or len(refineGeoms) == 0: return
        
        fields = list(gmsh.model.mesh.field.list())

        for geom in refineGeoms:

            if isinstance(geom, Domain):

                coordo = np.array([point.coord  for point in geom.points])

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
                gmsh.model.mesh.field.setNumber(field, "XAxis", extrude[0])
                gmsh.model.mesh.field.setNumber(field, "YAxis", extrude[1])
                gmsh.model.mesh.field.setNumber(field, "ZAxis", extrude[2])

            elif isinstance(geom, str):

                if not Folder.Exists(geom) :
                    print("The .pos file does not exist.")
                    continue

                if ".pos" not in geom:
                    print("Must provide a .pos file")
                    continue

                gmsh.merge(geom)

                # Add the post-processing view as a new size field:
                field = gmsh.model.mesh.field.add("PostView")
                # gmsh.model.mesh.field.setNumber(field, "ViewIndex", 0)
                # gmsh.model.mesh.field.setNumber(field, "UseClosest", 0)

            elif geom is None:
                continue

            else:
                print("refineGeoms must be of type Domain, Circle, str(.pos file)")
            
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
    def _Set_mesh_order(elemType: str) -> None:
        """Sets the mesh order"""
        if elemType in ["TRI3","QUAD4"]:
            gmsh.model.mesh.set_order(1)
        elif elemType in ["SEG3", "TRI6", "QUAD8", "QUAD9", "TETRA10", "HEXA20", "HEXA27", "PRISM15", "PRISM18"]:
            if elemType in ["QUAD8", "HEXA20", "PRISM15"]:
                gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
            gmsh.model.mesh.set_order(2)
        elif elemType in ["SEG4", "TRI10"]:
            gmsh.model.mesh.set_order(3)
        elif elemType in ["SEG5", "TRI15"]:
            gmsh.model.mesh.set_order(4)

    def _Set_mesh_algorithm(self, elemType: ElemType) -> None:
        """Sets the mesh algorithm.\n
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

        if elemType in ElemType.Get_1D() or elemType in ElemType.Get_2D():
            meshOption = "Mesh.Algorithm"
            meshAlgorithm = 6 # 6: Frontal-Delaunay
        elif elemType in ElemType.Get_3D():
            meshOption = "Mesh.Algorithm3D"
            meshAlgorithm = 1 # 1: Delaunay
        
        gmsh.option.setNumber(meshOption, meshAlgorithm)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)

    def _Mesh_Generate(self, dim: int, elemType: str,
                 crackLines:int=None, crackSurfaces:int=None, openPoints:int=None, openLines:int=None,
                 folder="", filename="mesh") -> None:
        """Generates the mesh with the available created entities.

        Parameters
        ----------
        dim : int
            mesh dimension (e.g 1, 2, 3)
        elemType : str
            element type
        crackLines : int, optional
            physical group for crack lines (associated with openPoints), by default None
        crackSurfaces : int, optional
            physical group for crack surfaces (associated with openLines), by default None
        openPoints: int, optional
            physical group for open points, by default None
        openLines : int, optional
            physical group for open lines, by default None
        folder : str, optional
            default mesh.msh folder, by default "" does not save the mesh
        filename : str, optional
            mesh saving file as filename.msh, by default mesh
        """
        
        self._Set_mesh_algorithm(elemType)
        # self._factory.removeAllDuplicates()
        self._Synchronize() # mandatory

        tic = Tic()

        gmsh.model.mesh.generate(dim)
        
        # set mesh order
        Mesher._Set_mesh_order(elemType)

        if dim > 1:
            # remove all duplicated nodes and elements
            gmsh.model.mesh.removeDuplicateNodes()
            gmsh.model.mesh.removeDuplicateElements()

        # PLUGIN CRACK
        if crackSurfaces != None or crackLines != None:

            if crackLines != None: # 1D CRACKS
                gmsh.plugin.setNumber("Crack", "Dimension", 1)
                gmsh.plugin.setNumber("Crack", "PhysicalGroup", crackLines)
                if openPoints != None:
                    gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openPoints)
                gmsh.plugin.run("Crack") # DONT DELETE must be called for lines and surfaces
            
            if crackSurfaces != None: # 2D CRACKS                
                gmsh.plugin.setNumber("Crack", "Dimension", 2)
                gmsh.plugin.setNumber("Crack", "PhysicalGroup", crackSurfaces)
                if openLines != None:
                    gmsh.plugin.setNumber("Crack", "OpenBoundaryPhysicalGroup", openLines)
                gmsh.plugin.run("Crack")
        
        # Open gmsh interface if necessary
        if '-nopopup' not in sys.argv and self.__openGmsh:
            gmsh.fltk.run()
        
        tic.Tac("Mesh","Meshing with gmsh", self.__verbosity)

        if folder != "":
            # gmsh.write(Dossier.Join([folder, "model.geo"])) # It doesn't seem to work, but that's okay
            self._Synchronize()

            if not Folder.Exists(folder):
                os.makedirs(folder)
            msh = Folder.Join(folder, f"{filename}.msh")
            gmsh.write(msh)
            tic.Tac("Mesh","Saving .msh", self.__verbosity)

    def _Mesh_Get_Mesh(self, coef=1) -> Mesh:
        """Creates the mesh object from the created gmsh mesh."""
        
        tic = Tic()

        dict_groupElem: dict[ElemType, _GroupElem] = {}
        meshDim = gmsh.model.getDimension()
        elementTypes = gmsh.model.mesh.getElementTypes()
        nodes, coord, __ = gmsh.model.mesh.getNodes()
        
        nodes = np.array(nodes, dtype=int) - 1 # node numbers
        Nn = nodes.shape[0] # Number of nodes

        # Old method was boggling
        # The bugs have been fixed because I didn't properly organize the nodes when I created them
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1926
        # thats why there is now 'changes' array because jumps in nodes may append when there is a open crack in the mesh

        # Organize nodes from smallest to largest
        sortedIdx = np.argsort(nodes)
        sortedNodes = nodes[sortedIdx]

        # Here we will detect jumps in node numbering
        # Example nodes = [0 1 2 3 4 5 6 8]
        
        # Here we will detect the jump between 6 and 8.
        # diff = [0 0 0 0 0 0 0 1]
        diff = sortedNodes - np.arange(Nn)

        # Array that stores the changes
        # For example below -> Changes = [0 1 2 3 4 5 6 0 7]
        # changes is used such correctedNodes = changes[nodes]
        changes = np.zeros(nodes.max()+1, dtype=int)
        changes[sortedNodes] = sortedNodes - diff

        # The coordinate matrix of all nodes used in the mesh is constructed        
        coord: np.ndarray = coord.reshape(-1,3)[sortedIdx,:]

        # Apply coef to scale the coordinates
        coord = coord * coef

        knownDims = [] # known dimensions in the mesh
        # For each element type
        for gmshId in elementTypes:
                                        
            # get element numbers and connection matrix
            elementTags, nodeTags = gmsh.model.mesh.getElementsByType(gmshId)
            elementTags = np.asarray(elementTags, dtype=int) - 1 # tags for each elements
            nodeTags = np.asarray(nodeTags, dtype=int) - 1 # connection matrix in shape (e * nPe)

            nodeTags: np.ndarray = changes[nodeTags] # Apply changes to correct jumps in nodes
            
            # Elements
            Ne = elementTags.shape[0] # number of elements
            nPe = GroupElemFactory.Get_ElemInFos(gmshId)[1] # nodes per element
            connect: np.ndarray = nodeTags.reshape(Ne, nPe) # creates connect matrix

            # Nodes
            nodes = np.asarray(list(set(nodeTags)), dtype=int)
            Nmax = nodes.max() # checks that max node numbers can be reached in coord
            assert Nmax <= (coord.shape[0]-1), f"Nodes {Nmax} doesn't exist in coord"

            # Element group creation
            groupElem = GroupElemFactory.Create(gmshId, connect, coord, nodes)
            
            # We add the element group to the dictionary containing all groups
            dict_groupElem[groupElem.elemType] = groupElem
            
            # Check that the mesh does not have a group of elements of this dimension
            if groupElem.dim in knownDims and groupElem.dim == meshDim:
                recoElement = 'Triangular' if meshDim == 2 else 'Tetrahedron'
                raise Exception(f"Importing the mesh from gmsh is impossible because several {meshDim}D elements have been detected.\nTry out with {recoElement} elements.\n You can also try to reduce the mesh size")
                # TODO make it work ?
                # Can be complicated especially in the creation of elemental matrices and assembly
                # Not impossible but not trivial
                # Restart the procedure if it doesn't work?
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
                groupElem._Set_Nodes_Tag(nodesGroup, name)
                groupElem._Set_Elements_Tag(nodesGroup, name)

            [__addPysicalGroup(group) for group in physicalGroups]
        
        tic.Tac("Mesh","Construct mesh object", self.__verbosity)

        gmsh.finalize()

        mesh = Mesh(dict_groupElem, self.__verbosity)

        return mesh
    
    @staticmethod
    def _Construct_2D_meshes(L=10, h=10, meshSize=3) -> list[Mesh]:
        """Creates 2D meshes."""

        mesher = Mesher(openGmsh=False, verbosity=False)

        list_mesh2D = []
        
        domain = Domain(Point(0,0,0), Point(L, h, 0), meshSize=meshSize)
        line = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=meshSize, isOpen=False)
        lineOpen = Line(Point(x=0, y=h/2, isOpen=True), Point(x=L/2, y=h/2), meshSize=meshSize, isOpen=True)
        circle = Circle(Point(x=L/2, y=h/2), L/3, meshSize=meshSize, isHollow=True)
        circleClose = Circle(Point(x=L/2, y=h/2), L/3, meshSize=meshSize, isHollow=False)

        domain_area = L*h

        def testArea(area):
            assert np.abs(domain_area-area)/domain_area <= 1e-10, "Incorrect surface"

        # For each type of 2D element
        for elemType in ElemType.Get_2D():

            print(elemType)

            mesh1 = mesher.Mesh_2D(domain, elemType=elemType, isOrganised=False)
            testArea(mesh1.area)
            
            mesh2 = mesher.Mesh_2D(domain, elemType=elemType, isOrganised=True)
            testArea(mesh2.area)

            mesh3 = mesher.Mesh_2D(domain, [circle], elemType)
            # Here we don't check because there are too few elements to properly represent the hole

            mesh4 = mesher.Mesh_2D(domain, [circleClose], elemType)
            testArea(mesh4.area)

            mesh5 = mesher.Mesh_2D(domain, cracks=[line], elemType=elemType)
            testArea(mesh5.area)

            mesh6 = mesher.Mesh_2D(domain, cracks=[lineOpen], elemType=elemType)
            testArea(mesh6.area)

            for m in [mesh1, mesh2, mesh3, mesh4, mesh5, mesh6]:
                list_mesh2D.append(m)
        
        return list_mesh2D

    @staticmethod
    def _Construct_3D_meshes(L=130, h=13, b=13, meshSize=7.5, useImport3D=False) -> list[Mesh]:
        """Creates 3D meshes."""        

        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=meshSize)
        circleCreux = Circle(Point(x=L/2, y=0,z=-b/2), h*0.7, meshSize=meshSize, isHollow=True)
        circle = Circle(Point(x=L/2, y=0 ,z=-b/2), h*0.7, meshSize=meshSize, isHollow=False)
        axis = Line(domain.pt1+[-1,0], domain.pt1+[-1,h])

        volume = L*h*b

        def testVolume(val):
            assert np.abs(volume-val)/volume <= 1e-10, "Incorrect volume"
        
        partPath = Folder.Join(Folder.EASYFEA_DIR, "examples","_parts","beam.stp")

        mesher = Mesher()

        list_mesh3D = []
        # For each type of 3D element
        for elemType in ElemType.Get_3D():
            
            if useImport3D and elemType in [ElemType.TETRA4,ElemType.TETRA10]:
                meshPart = mesher.Mesh_Import_part(partPath, 3, meshSize=meshSize, elemType=elemType)
                list_mesh3D.append(meshPart)

            mesh1 = mesher.Mesh_Extrude(domain, [], [0,0,-b], [3], elemType=elemType)
            list_mesh3D.append(mesh1)
            testVolume(mesh1.volume)

            mesh2 = mesher.Mesh_Extrude(domain, [circleCreux], [0,0,-b], [3], elemType)
            list_mesh3D.append(mesh2)

            mesh3 = mesher.Mesh_Extrude(domain, [circle], [0,0,-b], [3], elemType)
            list_mesh3D.append(mesh3)
            testVolume(mesh3.volume)

        return list_mesh3D
    
    def Save_Simu(self, simu, results: list[str]=[], details=False,
                  edgeColor='black', plotMesh=True, showAxes=False, folder: str="") -> None:
        """Save the simulation in gmsh.pos format using gmsh.view

        Parameters
        ----------
        simu : _Simu
            simulation
        results : list[str], optional
            list of result you want to plot, by default []
        details : bool, optional
            get default result values with details or not see `simu.Results_nodesField_elementsField(details)`, by default False
        edgeColor : str, optional
            color used to plot the edges, by default 'black'
        plotMesh : bool, optional
            plot the mesh, by default True
        showAxes : bool, optional
            show the axes, by default False
        folder : str, optional
            folder used to save .pos file, by default ""
        """

        from ..Simulations import _Simu
        assert isinstance(simu, _Simu), 'simu must be a simu object'
        
        assert isinstance(results, list), 'results must be a list'
        
        self._Init_gmsh()

        def getColor(c:str):
            """transform matplotlib color to rgb"""
            rgb = np.asarray(matplotlib.colors.to_rgb(edgeColor)) * 255
            rgb = np.asarray(rgb, dtype=int)
            return rgb

        def reshape(values: np.ndarray):
            """reshape values to get them at the corners of the elements"""
            values_n = np.reshape(values, (mesh.Nn, -1))
            values_e: np.ndarray = values_n[connect_e]
            if len(values_e.shape) == 3:
                values_e = np.transpose(values_e, (0,2,1))
            return values_e.reshape((mesh.Ne, -1))
        
        mesh = simu.mesh        
        Ne = mesh.Ne
        nbCorners = mesh.groupElem.nbCorners # do this because it is not working for quadratic elements
        connect_e = mesh.connect[:,:nbCorners]
        elements_e = reshape(mesh.coord)

        def types(elemType: str):
            """get gmsh type associated with elemType"""
            if 'POINT' in elemType:
                return 'P'
            elif 'SEG' in elemType:
                return 'L'
            elif 'TRI' in elemType:
                return 'T'
            elif 'QUAD' in elemType:
                return 'Q'
            elif 'TETRA' in elemType:
                return 'S'
            elif 'HEXA' in elemType:
                return 'H'
            elif 'PRISM' in elemType:
                return 'I'
            elif 'PYRA' in elemType:
                return 'Y'
        
        gmshType = types(mesh.elemType)
        colorElems = getColor(edgeColor)

        # get nodes and elements field to plot
        nodesField, elementsField = simu.Results_nodesField_elementsField()
        [results.append(result) for result in (nodesField + elementsField) if result not in results]

        dict_results: dict[str, list[np.ndarray]] = {result: [] for result in results}

        # activates the first iteration
        simu.Set_Iter(0, resetAll=True)

        # activates the first iteration
        simu.Set_Iter(0, resetAll=True)

        for i in range(simu.Niter):
            simu.Set_Iter(i)
            [dict_results[result].append(reshape(simu.Result(result))) for result in results]
            
        def AddView(name: str, values_e: np.ndarray):
            """Add a view"""

            if name == 'displacement_matrix_0':
                name='ux'
            elif name == 'displacement_matrix_1':
                name='uy'
            elif name == 'displacement_matrix_2':
                name='uz'                

            view = gmsh.view.add(name)
            
            gmsh.view.option.setNumber(view, "IntervalsType", 3)
            # (1: iso, 2: continuous, 3: discrete, 4: numeric)
            gmsh.view.option.setNumber(view, "NbIso", 10)

            if plotMesh:
                gmsh.view.option.setNumber(view, "ShowElement", 1)

            if showAxes:
                gmsh.view.option.setNumber(view, "Axes", 1)
                # (0: none, 1: simple axes, 2: box, 3: full grid, 4: open grid, 5: ruler)
            
            gmsh.view.option.setColor(view, 'Lines', *colorElems)
            gmsh.view.option.setColor(view, 'Triangles', *colorElems)
            gmsh.view.option.setColor(view, 'Quadrangles', *colorElems)
            gmsh.view.option.setColor(view, 'Tetrahedra', *colorElems)
            gmsh.view.option.setColor(view, 'Hexahedra', *colorElems)
            gmsh.view.option.setColor(view, 'Pyramids', *colorElems)
            gmsh.view.option.setColor(view, 'Prisms', *colorElems)

            # S for scalar, V for vector, T
            if values_e.shape[1] == nbCorners:
                vt = 'S'
            else:
                vt = 'S'

            res = np.concatenate((elements_e, values_e), 1)

            gmsh.view.addListData(view, vt+gmshType, Ne, res.ravel())
            
            if folder != "":
                gmsh.view.write(view, Folder.Join(folder, "simu.pos"), True)

            return view

        for result in dict_results.keys():

            nIter = len(dict_results[result])

            if nIter == 0: continue

            dof_n = dict_results[result][0].shape[-1] // nbCorners

            vals_e_i_n = np.concatenate(dict_results[result], 1).reshape((Ne, nIter, dof_n, -1))

            if dof_n == 1:
                view = AddView(result, vals_e_i_n[:,:,0].reshape((Ne,-1)))
            else:
                views = [AddView(result+f'_{n}', vals_e_i_n[:,:,n].reshape(Ne,-1)) for n in range(dof_n)]

        # Launch the GUI to see the results:
        if '-nopopup' not in sys.argv and self.__openGmsh:
            gmsh.fltk.run()

        gmsh.finalize()