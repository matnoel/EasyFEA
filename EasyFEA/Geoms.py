# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the geometric classes used to build meshes."""

from typing import Union
from collections.abc import Iterable
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from abc import ABC, abstractmethod

# ----------------------------------------------
# Geom objects
# ----------------------------------------------

class Point:

    def __init__(self, x=0.0, y=0.0, z=0.0, isOpen=False, r=0.0):
        """Creates a point.

        Parameters
        ----------
        x : float, optional
            x coordinate, default 0.0
        y : float, optional
            y coordinate, default 0.0
        z : float, optional
            z coordinate, default 0.0
        isOpen : bool, optional
            point can open (openCrack), default False
        r : float, optional
            radius used for fillet
        """
        self.r = r
        self.coord = np.array([x, y, z], dtype=float)
        self.isOpen = isOpen

    @property
    def x(self) -> float:
        """x coordinate"""
        return self.__coord[0]
    
    @x.setter
    def x(self, value) -> None:
        self.__coord[0] = value

    @property
    def y(self) -> float:
        """y coordinate"""
        return self.__coord[1]
    
    @y.setter
    def y(self, value) -> None:
        self.__coord[1] = value

    @property
    def z(self) -> float:
        """z coordinate"""
        return self.__coord[2]
    
    @z.setter
    def z(self, value) -> None:
        self.__coord[2] = value

    @property
    def r(self) -> float:
        """radius used for fillet"""
        return self.__r
    
    @r.setter
    def r(self, value) -> None:
        self.__r = value

    @property
    def coord(self) -> np.ndarray:
        """[x,y,z] coordinates"""
        return self.__coord.copy()
    
    @coord.setter
    def coord(self, value) -> None:
        coord = As_Coordinates(value)
        self.__coord = coord

    @property
    def isOpen(self) -> bool:
        """point is open"""
        return self.__isOpen
    
    @isOpen.setter
    def isOpen(self, value: bool) -> None:
        assert isinstance(value, bool)
        self.__isOpen = value
    
    def Check(self, coord) -> bool:
        """Checks if coordinates are identical"""
        coord = As_Coordinates(coord)
        n = np.linalg.norm(self.coord)
        n = 1 if n == 0 else n
        diff = np.linalg.norm(self.coord - coord)/n
        return diff <= 1e-12
    
    def Translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translates the point."""
        self.__coord = Translate_coord(self.__coord, dx, dy, dz).ravel()

    def Rotate(self, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> None:
        """Rotates the point with around an axis.

        Parameters
        ----------
        theta : float
            rotation angle [deg] 
        center : tuple, optional
            rotation center, by default (0,0,0)
        direction : tuple, optional
            rotation direction, by default (0,0,1)
        """
        self.__coord = Rotate_coord(self.__coord, theta, center, direction).ravel()

    def Symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrizes the point coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """
        self.__coord = Symmetry_coord(self.__coord, point, n).ravel()
    
    def __radd__(self, value):
        return self.__add__(value)

    def __add__(self, value):
        coord = As_Coordinates(value)        
        newCoord: np.ndarray = self.coord + coord
        return Point(*newCoord)

    def __rsub__(self, value):
        return self.__add__(value)
    
    def __sub__(self, value):
        coord = As_Coordinates(value)        
        newCoord: np.ndarray = self.coord - coord
        return Point(*newCoord)
    
    def __rmul__(self, value):
        return self.__mul__(value)

    def __mul__(self, value):
        coord = As_Coordinates(value)
        newCoord: np.ndarray = self.coord * coord
        return Point(*newCoord)
    
    def __rtruediv__(self, value):
        return self.__truediv__(value)

    def __truediv__(self, value):
        coord = As_Coordinates(value)
        newCoord: np.ndarray = self.coord / coord
        return Point(*newCoord)
    
    def __rfloordiv__(self, value):
        return self.__floordiv__(value)

    def __floordiv__(self, value):
        coord = As_Coordinates(value)
        newCoord: np.ndarray = self.coord // coord
        return Point(*newCoord)
    
    def copy(self):
        return copy.deepcopy(self)

class _Geom(ABC):

    def __init__(self, points: list[Point], meshSize: float, name: str, isHollow: bool, isOpen: bool):
        """Creates a geometric object.

        Parameters
        ----------
        points : list[Point]
            list of points to build the geometric object
        meshSize : float
            mesh size that will be used to create the mesh >= 0
        name : str
            object name
        isHollow : bool
            Indicates whether the the formed domain is hollow/empty
        isOpen : bool
            Indicates whether the object can open to represent an open crack (openCrack)
        """        
        self.meshSize: float = meshSize
        assert isinstance(points, Iterable) and isinstance(points[0], Point), "points must be a list of points."
        self.__points: list[Point] = points
        assert isinstance(name, str), "must be a string"
        self.__name = name
        assert isinstance(isHollow, bool), "must be a boolean"
        self.__isHollow: bool = isHollow
        assert isinstance(isOpen, bool), "must be a boolean"
        self.__isOpen: bool = isOpen

    @property
    def meshSize(self) -> float:
        """element size used for meshing"""
        return self.__meshSize
    
    @meshSize.setter
    def meshSize(self, value) -> None:        
        assert value >= 0, "meshSize must be >= 0"
        self.__meshSize = value

    # points doesn't have a public setter for safety
    @property
    def points(self) -> list[Point]:
        """Points used to build the object."""
        return self.__points
    
    @property
    def coord(self) -> np.ndarray:
        return np.asarray([p.coord for p in self.points])
    
    @abstractmethod
    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:
        """Returns coordinates for constructing lines and points."""
        lines = self.coord
        points = lines[[0,-1]]
        return lines, points    
    
    def copy(self):
        new = copy.deepcopy(self)
        new.name = new.name +'_copy'        
        return new

    @property
    def name(self) -> str:
        """object name"""
        return self.__name
    
    @name.setter
    def name(self, val: str) -> None:
        assert isinstance(val, str), "must be a string"
        self.__name = val
    
    @property
    def isHollow(self) -> bool:
        """Indicates whether the the formed domain is hollow/filled."""
        return self.__isHollow
    
    @isHollow.setter
    def isHollow(self, value: bool) -> None:
        assert isinstance(value, bool), "must be a boolean"
        self.__isHollow = value
    
    @property
    def isOpen(self) -> bool:
        """Indicates whether the object can open to represent an open crack."""
        return self.__isOpen
    
    @isOpen.setter
    def isOpen(self, value: bool) -> None:
        assert isinstance(value, bool), "must be a boolean"
        self.__isOpen = value
    
    def Translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translates the object."""
        # to translate an object, all you have to do is move these points
        [p.Translate(dx,dy,dz) for p in self.__points]
    
    def Rotate(self, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> None:        
        """Rotates the object coordinates around an axis.

        Parameters
        ----------        
        theta : float
            rotation angle in [deg] 
        center : tuple, optional
            rotation center, by default (0,0,0)
        direction : tuple, optional
            rotation direction, by default (0,0,1)
        """
        oldCoord = self.coord        
        newCoord = Rotate_coord(oldCoord, theta, center, direction)

        dec = newCoord - oldCoord
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]

    def Symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrizes the object coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """

        oldCoord = self.coord
        newCoord = Symmetry_coord(oldCoord, point, n)

        dec = newCoord - oldCoord
        [point.Translate(*dec[p]) for p, point in enumerate(self.points)]

    def Plot(self, ax: plt.Axes=None, color:str="", name:str="", lw=None, ls=None, plotPoints=True) -> plt.Axes:

        from .utilities import Display

        if ax is None:
            ax = Display.Init_Axes(3)
            
        inDim = 3 if ax.name == '3d' else 2

        name = self.name if name == "" else name

        lines, points = self.Get_coord_for_plot()
        if color != "":
            ax.plot(*lines[:,:inDim].T, color=color, label=name, lw=lw, ls=ls)
        else:
            ax.plot(*lines[:,:inDim].T, label=name, lw=lw, ls=ls)
        if plotPoints:
            ax.plot(*points[:,:inDim].T, ls='', marker='.',c='black')

        if inDim == 3:
            xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
            oldBounds = np.array([xlim, ylim, zlim]).T
            lines = np.concatenate((lines, oldBounds), 0)
            Display._Axis_equal_3D(ax, lines)
        else:
            ax.axis('equal')

        return ax
    
    @staticmethod
    def Plot_Geoms(geoms: list, ax: plt.Axes=None,
                   color:str="", name:str="", plotPoints=True, plotLegend=True) -> plt.Axes:
        geoms: list[_Geom] = geoms
        for g, geom in enumerate(geoms):
            if isinstance(geom, Point): continue
            if ax is None:
                ax = geom.Plot(color=color, name=name, plotPoints=plotPoints)
            else:
                geom.Plot(ax, color, name, plotPoints=plotPoints)

        if plotLegend:
            ax.legend()

        return ax

class Points(_Geom):

    __nbPoints = 0

    def __init__(self, points: list[Point], meshSize=0.0, isHollow=True, isOpen=False):
        """Creates points (list of point).\n
        Can be used to construct a closed surface or a spline.

        Parameters
        ----------
        points : list[Point]
            list of points
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            the formed domain is hollow/empty, by default True
        isOpen : bool, optional
            the spline formed by the points list can be opened (openCrack), by default False
        """

        assert isinstance(points, Iterable) and isinstance(points[0], Point), "points must be a list of points."

        self.pt1 = points[0]
        """First point"""
        self.pt2 = points[-1]
        """Last point"""

        Points.__nbPoints += 1
        name = f"Points{Points.__nbPoints}"
        super().__init__(points, meshSize, name, isHollow, isOpen)

    def Get_Contour(self):
        """Creates a contour from the points.\n
        Creates a fillet if a point has a radius which is not 0."""

        coordinates = self.coord
        N = coordinates.shape[0]
        mS = self.meshSize

        # TODO Allows the addition of chamfers?

        # Get corners
        corners: list[_Geom] = []
        geoms: list[_Geom] = []

        def Link(idx1: int, idx2: int):
            # this function makes the link between corners[idx1] and corners[idx2]
            
            # get the last point associated with idx1
            if isinstance(corners[idx1], Point):
                p1 = corners[idx1]
            else:
                # get the last coordinates
                p1 = corners[idx1].points[-1]

            # get the first point associated with idx2
            if isinstance(corners[idx2], Point):
                p2 = corners[idx2]
            else:
                # get the first coordinates
                p2 = corners[idx2].points[0]                
            
            if not p1.Check(p2):
                line = Line(p1, p2, mS, self.isOpen)
                geoms.append(line)

            if isinstance(corners[-1], (CircleArc, Line)) and idx2 != 0:
                geoms.append(corners[-1])

        for p, point in enumerate(self.points):

            prev = p-1
            next = p+1 if p+1 < N else 0

            isOpen = point.isOpen

            if point.r == 0:

                corners.append(point)

            else:
                A, B, C = Points_Rayon(point.coord, coordinates[prev], coordinates[next], point.r)

                pA = Point(*A, isOpen)
                pB = Point(*B, isOpen)
                pC = Point(*C, isOpen)

                corners.append(CircleArc(pA, pB, pC, meshSize=mS))
            
            if p > 0:
                Link(-2, -1)
            elif isinstance(corners[-1], (CircleArc, Line)):
                geoms.append(corners[-1])
                
        Link(-1, 0)

        contour = Contour(geoms, self.isHollow, self.isOpen).copy()
        contour.name = self.name + '_contour'
        # do the copy to unlink the points connexion with the list of points
        
        return contour

    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:
        return super().Get_coord_for_plot()
    
    @property
    def length(self) -> float:
        coord = self.coord
        length = np.linalg.norm(coord[1:]-coord[:-1], axis=1)
        length = np.sum(length)
        return length

class Line(_Geom):

    __nbLine = 0

    @staticmethod
    def distance(pt1: Point, pt2: Point) -> float:
        """Computes the distance between two points."""
        length = np.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2 + (pt1.z-pt2.z)**2)
        return np.abs(length)
    
    @staticmethod
    def get_unitVector(pt1: Point, pt2: Point) -> np.ndarray:
        """Creates the unit vector between two points."""
        length = Line.distance(pt1, pt2)        
        v = np.array([pt2.x-pt1.x, pt2.y-pt1.y, pt2.z-pt1.z])/length
        return v   

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isOpen=False):
        """Creates a line.

        Parameters
        ----------
        pt1 : Point
            first point
        pt2 : Point
            second point
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isOpen : bool, optional
            line can be opened (openCrack), by default False
        """

        assert isinstance(pt1, Point), "must be a point"
        self.pt1 = pt1
        assert isinstance(pt2, Point), "must be a point"
        self.pt2 = pt2

        Line.__nbLine += 1
        name = f"Line{Line.__nbLine}"
        _Geom.__init__(self, [pt1, pt2], meshSize, name, False, isOpen)
    
    @property
    def unitVector(self) -> np.ndarray:
        """The unit vector for the two points on the line (p2-p1)"""
        return Line.get_unitVector(self.pt1, self.pt2)

    @property
    def length(self) -> float:
        """distance between the two points of the line"""
        return Line.distance(self.pt1, self.pt2)
    
    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:
        return super().Get_coord_for_plot()

class Domain(_Geom):

    __nbDomain = 0

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isHollow=True):
        """Creates a 2d or 3d domain.

        Parameters
        ----------
        pt1 : Point
            first point
        pt2 : Point
            second point
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            the formed domain is hollow/empty, by default True
        """

        assert isinstance(pt1, Point), "must be a point"
        self.pt1 = pt1
        assert isinstance(pt2, Point), "must be a point"
        self.pt2 = pt2

        Domain.__nbDomain += 1
        name = f"Domain{Domain.__nbDomain}"
        # a domain can't be open
        _Geom.__init__(self, [pt1, pt2], meshSize, name, isHollow, False)

    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:

        p1 = self.pt1.coord
        p7 = self.pt2.coord

        dx, dy, dz = p7 - p1

        p2 = p1 + [dx,0,0]
        p3 = p1 + [dx,dy,0]
        p4 = p1 + [0,dy,0]
        p5 = p1 + [0,0,dz]
        p6 = p1 + [dx,0,dz]
        p8 = p1 + [0,dy,dz]

        lines = np.concatenate((p1,p2,p3,p4,p1,p5,p6,p2,p6,p7,p3,p7,p8,p4,p8,p5)).reshape((-1,3))

        points = np.concatenate((p1,p7)).reshape((-1,3))

        return lines, points

class Circle(_Geom):

    __nbCircle = 0

    def __init__(self, center: Point, diam: float, meshSize=0.0, isHollow=True, isOpen=False, n=(0,0,1)):
        """Creates a circle according to its center, diameter and the normal vector.

        Parameters
        ----------
        center : Point
            center of circle
        diam : float
            diameter
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            circle is hollow/empty, by default True
        isOpen : bool, optional
            circle can be opened (openCrack), by default False
        n : tuple, optional
            normal direction to the circle, by default (0,0,1)
        """
        
        assert diam > 0.0
        assert isinstance(center, Point), "must be a point"

        r = diam/2        

        # creates points associated with the circle
        self.center = center
        self.pt1 = center + [r, 0, 0]
        self.pt2 = center + [0, r, 0]
        self.pt3 = center + [-r, 0, 0]
        self.pt4 = center + [0, -r, 0]
        # creates circle arcs associated with the circle
        circleArc1 = CircleArc(self.pt1, self.pt2, center=center, meshSize=meshSize, isOpen=isOpen)
        circleArc2 = CircleArc(self.pt2, self.pt3, center=center, meshSize=meshSize, isOpen=isOpen)
        circleArc3 = CircleArc(self.pt3, self.pt4, center=center, meshSize=meshSize, isOpen=isOpen)
        circleArc4 = CircleArc(self.pt4, self.pt1, center=center, meshSize=meshSize, isOpen=isOpen)
        # creates the contour object associated with the circle
        self.contour = Contour([circleArc1, circleArc2, circleArc3, circleArc4], isHollow, isOpen)

        Circle.__nbCircle += 1
        name = f"Circle{Circle.__nbCircle}"
        _Geom.__init__(self, [center, self.pt1, self.pt2, self.pt3, self.pt4], meshSize, name, isHollow, isOpen)

        # rotate if necessary
        zAxis = np.array([0,0,1])
        n = Normalize_vect(As_Coordinates(n))
        rotAxis = np.cross(n, zAxis)
        # theta = AngleBetween_a_b(zAxis, n)
        
        # then we rotate along i
        if np.linalg.norm(rotAxis) == 0:
            # n and zAxis are collinear
            i = Normalize_vect((self.pt1 - center).coord) # i = p1 - center
        else:
            i = rotAxis

        mat = Jacobian_Matrix(i,n)

        coord = np.einsum('ij,nj->ni', mat, self.coord - center.coord) + center.coord

        for p, point in enumerate(self.points):
            point.coord = coord[p]

    @property
    def diam(self) -> float:
        """circle's diameter"""
        p1 = self.pt1.coord
        pC = self.center.coord
        return np.linalg.norm(p1-pC) * 2

    @property
    def n(self) -> np.ndarray:
        """axis normal to the circle"""
        i = Normalize_vect((self.pt1 - self.center).coord)
        j = Normalize_vect((self.pt2 - self.center).coord)
        n: np.ndarray = Normalize_vect(np.cross(i,j))
        return n

    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:        

        angle = np.linspace(0, np.pi*2, 40)

        pC = self.center
        R = self.diam/2

        points = self.coord
        
        lines = np.zeros((angle.size, 3))
        lines[:,0] = np.cos(angle)*R
        lines[:,1] = np.sin(angle)*R
        
        # construct jacobian matrix
        i = (self.pt1 - self.center).coord
        n = self.n
        mat = Jacobian_Matrix(i, n)

        # change base
        lines = np.einsum('ij,nj->ni', mat, lines) + pC.coord

        return lines, points[1:]
    
    @property
    def length(self) -> float:
        """circle perimeter"""
        return np.pi * self.diam

class CircleArc(_Geom):

    __nbCircleArc = 0

    def __init__(self, pt1: Point, pt2: Point, center:Point=None, R:float=None, P:Point=None, meshSize=0.0, n=(0,0,1), isOpen=False, coef=1):
        """Creates a circular arc using several methods:\n
        - 1: with 2 points, a radius R and a normal vector.\n
        - 2: with 2 points and a center.\n
        - 3: with 2 points and a point P belonging to the circle.\n
        The methods are chosen in the following order 3 2 1.\n
        This means that if you enter P, the other methods will not be used.

        Parameters
        ----------        
        pt1 : Point
            starting point
        pt2: Point
            ending point
        R: float, optional
            radius of the arc circle, by default None
        center: Point, optional
            center of circular arc, by default None
        P: Point, optional
            a point belonging to the circle, by default None
        meshSize : float, optional
            size to be used for mesh construction, by default 0.0
        n: np.ndarray | list | tuple, optional
            normal vector to the arc circle, by default (0,0,1)
        isOpen : bool, optional
            arc can be opened, by default False
        coef: int, optional
            Change direction, by default 1 or -1
        """

        assert isinstance(pt1, Point), "must be a point"
        assert isinstance(pt2, Point), "must be a point"

        # check that pt1 and pt2 dont share the same coordinates
        assert not pt1.Check(pt2), 'pt1 and pt2 are on the same coordinates'

        if P != None:
            center = Circle_Triangle(pt1, pt2, P)
            center = Point(*center)

        elif center != None:
            assert not pt1.Check(center), 'pt1 and center are on the same coordinates'

        elif R != None:            
            coord = np.array([pt1.coord, pt2.coord])
            center = Circle_Coord(coord, R, n)
            center = Point(*center)            
        else:
            raise Exception('must give P, center or R')
        
        r1 = np.linalg.norm((pt1-center).coord)
        r2 = np.linalg.norm((pt2-center).coord)
        assert (r1 - r2)**2/r2**2 <= 1e-12, "The given center doesn't have the right coordinates. If the center coordinate is difficult to identify, you can give:\n - the radius R with the vector normal to the circle n\n - another point belonging to the circle."

        self.center = center
        """Point at the center of the arc."""
        self.pt1 = pt1
        """Starting point of the arc."""
        self.pt2 = pt2
        """Ending point of the arc."""

        # Here we'll create an intermediate point, because in gmsh, circular arcs are limited to an pi angle.

        i1 = (pt1-center).coord
        i2 = (pt2-center).coord

        # construction of the passage matrix
        k = np.array([0,0,1])
        if np.linalg.norm(np.cross(i1, i2)) <= 1e-12:
            vect = Normalize_vect(i2-i1)
            i = np.cross(k,vect)
        else:
            i = Normalize_vect((i1+i2)/2)
            k = Normalize_vect(np.cross(i1, i2))
        j = np.cross(k, i)

        mat = np.array([i,j,k]).T

        # midpoint coordinates
        assert coef in [-1, 1], 'coef must be in [-1, 1]'
        pt3 = center.coord + mat @ [coef*r1,0,0]

        self.pt3 = Point(*pt3)
        """Midpoint of the circular arc."""

        self.coef = coef

        CircleArc.__nbCircleArc += 1
        name = f"CircleArc{CircleArc.__nbCircleArc}"
        _Geom.__init__(self, [pt1, center, self.pt3, pt2], meshSize, name, False, isOpen)

    @property
    def n(self) -> np.ndarray:
        """axis normal to the circle arc"""
        i = Normalize_vect((self.pt1 - self.center).coord)        
        if self.angle in [0, np.pi]:            
            j = Normalize_vect((self.pt3 - self.center).coord)
        else:
            j = Normalize_vect((self.pt2 - self.center).coord)
        n = Normalize_vect(np.cross(i,j))
        return n
    
    @property
    def angle(self):
        """circular arc angle [rad]"""
        i = (self.pt1 - self.center).coord
        j = (self.pt2 - self.center).coord
        return AngleBetween_a_b(i,j)
    
    @property
    def r(self):
        """circular arc radius"""
        return np.linalg.norm((self.pt1-self.center).coord)
    
    @property
    def length(self) -> float:
        """circular arc length"""
        return np.abs(self.angle * self.r)

    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:

        points = self.coord

        pC = self.center
        r = self.r

        # plot arc circle in 2D space
        angles = np.linspace(0, np.abs(self.angle), 11)
        lines = np.zeros((angles.size,3))
        lines[:,0] = np.cos(angles) * r
        lines[:,1] = np.sin(angles) * r

        # get the jabobian matrix
        i = (self.pt1 - self.center).coord        
        n = self.n
        
        mat = Jacobian_Matrix(i,n)

        # transform coordinates
        lines = np.einsum('ij,nj->ni', mat, lines) + pC.coord

        return lines, points[[0,-1]]

class Contour(_Geom):

    __nbContour = 0

    def __init__(self, geoms: list[Union[Line,CircleArc,Points]], isHollow=True, isOpen=False):
        """Creates a contour from a list of line circleArc and points.

        Parameters
        ----------
        geoms : list[Line | CircleArc | Points]
            list of objects used to build the contour
        isHollow : bool, optional
            the formed domain is hollow/empty, by default True
        isOpen : bool, optional
            contour can be opened, by default False
        """

        # Check that the points form a closed loop
        points: list[Point] = []

        tol = 1e-12        

        for i, geom in enumerate(geoms):

            assert isinstance(geom, (Line, CircleArc, Points)), "Must give a list of lines and arcs or points."

            if i == 0:
                ecart = tol
            elif i > 0 and i < len(geoms)-1:
                # check that the starting point has the same coordinate as the last point of the previous object
                ecart = np.linalg.norm(geom.points[0].coord - points[-1].coord)

                assert ecart <= tol, "The contour must form a closed loop."
            else:
                # check that the end point of the last geometric object is the first point created.
                ecart1 = np.linalg.norm(geom.points[0].coord - points[-1].coord)
                ecart2 = np.linalg.norm(geom.points[-1].coord - points[0].coord)

                assert ecart1 <= tol and ecart2 <= tol, "The contour must form a closed loop."

            # Add the first and last points
            points.extend([p for p in geom.points if p not in points])

        self.geoms = geoms

        Contour.__nbContour += 1
        name = f"Contour{Contour.__nbContour}"
        meshSize = np.mean([geom.meshSize for geom in geoms])
        _Geom.__init__(self, points, meshSize, name, isHollow, isOpen)

    def Get_coord_for_plot(self) -> tuple[np.ndarray,np.ndarray]:

        lines = []
        points = []

        for geom in self.geoms:
            l, p = geom.Get_coord_for_plot()
            lines.extend(l.ravel())
            points.extend(p.ravel())

        lines = np.reshape(lines, (-1,3))
        points = np.reshape(points, (-1,3))

        return lines, points
    
    @property
    def length(self) -> float:
        return np.sum([geom.length for geom in self.geoms])

# ----------------------------------------------
# Functions
# ----------------------------------------------
    
def As_Coordinates(value) -> np.ndarray:
    """Returns value as a 3D vector"""
    if isinstance(value, Point):
        coord = value.coord        
    elif isinstance(value, (list, tuple, np.ndarray)):
        val = np.asarray(value, dtype=float)
        if len(val.shape) == 2:
            assert val.shape[-1] <= 3, 'must be 3d vector or vectors'
            coord = val
        else:
            coord = np.zeros(3)
            assert val.size <= 3, 'must not exceed size 3'
            coord[:val.size] = val
    elif isinstance(value, (float, int)):            
        coord = np.asarray([value]*3)
    else:
        raise Exception(f'{type(value)} is not supported. Must be (Point | float, int list | tuple | dict | set | np.ndarray)')
    
    return coord

def Normalize_vect(vect: np.ndarray) -> np.ndarray:
    """Returns the normalized vector."""
    vect = np.asarray(vect)
    if len(vect.shape) == 1:
        return vect / np.linalg.norm(vect)
    elif len(vect.shape) == 2:
        return np.einsum('ij,i->ij',vect, 1/np.linalg.norm(vect, axis=1), optimize="optimal")
    else:
        raise Exception("The vector is the wrong size")

def Rotation_matrix(vect: np.ndarray, theta: float) -> np.ndarray:
    """Gets the rotation matrix for turning along an axis with theta angle (rad).\n
    p(x,y) = mat • p(i,j)\n
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle"""

    x, y, z = Normalize_vect(vect)
    
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    mat = np.array([[x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
                    [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
                    [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])
    
    return mat

def AngleBetween_a_b(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the angle between vectors a and b (rad).
    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors"""

    a = As_Coordinates(a)
    b = As_Coordinates(b)    
    
    ida = 'ni' if len(a.shape) == 2 else 'i'
    idb = 'ni' if len(b.shape) == 2 else 'i'
    id = 'n' if (len(a.shape) == 2 or len(b.shape) == 2) else ''
    
    proj = np.einsum(f'{ida},{idb}->{id}', Normalize_vect(a), Normalize_vect(b), optimize='optimal')

    if np.max(np.abs(proj)) == 1:
        # a and b are colinear
        angle = 0 if proj == 1 else np.pi

    else:    
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        proj = np.einsum(f'{ida},{idb}->{id}', a, b, optimize='optimal')
        angle = np.arccos(proj/(norm_a*norm_b))
    
    return angle

def Translate_coord(coord: np.ndarray, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> np.ndarray:
    """Translates the coordinates."""

    oldCoord = np.reshape(coord, (-1, 3))

    dec = As_Coordinates([dx, dy, dz])

    newCoord = oldCoord + dec

    return newCoord

def Rotate_coord(coord: np.ndarray, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> np.ndarray:
    """Rotates the coordinates arround a specified center and axis.

    Parameters
    ----------
    coord : np.ndarray
        coordinates to rotate (n,3)
    theta : float
        rotation angle [deg] 
    center : tuple, optional
        rotation center, by default (0,0,0)
    direction : tuple, optional
        rotation direction, by default (0,0,1)

    Returns
    -------
    np.ndarray
        rotated coordinates
    """

    center = As_Coordinates(center)
    direction = As_Coordinates(direction)

    theta *= np.pi/180

    # rotation matrix
    rotMat = Rotation_matrix(direction, theta)

    oldCoord = np.reshape(coord, (-1,3))
    
    newCoord: np.ndarray = np.einsum('ij,nj->ni', rotMat, oldCoord - center, optimize='optimal') + center

    return newCoord

def Symmetry_coord(coord: np.ndarray, point=(0,0,0), n=(1,0,0)) -> np.ndarray:
    """Symmetrizes coordinates with a plane.

    Parameters
    ----------
    coord : np.ndarray
        coordinates that we want to symmetrise
    point : tuple, optional
        a point belonging to the plane, by default (0,0,0)
    n : tuple, optional
        normal to the plane, by default (1,0,0)

    Returns
    -------
    np.ndarray
        the new coordinates
    """

    point = As_Coordinates(point)
    n = Normalize_vect(As_Coordinates(n))

    oldCoord = np.reshape(coord, (-1,3))

    d = (oldCoord - point) @ n    

    newCoord = oldCoord - np.einsum('n,i->ni', 2*d, n, optimize='optimal')

    return newCoord

def Jacobian_Matrix(i: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Computes the Jacobian matrix to transform local coordinates (i,j,k) to global (x,y,z) coordinates.\n
    p(x,y,z) = J • p(i,j,k) and p(i,j,k) = inv(J) • p(x,y,z)\n\n

    ix jx kx\n
    iy jy ky\n
    iz jz kz

    Parameters
    ----------
    i : np.ndarray
        i vector
    k : np.ndarray
        k vector
    """        

    i = Normalize_vect(i)
    k = Normalize_vect(k)
    
    j = np.cross(k, i)
    j = Normalize_vect(j)

    F = np.zeros((3,3))

    F[:,0] = i
    F[:,1] = j
    F[:,2] = k

    return F

def Points_Rayon(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes fillet in a corner P0.\n
    returns A, B, C

    Parameters
    ----------
    P0 : np.ndarray
        coordinates of point with radius
    P1 : np.ndarray
        coordinates before P0 coordinates
    P2 : np.ndarray
        coordinates after P0 coordinates
    r : float
        radius (or fillet) at point P0

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        coordinates calculated to construct the radius
    """
                
    # vectors
    i = P1-P0
    j = P2-P0
    
    n = np.cross(i, j) # normal vector to the plane formed by i, j

    if r > 0:
        # angle from i to k
        betha = AngleBetween_a_b(i, j)/2
        
        d = np.abs(r)/np.tan(betha) # distance between P0 and A on i and distance between P0 and B on j

        d *= np.sign(betha)

        A = Jacobian_Matrix(i, n) @ np.array([d,0,0]) + P0
        B = Jacobian_Matrix(j, n) @ np.array([d,0,0]) + P0
        C = Jacobian_Matrix(i, n) @ np.array([d,r,0]) + P0
    else:
        d = np.abs(r)
        A = Jacobian_Matrix(i, n) @ np.array([d,0,0]) + P0
        B = Jacobian_Matrix(j, n) @ np.array([d,0,0]) + P0
        C = P0

    return A, B, C

def Circle_Triangle(p1, p2, p3) -> np.ndarray:
    """Returns triangle's center for the circumcicular arc formed by 3 points.\n
    returns center
    """

    # https://math.stackexchange.com/questions/1076177/3d-coordinates-of-circle-center-given-three-point-on-the-circle

    p1 = As_Coordinates(p1)
    p2 = As_Coordinates(p2)
    p3 = As_Coordinates(p3)

    v1 = p2-p1
    v2 = p3-p1

    v11 = v1 @ v1
    v22 = v2 @ v2
    v12 = v1 @ v2

    b = 1 / (2*(v11*v22-v12**2))
    k1 = b * v22 * (v11-v12)
    k2 = b * v11 * (v22-v12)

    center = p1 + k1 * v1 + k2 * v2

    return center

def Circle_Coord(coord: np.ndarray, R: float, n: np.ndarray) -> np.ndarray:
    """Returns center from coordinates a radius and and a vector normal to the circle.\n
    return center
    """

    R = np.abs(R)

    coord = np.reshape(coord, (-1, 3))

    assert coord.shape[0] >= 2, 'must give at least 2 points'
    
    n = As_Coordinates(n)

    p0 = np.mean(coord, 0)
    x0, y0, z0 = coord[0]        

    def eval(v):
        x,y,z = v
        f = np.linalg.norm(np.linalg.norm(coord-v, axis=1) - R**2)
        return f

    # point must belong to the plane
    eqPlane = lambda v: v @ n
    cons = ({'type': 'eq', 'fun': eqPlane})
    res = minimize(eval, p0, constraints=cons, tol=1e-12)

    assert res.success, 'the center has not been found'
    center: np.ndarray = res.x

    return center

def Points_Intersect_Circles(circle1: Circle, circle2: Circle) -> np.ndarray:
    """Computes the coordinates at the intersection of the two circles (i,3).\n
    This only works if they're on the same plane.

    Parameters
    ----------
    circle1 : Circle
        circle 1
    circle2 : Circle
        circle 2
    """

    r1 = circle1.diam/2
    r2 = circle2.diam/2

    p1 = circle1.center.coord
    p2 = circle2.center.coord

    d = np.linalg.norm(p2 - p1)

    if d > r1 + r2:
        print("The circles are separated")
        return None
    elif d < np.abs(r1 - r2):
        print("The circles are concentric")
        return None
    elif d == 0 and r1 == r2:
        print("The circles are the same")
        return None
    
    a = (r1**2  - r2**2 + d**2)/(2*d)
    h = np.sqrt(r1**2 - a**2)

    p3 = p1 + a*(p2-p1)/d

    if d == r1 + r2:
        return p3.reshape(1, 3)
    else:
        i = Normalize_vect(p2-p1)
        k = np.array([0,0,1])
        j = np.cross(k, i)

        mat = np.array([i,j,k]).T

        coord = np.zeros((2, 3))
        coord[0,:] = p3 + mat @ np.array([0,-h,0]) 
        coord[1,:] = p3 + mat @ np.array([0,+h,0])
        return coord