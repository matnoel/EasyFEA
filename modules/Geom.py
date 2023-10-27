"""Module for creating geometric objects."""

from typing import Union
import numpy as np

class Point:    

    def __init__(self, x=0.0, y=0.0, z=0.0, isOpen=False, r=0.0):
        """Build a point.

        Parameters
        ----------
        x : float, optional
            x coordinate, default 0.0
        y : float, optional
            y coordinate, default 0.0
        z : float, optional
            z coordinate, default 0.0
        isOpen : bool, optional
            point can open, default False
        r : float, optional
            radius used for fillet
        """
        self.__x = x
        self.__y = y
        self.__z = z        
        self.__r = r
        self.__coordo = np.array([x, y, z])
        self.__isOpen = isOpen

    @property
    def x(self) -> float:
        """x coordinate"""
        return self.__x

    @property
    def y(self) -> float:
        """y coordinate"""
        return self.__y

    @property
    def z(self) -> float:
        """z coordinate"""
        return self.__z

    @property
    def r(self) -> float:
        """radius used for fillet"""
        return self.__r

    @property
    def coordo(self) -> np.ndarray:
        """[x,y,z] coordinates"""
        return self.__coordo

    @property
    def isOpen(self):
        """point is open"""
        return self.__isOpen
    
    def __radd__(self, value):
        return self.__add__(value)

    def __add__(self, value):
        if isinstance(value, float) and isinstance(value, int):
            x = self.x + value
            y = self.y + value
            z = self.z + value
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list):
            x = self.x + value[0] if len(value) > 0 else self.x
            y = self.y + value[1] if len(value) > 1 else self.y
            z = self.z + value[2] if len(value) > 2 else self.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, Point):
            x = self.x + value.x
            y = self.y + value.y
            z = self.z + value.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list[Point]):
            points = [] 
            for point in value:
                x = self.x + point.x
                y = self.y + point.y
                z = self.z + point.z
            points.append(Point(x, y, z, self.isOpen, self.r))
        elif isinstance(value, np.ndarray):
            points = []
            for x, y, z in zip(value[:,0], value[:,1], value[:,2]):
                x = self.x + x
                y = self.y + y
                z = self.z + z
                points.append(Point(x, y, z, self.isOpen, self.r))

    def __rsub__(self, value):
        return self.__add__(value)
    
    def __sub__(self, value):
        if isinstance(value, float) and isinstance(value, int):
            x = self.x - value
            y = self.y - value
            z = self.z - value
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list):
            x = self.x - value[0] if len(value) > 0 else self.x
            y = self.y - value[1] if len(value) > 1 else self.y
            z = self.z - value[2] if len(value) > 2 else self.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, Point):
            x = self.x - value.x
            y = self.y - value.y
            z = self.z - value.z
            return Point(x, y, z, self.isOpen, self.r)
        elif isinstance(value, list[Point]):
            points = [] 
            for point in value:
                x = self.x - point.x
                y = self.y - point.y
                z = self.z - point.z
            points.append(Point(x, y, z, self.isOpen, self.r))
        elif isinstance(value, np.ndarray):
            points = []
            for x, y, z in zip(value[:,0], value[:,1], value[:,2]):
                x = self.x - x
                y = self.y - y
                z = self.z - z
                points.append(Point(x, y, z, self.isOpen, self.r))    

class Geom:

    def __init__(self, points: list[Point], meshSize: float, name: str):
        """Builds a geometric object.

        Parameters
        ----------
        points : list[Point]
            list of points to build the geometric object
        meshSize : float
            mesh size that will be used to create the mesh >= 0
        name : str
            object name
        """
        assert meshSize >= 0
        self.__meshSize = meshSize

        self.__points = points

        self.__name = name

    @property
    def meshSize(self) -> float:
        """Element size used for meshing"""
        return self.__meshSize

    @property
    def points(self) -> list[Point]:
        """Points used to build the object"""
        return self.__points

    @property
    def name(self) -> str:
        """object name"""
        return self.__name

class PointsList(Geom):

    __nbPointsList = 0

    def __init__(self, contour: list[Point], meshSize=0.0, isHollow=False):
        """Builds a point list

        Parameters
        ----------
        points : list[Point]
            list of geom objects to build a contour
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            formed domain is hollow/empty, by default False
        """

        self.isHollow=isHollow

        PointsList.__nbPointsList += 1
        name = f"PointsList{PointsList.__nbPointsList}"
        super().__init__(contour, meshSize, name)

class Line(Geom):

    __nbLine = 0

    @staticmethod
    def distance(pt1: Point, pt2: Point) -> float:
        """Calculate the distance between two points."""
        length = np.sqrt((pt1.x-pt2.x)**2 + (pt1.y-pt2.y)**2 + (pt1.z-pt2.z)**2)
        return np.abs(length)
    
    @staticmethod
    def get_unitVector(pt1: Point, pt2: Point) -> np.ndarray:
        """Construct the unit vector between two points."""
        length = Line.distance(pt1, pt2)        
        v = np.array([pt2.x-pt1.x, pt2.y-pt1.y, pt2.z-pt1.z])/length
        return v   

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isOpen=False):
        """Builds a line.

        Parameters
        ----------
        pt1 : Point
            first point
        pt2 : Point
            second point
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isOpen : bool, optional
            line can open, by default False
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.coordo = np.array([[pt1.x, pt1.y, pt1.z], [pt2.x, pt2.y, pt2.z]]).reshape(2,3)

        self.__isOpen = isOpen

        Line.__nbLine += 1
        name = f"Line{Line.__nbLine}"
        Geom.__init__(self, points=[pt1, pt2], meshSize=meshSize, name=name)
    

    @property
    def isOpen(self) -> bool:
        """Returns whether the line can open to represent an open crack"""
        return self.__isOpen
    
    @property
    def unitVector(self) -> np.ndarray:
        """The unit vector for the two points on the line (p2-p1)"""
        return Line.get_unitVector(self.pt1, self.pt2)

    @property
    def length(self) -> float:
        """Calculate the distance between the two points on the line"""
        return Line.distance(self.pt1, self.pt2)

class Domain(Geom):

    __nbDomain = 0

    def __init__(self, pt1: Point, pt2: Point, meshSize=0.0, isHollow=False):
        """Builds a domain

        Parameters
        ----------
        pt1 : Point
            first point
        pt2 : Point
            second point
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            formed domain is hollow/empty, by default False
        """
        self.pt1 = pt1
        self.pt2 = pt2

        self.isHollow = isHollow

        Domain.__nbDomain += 1
        name = f"Domain{Domain.__nbDomain}"
        Geom.__init__(self, points=[pt1, pt2], meshSize=meshSize, name=name)

class Circle(Geom):

    __nbCircle = 0

    def __init__(self, center: Point, diam: float, meshSize=0.0, isHollow=True):
        """Constructing a circle according to its center and diameter
        This circle will be projected onto the (x,y) plane.

        Parameters
        ----------
        center : Point
            center of circle
        diam : float
            diameter
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            circle is hollow, by default True
        """
        
        assert diam > 0.0

        self.center = center
        self.diam = diam
        
        self.isHollow = isHollow

        Circle.__nbCircle += 1
        name = f"Circle{Circle.__nbCircle}"
        Geom.__init__(self, points=[center], meshSize=meshSize, name=name)

class CircleArc(Geom):

    __nbCircleArc = 0

    def __init__(self, pt1: Point, center: Point, pt2: Point, meshSize=0.0, coef=1.0, isOpen=False):
        """Construct a circular arc based on its center, start and end points. \n
        This circular arc will be projected onto the (x,y) plane.

        Parameters
        ----------        
        pt1 : Point
            starting point
        center: Point
            center of arc
        pt2: Point
            end point
        meshSize : float, optional
            size to be used for mesh construction, by default 0.0
        coef : float, optional
            coef for multiplication with radius -1 or 1, by default 1.0
        isOpen : bool, optional
            arc can open, by default False
        """

        assert coef in [-1, 1], "coef must be in [-1, 1]."

        r1 = np.linalg.norm((pt1-center).coordo)
        r2 = np.linalg.norm((pt2-center).coordo)

        assert r1 == r2, "The points are not on the same arc."

        self.center = center
        """Point at the center of the arc."""
        self.pt1 = pt1
        """Starting point of the arc."""
        self.pt2 = pt2
        """Ending point of the arc."""

        self.__isOpen = isOpen

        # Here we'll create an intermediate point, because in gmsh, circular arcs are limited to an angle pi.

        i1 = (pt1-center).coordo
        i2 = (pt2-center).coordo

        collinear = np.linalg.norm(np.cross(i1, i2)) <= 1e-12

        # construction of the passage matrix
        k = np.array([0,0,1])
        if collinear:
            vect = normalize_vect(i2-i1)
            i = np.cross(k,vect)
        else:
            i = normalize_vect((i1+i2)/2)
        j = np.cross(k, i)

        mat = np.array([i,j,k]).T

        # midpoint coordinates
        pt3 = center.coordo + mat @ [coef*r1,0,0]

        self.pt3 = Point(pt3[0], pt3[1], pt3[2])
        """Midpoint of the circular arc."""

        CircleArc.__nbCircleArc += 1
        name = f"Circle{CircleArc.__nbCircleArc}"
        Geom.__init__(self, points=[pt1, center, pt2], meshSize=meshSize, name=name)

    @property
    def isOpen(self) -> bool:
        """Returns whether the arc can open to represent a crack."""
        return self.__isOpen

class Contour(Geom):

    __nbContour = 0

    def __init__(self, geoms: list[Union[Line,CircleArc]], isHollow=True):
        """Create a contour from a list of lines or arcs.

        Parameters
        ----------
        geoms : list[Line, CircleArc]
            list of objects used to build the contour
        isHollow : bool, optional
            contour is hollow/empty, by default True
        """

        # Check that the points form a closed loop
        points = []

        tol = 1e-12        

        for i, geom in enumerate(geoms):

            assert isinstance(geom, (Line, CircleArc)), "Must give a list of lines and arcs."

            if i == 0:
                ecart = tol
            elif i > 0 and i < len(geoms)-1:
                # check that the starting point has the same coordinate as the last point of the previous object
                ecart = np.linalg.norm(geom.points[0].coordo - points[-1].coordo)

                assert ecart <= tol, "The contour must form a closed loop."
            else:
                # checks that the end point of the last geometric object is the first point created.
                ecart1 = np.linalg.norm(geom.points[0].coordo - points[-1].coordo)
                ecart2 = np.linalg.norm(geom.points[-1].coordo - points[0].coordo)

                assert ecart1 <= tol and ecart2 <= tol, "The contour must form a closed loop."

            # Adds the first and last points
            points.extend(geom.points)

        self.geoms = geoms

        self.isHollow = isHollow

        Contour.__nbContour += 1
        name = f"Contour{Contour.__nbContour}"
        meshSize = np.mean([geom.meshSize for geom in geoms])
        Geom.__init__(self, points=points, meshSize=meshSize, name=name)


class Section:

    def __init__(self, mesh):
        """Section."""

        from Mesh import Mesh
        assert isinstance(mesh, Mesh), "Must be a 2D mesh"
        assert mesh.dim == 2, "Must be a 2D mesh"
        
        self.__mesh = mesh

    @property
    def mesh(self):
        """Section Mesh"""
        return self.__mesh

    @property
    def epaisseur(self) -> float:
        """Section thickness (x)"""
        coordo = self.__mesh.coordo
        epaisseur = np.abs(coordo[:,0].max() - coordo[:,0].min())
        return epaisseur
    
    @property
    def hauteur(self) -> float:
        """Section height (y)"""
        coordo = self.__mesh.coordo
        hauteur = np.abs(coordo[:,1].max() - coordo[:,1].min())
        return hauteur
    
    @property
    def area(self) -> float:
        """Section area"""
        return self.__mesh.area

    @property
    def Iy(self) -> float:        
        """Squared moment of the section following y\n
        int_S z^2 dS """
        return self.__mesh.Ix

    @property
    def Iz(self) -> float:
        """Squared moment of the section following z\n
        int_S y^2 dS """
        return self.__mesh.Iy

    @property
    def Iyz(self) -> float:
        """Squared moment of the section following yz\n
        int_S y z dS """
        return self.__mesh.Ixy

    @property
    def J(self) -> float:
        """Polar quadratic moment\n
        J = Iz + Iy
        """
        return self.__mesh.J

# Functions for calculating distances, angles, etc.

def normalize_vect(vect: np.ndarray) -> np.ndarray:
    """Returns the normalized vector."""
    if len(vect.shape) == 1:
        return vect / np.linalg.norm(vect)
    elif len(vect.shape) == 2:
        return np.einsum('ij,i->ij',vect, 1/np.linalg.norm(vect, axis=1), optimize="optimal")
    else:
        raise Exception("The vector is the wrong size")

def AngleBetween_a_b(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates the angle between vector a and vector b.
    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors"""

    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), "a et b doivent être des np.array"
    assert a.shape == (3,) and b.shape == (3,), "a et b doivent être des vecteur 3D"
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    cosAngle = a.dot(b)/(norm_a * norm_b)
    sinAngle = np.cross(a,b)/(norm_a * norm_b)
    angles = np.arctan2(sinAngle, cosAngle)

    vectNorm = normalize_vect(np.cross(b,a))

    # angle = angles[-1]
    angle = np.dot(angles, vectNorm)
    
    return angle

def JacobianMatrix(i: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute the Jacobian matrix for the transition from the (i,j,k) basis to the (x,y,z) basis.\n
    p(x,y) = matrice • p(i,j)

    Parameters
    ----------
    i : np.ndarray
        i vector
    k : np.ndarray
        j vector
    """        

    i = normalize_vect(i)
    k = normalize_vect(k)
    
    j = np.cross(k, i)
    j = normalize_vect(j)

    F = np.zeros((3,3))

    F[:,0] = i
    F[:,1] = j
    F[:,2] = k

    return F

def Points_Rayon(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculation of point coordinates to create a radius in a corner.

    Parameters
    ----------
    P0 : np.ndarray
        coordinates of point with radius
    P1 : np.ndarray
        coordinates before P0 coordinates
    P2 : np.ndarray
        coordinates after P0 coordinates
    r : float
        radius at point P0

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

        A = JacobianMatrix(i, n) @ np.array([d,0,0]) + P0
        B = JacobianMatrix(j, n) @ np.array([d,0,0]) + P0
        C = JacobianMatrix(i, n) @ np.array([d, r,0]) + P0
    else:
        d = np.abs(r)
        A = JacobianMatrix(i, n) @ np.array([d,0,0]) + P0
        B = JacobianMatrix(j, n) @ np.array([d,0,0]) + P0
        C = P0

    return A, B, C

def Points_IntersectCircles(circle1: Circle, circle2: Circle) -> np.ndarray:
    """Calculates the coordinates at the intersection of the two circles (i,3). This only works if they're on the same plane.

    Parameters
    ----------
    circle1 : Circle
        circle 1
    circle2 : Circle
        circle 2
    """

    r1 = circle1.diam/2
    r2 = circle2.diam/2

    p1 = circle1.center.coordo
    p2 = circle2.center.coordo

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

        i = normalize_vect(p2-p1)
        k = np.array([0,0,1])
        j = np.cross(k, i)

        mat = np.array([i,j,k]).T

        coord = np.zeros((2, 3))
        coord[0,:] = p3 + mat @ np.array([0,-h,0]) 
        coord[1,:] = p3 + mat @ np.array([0,+h,0])
        return coord