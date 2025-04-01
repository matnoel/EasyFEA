# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import copy
from scipy.optimize import minimize
from collections.abc import Iterable

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
        self.coord = (x, y, z)
        self.isOpen = isOpen

    @property
    def x(self) -> float:
        """x coordinate"""
        return self.__coord[0]
    
    @x.setter
    def x(self, value) -> None:
        assert isinstance(value, (float, int))
        self.__coord[0] = value

    @property
    def y(self) -> float:
        """y coordinate"""
        return self.__coord[1]
    
    @y.setter
    def y(self, value) -> None:
        assert isinstance(value, (float, int))
        self.__coord[1] = value

    @property
    def z(self) -> float:
        """z coordinate"""
        return self.__coord[2]
    
    @z.setter
    def z(self, value) -> None:
        assert isinstance(value, (float, int))
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
        coord = AsCoords(value)
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
        coord = AsCoords(coord)
        n = np.linalg.norm(self.coord)
        n = 1 if n == 0 else n
        diff = np.linalg.norm(self.coord - coord)/n
        return diff <= 1e-12
    
    def Translate(self, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> None:
        """Translates the point."""
        self.__coord = Translate(self.__coord, dx, dy, dz).ravel()

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
        self.__coord = Rotate(self.__coord, theta, center, direction).ravel()

    def Symmetry(self, point=(0,0,0), n=(1,0,0)) -> None:
        """Symmetrizes the point coordinates with a plane.

        Parameters
        ----------
        point : tuple, optional
            a point belonging to the plane, by default (0,0,0)
        n : tuple, optional
            normal to the plane, by default (1,0,0)
        """
        self.__coord = Symmetry(self.__coord, point, n).ravel()
    
    def __radd__(self, value):
        return self.__add__(value)

    def __add__(self, value):
        coord = AsCoords(value)        
        newCoord: np.ndarray = self.coord + coord
        return Point(*newCoord)

    def __rsub__(self, value):
        return self.__add__(value)
    
    def __sub__(self, value):
        coord = AsCoords(value)        
        newCoord: np.ndarray = self.coord - coord
        return Point(*newCoord)
    
    def __rmul__(self, value):
        return self.__mul__(value)

    def __mul__(self, value):
        coord = AsCoords(value)
        newCoord: np.ndarray = self.coord * coord
        return Point(*newCoord)
    
    def __rtruediv__(self, value):
        return self.__truediv__(value)

    def __truediv__(self, value):
        coord = AsCoords(value)
        newCoord: np.ndarray = self.coord / coord
        return Point(*newCoord)
    
    def __rfloordiv__(self, value):
        return self.__floordiv__(value)

    def __floordiv__(self, value):
        coord = AsCoords(value)
        newCoord: np.ndarray = self.coord // coord
        return Point(*newCoord)
    
    def copy(self):
        return copy.deepcopy(self)

def AsPoint(coords) -> Point:
    """Returns coords as a point."""
    if isinstance(coords, Point):
        return coords
    elif isinstance(coords, Iterable):
        coords = AsCoords(coords)
        return Point(*coords)
    else:
        raise TypeError("coords must be a Point or an Iterable")

def AsCoords(value) -> np.ndarray:
    """Returns value as a 3D vector"""
    if isinstance(value, Point):
        coords = value.coord        
    elif isinstance(value, Iterable):
        val = np.asarray(value, dtype=float)
        if len(val.shape) == 2:
            assert val.shape[-1] <= 3, 'must be 3d vector or 3d vectors'
            coords = val
        else:
            coords = np.zeros(3)
            assert val.size <= 3, 'must not exceed size 3'
            coords[:val.size] = val
    elif isinstance(value, (float, int)):            
        coords = np.asarray([value]*3)
    else:
        raise TypeError(f'{type(value)} is not supported. Must be (Point | float | int | Iterable)')
    
    return coords

def Normalize(array: np.ndarray) -> np.ndarray:
    """Must be a vector or matrix."""
    array = np.asarray(array)
    if array.ndim == 1:
        return array / np.linalg.norm(array)
    elif array.ndim == 2:
        return np.einsum('ij,i->ij',array, 1/np.linalg.norm(array, axis=1), optimize="optimal")
    else:
        raise Exception("The array is the wrong size")

def Translate(coord: np.ndarray, dx: float=0.0, dy: float=0.0, dz: float=0.0) -> np.ndarray:
    """Translates the coordinates."""

    oldCoord = np.reshape(coord, (-1, 3))

    dec = AsCoords([dx, dy, dz])

    newCoord = oldCoord + dec

    return newCoord

def __Rotation_matrix(vect: np.ndarray, theta: float) -> np.ndarray:
    """Gets the rotation matrix for turning along an axis with theta angle (rad).\n
    p(x,y) = mat • p(i,j)\n
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle"""

    x, y, z = Normalize(vect)
    
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    mat = np.array([[x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
                    [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
                    [z*x*C - y*s, z*y*C + x*s, z*z*C + c]])
    
    return mat

def Rotate(coord: np.ndarray, theta: float, center: tuple=(0,0,0), direction: tuple=(0,0,1)) -> np.ndarray:
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

    center = AsCoords(center)
    direction = AsCoords(direction)

    theta *= np.pi/180

    # rotation matrix
    rotMat = __Rotation_matrix(direction, theta)

    oldCoord = np.reshape(coord, (-1,3))
    
    newCoord: np.ndarray = np.einsum('ij,nj->ni', rotMat, oldCoord - center, optimize='optimal') + center

    return newCoord

def Symmetry(coord: np.ndarray, point=(0,0,0), n=(1,0,0)) -> np.ndarray:
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

    point = AsCoords(point)
    n = Normalize(AsCoords(n))

    oldCoord = np.reshape(coord, (-1,3))

    d = (oldCoord - point) @ n    

    newCoord = oldCoord - np.einsum('n,i->ni', 2*d, n, optimize='optimal')

    return newCoord

# circles

def Circle_Triangle(p1, p2, p3) -> np.ndarray:
    """Returns triangle's center for the circumcicular arc formed by 3 points.\n
    returns center
    """

    # https://math.stackexchange.com/questions/1076177/3d-coordinates-of-circle-center-given-three-point-on-the-circle

    p1 = AsCoords(p1)
    p2 = AsCoords(p2)
    p3 = AsCoords(p3)

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

def Circle_Coords(coord: np.ndarray, R: float, n: np.ndarray) -> np.ndarray:
    """Returns center from coordinates a radius and and a vector normal to the circle.\n
    return center
    """

    R = np.abs(R)

    coord = np.reshape(coord, (-1, 3))

    assert coord.shape[0] >= 2, 'must give at least 2 points'
    
    n = AsCoords(n)

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

def Points_Intersect_Circles(circle1, circle2) -> np.ndarray:
    """Computes the coordinates at the intersection of the two circles (i,3).\n
    This only works if they're on the same plane.

    Parameters
    ----------
    circle1 : Circle
        circle 1
    circle2 : Circle
        circle 2
    """
    from ._circle import Circle

    assert isinstance(circle1, Circle)
    assert isinstance(circle2, Circle)

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
        i = Normalize(p2-p1)
        k = np.array([0,0,1])
        j = np.cross(k, i)

        mat = np.array([i,j,k]).T

        coord = np.zeros((2, 3))
        coord[0,:] = p3 + mat @ np.array([0,-h,0]) 
        coord[1,:] = p3 + mat @ np.array([0,+h,0])
        return coord
    
# others 

def Angle_Between(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the angle between vectors a and b (rad).
    https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors"""

    a = AsCoords(a)
    b = AsCoords(b)    
    
    ida = 'ni' if len(a.shape) == 2 else 'i'
    idb = 'ni' if len(b.shape) == 2 else 'i'
    id = 'n' if (len(a.shape) == 2 or len(b.shape) == 2) else ''
    
    proj = np.einsum(f'{ida},{idb}->{id}', Normalize(a), Normalize(b), optimize='optimal')

    if np.max(np.abs(proj)) == 1:
        # a and b are colinear
        angle = 0 if proj == 1 else np.pi

    else:    
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        proj = np.einsum(f'{ida},{idb}->{id}', a, b, optimize='optimal')
        angle = np.arccos(proj/(norm_a*norm_b))
    
    return angle

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

    i = Normalize(i)
    k = Normalize(k)
    
    j = np.cross(k, i)
    j = Normalize(j)

    F = np.zeros((3,3))

    F[:,0] = i
    F[:,1] = j
    F[:,2] = k

    return F

def Fillet(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        betha = Angle_Between(i, j)/2
        
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