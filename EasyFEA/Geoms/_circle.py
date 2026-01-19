# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Circle and CircleArc classes."""

import numpy as np
from typing import Union, Optional

from ._utils import (
    Point,
    AsCoords,
    AsPoint,
    Normalize,
    Jacobian_Matrix,
    Angle_Between,
    Circle_Triangle,
    Circle_Coords,
)
from ._geom import _Geom
from ..Utilities import _params, _types


class Circle(_Geom):
    """Circle class."""

    __NInstance = 0

    def _Init_Ninstance():
        Circle.__NInstance = 0

    def __init__(
        self,
        center: Point.PointALike,
        diam: float,
        meshSize: float = 0.0,
        isHollow: bool = True,
        isOpen: bool = False,
        n: _types.Coords = (0, 0, 1),
    ):
        """Creates a circle according to its center, diameter and the normal vector.

        Parameters
        ----------
        center : Point | Coords
            center of circle
        diam : float
            diameter
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            circle is hollow/empty, by default True
        isOpen : bool, optional
            circle can be opened (openCrack), by default False
        n : Coords, optional
            normal direction to the circle, by default (0,0,1)
        """

        _params._CheckIsPositive(diam)

        center = AsPoint(center)

        r = diam / 2

        # creates points associated with the circle
        self.center = center
        self.pt1 = center + [r, 0, 0]
        self.pt2 = center + [0, r, 0]
        self.pt3 = center + [-r, 0, 0]
        self.pt4 = center + [0, -r, 0]

        Circle.__NInstance += 1
        name = f"Circle{Circle.__NInstance}"
        _Geom.__init__(
            self,
            [self.center, self.pt1, self.pt2, self.pt3, self.pt4],
            meshSize,
            name,
            isHollow,
            isOpen,
        )

        # rotate if necessary
        zAxis = np.array([0, 0, 1])
        n = Normalize(AsCoords(n))
        rotAxis = np.cross(n, zAxis)
        # theta = AngleBetween_a_b(zAxis, n)

        # then we rotate along i
        if np.linalg.norm(rotAxis) == 0:
            # n and zAxis are collinear
            i = Normalize((self.pt1 - center).coord)  # i = p1 - center
        else:
            i = rotAxis

        mat = Jacobian_Matrix(i, n)

        coord = np.einsum("ij,nj->ni", mat, self.coord - center.coord) + center.coord

        for p, point in enumerate(self.points):
            point.coord = coord[p]

    @property
    def diam(self) -> float:
        """circle's diameter"""
        p1 = self.pt1.coord
        pC = self.center.coord
        return np.linalg.norm(p1 - pC).astype(float) * 2

    @property
    def n(self) -> _types.Coords:
        """axis normal to the circle"""
        i = Normalize((self.pt1 - self.center).coord)
        j = Normalize((self.pt2 - self.center).coord)
        n: _types.Coords = Normalize(np.cross(i, j))
        return n

    def Get_coord_for_plot(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        angle = np.linspace(0, np.pi * 2, 40)

        pC = self.center
        R = self.diam / 2

        points = self.coord

        lines = np.zeros((angle.size, 3))
        lines[:, 0] = np.cos(angle) * R
        lines[:, 1] = np.sin(angle) * R

        # construct jacobian matrix
        i = (self.pt1 - self.center).coord
        n = self.n
        mat = Jacobian_Matrix(i, n)

        # change base
        lines = np.einsum("ij,nj->ni", mat, lines) + pC.coord

        return lines, points[1:]

    @property
    def length(self) -> float:
        """circle perimeter"""
        return np.pi * self.diam

    def Get_Contour(self):
        """Creates the contour object associated with the circle"""

        center = self.center
        meshSize = self.meshSize
        isHollow = self.isHollow
        isOpen = self.isOpen

        # creates circle arcs associated with the circle
        circleArc1 = CircleArc(
            self.pt1, self.pt2, center=center, meshSize=meshSize, isOpen=isOpen
        )
        circleArc2 = CircleArc(
            self.pt2, self.pt3, center=center, meshSize=meshSize, isOpen=isOpen
        )
        circleArc3 = CircleArc(
            self.pt3, self.pt4, center=center, meshSize=meshSize, isOpen=isOpen
        )
        circleArc4 = CircleArc(
            self.pt4, self.pt1, center=center, meshSize=meshSize, isOpen=isOpen
        )

        from ._contour import Contour

        return Contour(
            [circleArc1, circleArc2, circleArc3, circleArc4], isHollow, isOpen
        )


class CircleArc(_Geom):
    """CircleArc class."""

    __NInstance = 0

    def _Init_Ninstance():
        CircleArc.__NInstance = 0

    def __init__(
        self,
        pt1: Point.PointALike,
        pt2: Point.PointALike,
        center: Union[Point, None] = None,
        R: Optional[_types.Number] = None,
        P: Optional[_types.Coords] = None,
        meshSize: _types.Number = 0.0,
        n: _types.Coords = (0, 0, 1),
        isOpen: bool = False,
        coef: int = 1,
    ):
        """Creates a circular arc using several methods:\n
        - 1: with 2 points, a radius R and a normal vector.\n
        - 2: with 2 points and a center.\n
        - 3: with 2 points and a point P belonging to the circle.\n
        The methods are chosen in the following order 3 2 1.\n
        This means that if you enter P, the other methods will not be used.

        Parameters
        ----------
        pt1 : Point | Coords
            starting point
        pt2: Point | Coords
            ending point
        R: _types.Number, optional
            radius of the arc circle, by default None
        center: Point, optional
            center of circular arc, by default None
        P: _types.Coords, optional
            a point belonging to the circle, by default None
        meshSize : _types.Number, optional
            size to be used for mesh construction, by default 0.0
        n: Coords, optional
            normal vector to the arc circle, by default (0,0,1)
        isOpen : bool, optional
            arc can be opened, by default False
        coef: int, optional
            Change direction, by default 1 or -1
        """

        pt1 = AsPoint(pt1)
        pt2 = AsPoint(pt2)

        # check that pt1 and pt2 dont share the same coordinates
        assert not pt1.Check(pt2), "pt1 and pt2 are on the same coordinates"

        if center is not None:
            center = AsPoint(center)
            assert not pt1.Check(center), "pt1 and center are on the same coordinates"

        elif P is not None:
            center = Point(*Circle_Triangle(pt1, pt2, P))

        elif R is not None:
            coord = np.array([pt1.coord, pt2.coord])
            center = Point(*Circle_Coords(coord, R, n))
        else:
            raise Exception("must give P, center or R")

        r1 = np.linalg.norm((pt1 - center).coord)
        r2 = np.linalg.norm((pt2 - center).coord)
        assert (
            r1 - r2
        ) ** 2 / r2**2 <= 1e-12, "The given center doesn't have the right coordinates. If the center coordinate is difficult to identify, you can give:\n - the radius R with the vector normal to the circle n\n - another point belonging to the circle."

        self.center = center
        """Point at the center of the arc."""
        self.pt1 = pt1
        """Starting point of the arc."""
        self.pt2 = pt2
        """Ending point of the arc."""

        # Here we'll create an intermediate point, because in gmsh, circular arcs are limited to an pi angle.

        i1 = (pt1 - center).coord
        i2 = (pt2 - center).coord

        # construction of the passage matrix
        k = np.array([0, 0, 1])
        if np.linalg.norm(np.cross(i1, i2)) <= 1e-12:
            vect = Normalize(i2 - i1)
            i = np.cross(k, vect)
        else:
            i = Normalize((i1 + i2) / 2)
            k = Normalize(np.cross(i1, i2))
        j = np.cross(k, i)

        mat = np.array([i, j, k]).T

        # midpoint coordinates
        _params._CheckIsInIntervaloo(coef, -1, 1)
        pt3 = center.coord + mat @ [coef * r1, 0, 0]

        self.pt3 = Point(*pt3)
        """Midpoint of the circular arc."""

        self.coef = coef

        CircleArc.__NInstance += 1
        name = f"CircleArc{CircleArc.__NInstance}"
        _Geom.__init__(
            self, [pt1, center, self.pt3, pt2], meshSize, name, False, isOpen
        )

    @property
    def n(self) -> _types.Coords:
        """axis normal to the circle arc"""
        i = Normalize((self.pt1 - self.center).coord)
        if np.any(np.isclose(self.angle, [0, np.pi])):
            j = Normalize((self.pt3 - self.center).coord)
        else:
            j = Normalize((self.pt2 - self.center).coord)
        n = Normalize(np.cross(i, j))
        return n

    @property
    def angle(self):
        """circular arc angle [rad]"""
        i = (self.pt1 - self.center).coord
        j = (self.pt2 - self.center).coord
        return Angle_Between(i, j)

    @property
    def r(self):
        """circular arc radius"""
        return np.linalg.norm((self.pt1 - self.center).coord)

    @property
    def length(self) -> float:
        """circular arc length"""
        return np.abs(self.angle * self.r)

    def Get_coord_for_plot(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        points = self.coord

        pC = self.center
        r = self.r

        # plot arc circle in 2D space
        angles = np.linspace(0, np.abs(self.angle), 11)
        lines = np.zeros((angles.size, 3))
        lines[:, 0] = np.cos(angles) * r
        lines[:, 1] = np.sin(angles) * r

        # get the jabobian matrix
        i = (self.pt1 - self.center).coord
        n = self.n

        mat = Jacobian_Matrix(i, n)

        # transform coordinates
        lines = np.einsum("ij,nj->ni", mat, lines) + pC.coord

        return lines, points[[0, -1]]
