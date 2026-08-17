# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Domain class."""

import numpy as np

from ._utils import Point, AsPoint, AsCoords
from ._geom import _Geom
from ..Utilities import _types


class Domain(_Geom):
    """Domain (2d or 3d) class.

    A domain is held as its two opposite corners `pt1` and `pt2`, and every other corner is rebuilt from them component by component, so it is always the axis-aligned box those two corners span. `Translate`, `Rotate` and `Symmetry` move `pt1` and `pt2`, which means a transformation that does not keep the box axis-aligned gives the axis-aligned box spanned by the two transformed corners, not the transformed shape: rotating a 2 x 1 domain by 45 deg leaves a domain of area 1.5. Use `Points` with the four corners when a shape has to keep an arbitrary orientation.
    """

    __NInstance = 0

    def _Init_Ninstance():
        Domain.__NInstance = 0

    def __init__(
        self,
        pt1: Point.PointALike,
        pt2: Point.PointALike,
        meshSize: _types.Number = 0.0,
        isFilled: bool = False,
    ):
        """Creates a 2d or 3d domain.

        Parameters
        ----------
        pt1 : Point | Coords
            first point
        pt2 : Point | Coords
            second point
        meshSize : float, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isFilled : bool, optional
            the enclosed region is filled (solid inclusion), by default False
        """

        Domain.__NInstance += 1
        name = f"Domain{Domain.__NInstance}"
        # a domain can't be open
        _Geom.__init__(
            self, [AsPoint(pt1), AsPoint(pt2)], meshSize, name, isFilled, False
        )

    # the points are exposed as views on `points`, so they cannot desync from it
    @property
    def pt1(self) -> Point:
        """first point"""
        return self.points[0]

    @property
    def pt2(self) -> Point:
        """second point"""
        return self.points[1]

    def Get_Contour(self):
        """Creates the contour object associated with the domain: the four edges in the `pt1.z` plane, as the domain is meshed."""
        pt1, pt2 = self.pt1, self.pt2
        corners = [
            Point(pt1.x, pt1.y, pt1.z),
            Point(pt2.x, pt1.y, pt1.z),
            Point(pt2.x, pt2.y, pt1.z),
            Point(pt1.x, pt2.y, pt1.z),
        ]

        from ._contour import Contour
        from ._line import Line

        edges = [
            Line(start, end, self.meshSize)
            for start, end in zip(corners, corners[1:] + corners[:1])
        ]

        return Contour(edges, self.isFilled, self.isOpen)

    def Contains(self, coord: _types.Coords, tol: float = 1e-12) -> _types.BoolArray:
        """Returns, for each of the given points, whether it lies on one of the four edges."""
        return self.Get_Contour().Contains(coord, tol)

    def Encloses(self, coord: _types.Coords, tol: float = 1e-12) -> _types.BoolArray:
        """Returns, for each of the given points, whether it lies in the box, faces included.

        Unlike `Contains`, this is the 3d box spanned by `pt1` and `pt2`, not the four edges in the `pt1.z` plane.
        """
        coord = np.reshape(AsCoords(coord), (-1, 3))
        pt1, pt2 = self.pt1.coord, self.pt2.coord

        return np.all(
            (coord >= np.minimum(pt1, pt2) - tol)
            & (coord <= np.maximum(pt1, pt2) + tol),
            axis=1,
        )

    def Get_coord_for_plot(
        self, N: int = None
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        p1 = self.pt1.coord
        p7 = self.pt2.coord

        dx, dy, dz = p7 - p1

        p2 = p1 + [dx, 0, 0]
        p3 = p1 + [dx, dy, 0]
        p4 = p1 + [0, dy, 0]
        p5 = p1 + [0, 0, dz]
        p6 = p1 + [dx, 0, dz]
        p8 = p1 + [0, dy, dz]

        lines = np.concatenate(
            (p1, p2, p3, p4, p1, p5, p6, p2, p6, p7, p3, p7, p8, p4, p8, p5)
        ).reshape((-1, 3))

        points = np.concatenate((p1, p7)).reshape((-1, 3))

        return lines, points
