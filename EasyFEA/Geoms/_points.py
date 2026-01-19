# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Points class."""

from collections.abc import Iterable
import numpy as np

from ._utils import Point, AsPoint, Fillet
from ._geom import _Geom
from ._line import Line
from ._circle import CircleArc

from typing import Union, Collection
from ..Utilities import _types


class Points(_Geom):
    """Points class."""

    __NInstance = 0

    def _Init_Ninstance():
        Points.__NInstance = 0

    def __init__(
        self,
        points: Collection[Point.PointALike],
        meshSize: _types.Number = 0.0,
        isHollow: bool = True,
        isOpen: bool = False,
    ):
        """Creates points (list of point).\n
        Can be used to construct a closed surface or a spline.

        Parameters
        ----------
        points : Collection[Point | _types.Coords]
            list of points
        meshSize : _types.Number, optional
            mesh size that will be used to create the mesh >= 0, by default 0.0
        isHollow : bool, optional
            the formed domain is hollow/empty, by default True
        isOpen : bool, optional
            the spline formed by the points list can be opened (openCrack), by default False
        """

        assert isinstance(points, Iterable), "points must be Iterable."

        list_point: list[Point] = [AsPoint(point) for point in points]

        self.pt1 = list_point[0]
        """First point"""
        self.pt2 = list_point[-1]
        """Last point"""

        Points.__NInstance += 1
        name = f"Points{Points.__NInstance}"
        super().__init__(list_point, meshSize, name, isHollow, isOpen)

    def Get_Contour(self):
        """Creates a contour from the points.\n
        Creates a fillet if a point has a radius which is not 0."""

        coordinates = self.coord
        N = coordinates.shape[0]
        mS = self.meshSize

        # TODO Allows the addition of chamfers?

        from ._contour import Contour, ContourCompatible

        # Get corners
        corners: list[Union[Point, _Geom]] = []
        geoms: list[ContourCompatible] = []

        def Link(idx1: int, idx2: int):
            # this function makes the link between corners[idx1] and corners[idx2]

            # get the last point associated with idx1
            if isinstance(corners[idx1], Point):
                p1 = AsPoint(corners[idx1])  # type: ignore [arg-type]
            else:
                p1 = AsPoint(corners[idx1].points[-1])  # type: ignore [union-attr]

            # get the first point associated with idx2
            if isinstance(corners[idx2], Point):
                p2 = AsPoint(corners[idx2])  # type: ignore [arg-type]
            else:
                p2 = AsPoint(corners[idx2].points[0])  # type: ignore [union-attr]

            if not p1.Check(p2):
                line = Line(p1, p2, mS, self.isOpen)
                geoms.append(line)

            if isinstance(corners[-1], (CircleArc, Line)) and idx2 != 0:
                geoms.append(corners[-1])

        for p, point in enumerate(self.points):
            prev = p - 1
            next = p + 1 if p + 1 < N else 0

            isOpen = point.isOpen

            if point.r == 0:
                corners.append(point)

            else:
                A, B, C = Fillet(
                    point.coord, coordinates[prev], coordinates[next], point.r
                )

                pA = Point(*A, isOpen=isOpen)
                pB = Point(*B, isOpen=isOpen)
                pC = Point(*C, isOpen=isOpen)

                corners.append(CircleArc(pA, pB, pC, meshSize=mS))

            if p > 0:
                Link(-2, -1)
            elif isinstance(corners[-1], (CircleArc, Line)):
                geoms.append(corners[-1])

        Link(-1, 0)

        contour = Contour(geoms, self.isHollow, self.isOpen).copy()
        contour.name = self.name + "_contour"
        # do the copy to unlink the points connexion with the list of points

        return contour

    def Get_coord_for_plot(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        return super().Get_coord_for_plot()

    @property
    def length(self) -> float:
        coord = self.coord
        length = np.linalg.norm(coord[1:] - coord[:-1], axis=1)
        length = np.sum(length)
        return length
