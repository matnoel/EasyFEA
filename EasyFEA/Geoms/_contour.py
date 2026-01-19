# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Contour class."""

from typing import Union
import numpy as np

from ..Utilities import _types

from ._utils import Point
from ._line import Line
from ._circle import CircleArc
from ._points import Points

from ._geom import _Geom


ContourCompatible = Union["Line", "CircleArc", "Points"]


class Contour(_Geom):
    """Contour class."""

    __NInstance = 0

    def _Init_Ninstance():
        Contour.__NInstance = 0

    def __init__(
        self,
        geoms: list[ContourCompatible],
        isHollow: bool = True,
        isOpen: bool = False,
    ):
        """Creates a contour from a list of line CircleArc and points.

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
        points: list["Point"] = []

        tol = 1e-12

        for i, geom in enumerate(geoms):
            assert isinstance(
                geom, (Line, CircleArc, Points)
            ), "Must give a list of lines and arcs or points."

            if i == 0:
                gap = tol
            elif i > 0 and i < len(geoms) - 1:
                # check that the starting point has the same coordinate as the last point of the previous object
                gap = np.linalg.norm(geom.points[0].coord - points[-1].coord).astype(
                    float
                )

                assert gap <= tol, "The contour must form a closed loop."
            else:
                # check that the end point of the last geometric object is the first point created.
                gap1 = np.linalg.norm(geom.points[0].coord - points[-1].coord)
                gap2 = np.linalg.norm(geom.points[-1].coord - points[0].coord)

                assert (
                    gap1 <= tol and gap2 <= tol
                ), "The contour must form a closed loop."

            # Add the first and last points
            points.extend([p for p in geom.points if p not in points])

        self.geoms = geoms

        Contour.__NInstance += 1
        name = f"Contour{Contour.__NInstance}"
        meshSize = np.mean([geom.meshSize for geom in geoms]).astype(float)
        _Geom.__init__(self, points, meshSize, name, isHollow, isOpen)

    def Get_coord_for_plot(self) -> tuple[_types.FloatArray, _types.FloatArray]:
        list_line: list[_types.FloatArray] = []
        list_point: list[_types.FloatArray] = []

        for geom in self.geoms:
            line, point = geom.Get_coord_for_plot()
            list_line.extend(line.ravel())
            list_point.extend(point.ravel())

        lines = np.array(list_line, dtype=float).reshape(-1, 3)
        points = np.array(list_point, dtype=float).reshape(-1, 3)

        return lines, points

    @property
    def length(self) -> float:
        return np.sum([geom.length for geom in self.geoms])
