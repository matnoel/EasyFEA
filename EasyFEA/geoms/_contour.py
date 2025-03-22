# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Contour class."""

from typing import Union
import numpy as np

from ._utils import Point
from ._geom import _Geom
from ._line import Line
from ._circle import CircleArc
from ._points import Points

ContourCompatible = Union[Line, CircleArc, Points]

class Contour(_Geom):

    __nbContour = 0

    def __init__(self, geoms: list[ContourCompatible], isHollow=True, isOpen=False):
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

            assert isinstance(geom, ContourCompatible), "Must give a list of lines and arcs or points."

            if i == 0:
                gap = tol
            elif i > 0 and i < len(geoms)-1:
                # check that the starting point has the same coordinate as the last point of the previous object
                gap = np.linalg.norm(geom.points[0].coord - points[-1].coord)

                assert gap <= tol, "The contour must form a closed loop."
            else:
                # check that the end point of the last geometric object is the first point created.
                gap1 = np.linalg.norm(geom.points[0].coord - points[-1].coord)
                gap2 = np.linalg.norm(geom.points[-1].coord - points[0].coord)

                assert gap1 <= tol and gap2 <= tol, "The contour must form a closed loop."

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