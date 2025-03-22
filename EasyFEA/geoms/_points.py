# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Points class."""

from collections.abc import Iterable
import numpy as np

from ._utils import Point, Fillet
from ._geom import _Geom
from ._line import Line
from ._circle import CircleArc

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
                A, B, C = Fillet(point.coord, coordinates[prev], coordinates[next], point.r)

                pA = Point(*A, isOpen)
                pB = Point(*B, isOpen)
                pC = Point(*C, isOpen)

                corners.append(CircleArc(pA, pB, pC, meshSize=mS))
            
            if p > 0:
                Link(-2, -1)
            elif isinstance(corners[-1], (CircleArc, Line)):
                geoms.append(corners[-1])
                
        Link(-1, 0)

        from ._contour import Contour
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