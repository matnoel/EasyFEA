# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Line class."""

import numpy as np

from ._utils import Point, AsPoint
from ._geom import _Geom

class Line(_Geom):

    __nbLine = 0

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
        
        self.pt1 = AsPoint(pt1)
        self.pt2 = AsPoint(pt2)

        Line.__nbLine += 1
        name = f"Line{Line.__nbLine}"
        _Geom.__init__(self, [self.pt1, self.pt2], meshSize, name, False, isOpen)
    
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