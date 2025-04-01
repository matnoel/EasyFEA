# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Domain class."""

import numpy as np

from ._utils import Point, AsPoint
from ._geom import _Geom

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
        
        self.pt1 = AsPoint(pt1)        
        self.pt2 = AsPoint(pt2)

        Domain.__nbDomain += 1
        name = f"Domain{Domain.__nbDomain}"
        # a domain can't be open
        _Geom.__init__(self, [self.pt1, self.pt2], meshSize, name, isHollow, False)

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