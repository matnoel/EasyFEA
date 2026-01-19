# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the geometric functions used to build meshes."""

from ._circle import Circle, CircleArc
from ._contour import Contour
from ._domain import Domain
from ._geom import _Geom
from ._line import Line
from ._points import Points

from ._utils import (
    Point,
    AsPoint,
    AsCoords,
    Normalize,
    Translate,
    Rotate,
    Symmetry,
    Circle_Triangle,
    Circle_Coords,
    Points_Intersect_Circles,
    Angle_Between,
    Jacobian_Matrix,
    Fillet,
)


def _Init_Geoms_NInstance():
    """Initalizes the number of instance for each geom classes."""
    Circle._Init_Ninstance()
    CircleArc._Init_Ninstance()
    Contour._Init_Ninstance()
    Domain._Init_Ninstance()
    Line._Init_Ninstance()
    Points._Init_Ninstance()
