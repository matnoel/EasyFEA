# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the geometric functions used to build meshes."""

from ._utils import (
    Point, AsPoint, AsCoords, Normalize,
    Translate, Rotate, Symmetry,
    Circle_Triangle, Circle_Coords, Points_Intersect_Circles,
    Angle_Between, Jacobian_Matrix,
    Fillet
)