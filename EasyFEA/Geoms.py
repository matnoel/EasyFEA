# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the geometric classes used to build meshes."""

from .geoms._utils import Point

from .geoms._geom import _Geom

from .geoms._points import Points

from .geoms._domain import Domain

from .geoms._line import Line

from .geoms._circle import Circle, CircleArc

from .geoms._contour import Contour
