# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Geometries shared by these tests and by Viz, which draws them."""

from EasyFEA.Geoms import Point, Points


def rounded_corners() -> Points:
    """A 5 × 5 square, one corner filleted (r = 2), one chamfered (r = -3)."""
    return Points([(0, 0), (5, 0), Point(5, 5, r=2), Point(0, 5, r=-3)])
