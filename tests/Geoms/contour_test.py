# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.Geoms import Line, CircleArc, Points, Circle, Contour

GAP_IN_THE_MIDDLE = [Line((0, 0), (1, 0)), Line((1, 1), (0, 1)), Line((0, 1), (0, 0))]
NOT_CLOSED = [Line((0, 0), (1, 0)), Line((1, 0), (1, 1)), Line((1, 1), (0, 1))]


def test_Contour_shares_the_points_between_consecutive_geoms():
    bottom = Line((0, 0), (5, 0), meshSize=1)
    arc = CircleArc(bottom.pt2, (-5, 0), P=(0, 5), meshSize=2)
    back = Line(arc.pt2, bottom.pt1, meshSize=3)
    contour = Contour([bottom, arc, back], isFilled=True)

    assert len(contour.points) == 5  # bottom: 2, arc: center + midpoint + end
    assert contour.isFilled


def test_Contour_meshSize_is_the_mean_of_its_geoms():
    bottom = Line((0, 0), (5, 0), meshSize=1)
    arc = CircleArc(bottom.pt2, (-5, 0), P=(0, 5), meshSize=2)
    back = Line(arc.pt2, bottom.pt1, meshSize=3)
    contour = Contour([bottom, arc, back])

    assert contour.meshSize == pytest.approx(2)


def test_Contour_mixes_lines_arcs_and_points():
    spline = Points([(0, 0), (1, -0.5), (2, 0)])
    arc = CircleArc((2, 0), (0, 0), center=(1, 0))
    contour = Contour([spline, arc])

    assert contour.length == pytest.approx(2 * np.sqrt(1.25) + np.pi)


@pytest.mark.parametrize(
    "geoms",
    [GAP_IN_THE_MIDDLE, NOT_CLOSED],
    ids=["gap in the middle", "not closed"],
)
def test_Contour_rejects_an_open_loop(geoms):
    with pytest.raises(AssertionError, match="closed loop"):
        Contour(geoms)


def test_Contour_rejects_a_circle():
    circle = Circle((0, 0), 1)

    with pytest.raises(AssertionError):
        Contour([circle])  # type: ignore [list-item]


def test_Contour_length_sums_its_geoms():
    bottom = Line((0, 0), (5, 0))
    arc = CircleArc((5, 0), (-5, 0), P=(0, 5))
    back = Line((-5, 0), (0, 0))
    contour = Contour([bottom, arc, back])

    assert contour.length == pytest.approx(10 + 5 * np.pi)
