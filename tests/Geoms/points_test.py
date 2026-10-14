# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Geoms import Point, Points, Line, CircleArc


def _square(r: float = 0.0, corner: int = 2) -> Points:
    corners: list = [(0, 0), (2, 0), (2, 2), (0, 2)]
    x, y = corners[corner]
    corners[corner] = Point(x, y, r=r)
    return Points(corners, meshSize=0.2)


def test_Points_takes_points_or_coordinates():
    points = Points([Point(1, 2), (3, 4), [5, 6, 7]], meshSize=0.5, isOpen=True)

    assert np.array_equal(points.coord, [(1, 2, 0), (3, 4, 0), (5, 6, 7)])
    assert points.isOpen
    assert points.meshSize == 0.5


def test_Points_rejects_a_non_iterable():
    with pytest.raises(AssertionError):
        Points(Point())  # type: ignore [arg-type]


def test_Points_pt1_is_the_first_point():
    assert _square().pt1.Check((0, 0))


def test_Points_pt2_is_the_last_point():
    assert _square().pt2.Check((0, 2))


def test_Points_Get_Contour_of_sharp_corners_is_lines():
    contour = _square().Get_Contour()

    assert [type(geom) for geom in contour.geoms] == [Line] * 4
    assert contour.length == pytest.approx(8)


@pytest.mark.parametrize("corner", [0, 2, 3], ids=["first", "inner", "last"])
def test_Points_Get_Contour_rounds_a_corner_with_its_radius(corner: int):
    contour = _square(r=0.5, corner=corner).Get_Contour()

    assert sum(isinstance(geom, CircleArc) for geom in contour.geoms) == 1
    assert contour.length == pytest.approx(8 - 2 * 0.5 + np.pi / 4)


@pytest.mark.parametrize(
    "r, enclosed", [(1, [True, True, False]), (-1, [True, False, False])]
)
def test_Points_Get_Contour_rounds_outward_or_inward_with_the_radius_sign(
    r: float, enclosed
):
    """r > 0 rounds the corner (center inside), r < 0 cuts it with an arc centered on the corner."""

    contour = _square(r=r).Get_Contour()

    probes = [(1.2, 1.2), (1.5, 1.5), (1.9, 1.9)]
    assert contour.Encloses(probes).tolist() == enclosed


def test_Points_Get_Contour_is_a_named_copy():
    points = _square(r=0.5)
    contour = points.Get_Contour()
    contour.Translate(10)

    assert contour.name == points.name + "_contour"
    assert points.pt1.Check((0, 0))


def test_Points_length_follows_the_open_polyline():
    assert _square().length == pytest.approx(6)
