# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.Geoms import Circle, CircleArc

S = np.sqrt(2) / 2


def test_Circle_rejects_a_negative_diameter():
    with pytest.raises(AssertionError):
        Circle((0, 0), -1)


def test_Circle_lies_in_the_plane_normal_to_n():
    circle = Circle((1, 2, 3), 2, n=(1, 1, 0))

    assert np.allclose((circle.coord - (1, 2, 3)) @ (1, 1, 0), 0)


def test_Circle_center_is_the_given_point_and_moves_with_the_circle():
    circle = Circle((1, 2, 3), 2)
    assert circle.center.Check((1, 2, 3))

    circle.Translate(1)
    assert circle.center.Check((2, 2, 3))


def test_Circle_pt1_is_at_r_along_the_local_x_axis():
    assert Circle((1, 2, 3), 2).pt1.Check((2, 2, 3))


def test_Circle_pt2_is_at_r_along_the_local_y_axis():
    assert Circle((1, 2, 3), 2).pt2.Check((1, 3, 3))


def test_Circle_pt3_is_at_minus_r_along_the_local_x_axis():
    assert Circle((1, 2, 3), 2).pt3.Check((0, 2, 3))


def test_Circle_pt4_is_at_minus_r_along_the_local_y_axis():
    assert Circle((1, 2, 3), 2).pt4.Check((1, 1, 3))


def test_Circle_pt2_follows_a_rotation():
    circle = Circle((1, 2, 3), 1)
    circle.Rotate(-90, (1, 2, 3), (1, 0, 0))

    assert circle.pt1.Check((1.5, 2, 3))
    assert circle.pt2.Check((1, 2, 2.5))


def test_Circle_diam_is_kept_by_a_rotation():
    circle = Circle((0, 0), 3, n=(1, 1, 1))
    assert circle.diam == pytest.approx(3)

    circle.Rotate(30, direction=(0, 1, 0))
    assert circle.diam == pytest.approx(3)


@pytest.mark.parametrize("n", [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
def test_Circle_n_is_the_given_normal(n):
    assert np.allclose(Circle((0, 0), 1, n=n).n, np.array(n) / np.linalg.norm(n))


def test_Circle_n_follows_a_rotation():
    circle = Circle((0, 0), 1)
    circle.Rotate(-90, direction=(1, 0, 0))

    assert np.allclose(circle.n, (0, 1, 0))


def test_Circle_length_is_the_perimeter():
    assert Circle((0, 0), 2).length == pytest.approx(2 * np.pi)


def test_Circle_Get_Contour_is_four_quarter_arcs():
    circle = Circle((1, 1), 2, meshSize=0.1, isFilled=True)
    contour = circle.Get_Contour()

    assert [type(geom) for geom in contour.geoms] == [CircleArc] * 4
    assert contour.length == pytest.approx(circle.length)
    assert contour.isFilled
    assert contour.meshSize == pytest.approx(0.1)


def test_CircleArc_center_R_and_P_build_the_same_arc():
    byCenter = CircleArc((3, 1), (-3, 1), center=(0, 1))
    byRadius = CircleArc((3, 1), (-3, 1), R=3)
    byPoint = CircleArc((3, 1), (-3, 1), P=(0, 4))

    assert byCenter.center.Check(byRadius.center)
    assert byCenter.center.Check(byPoint.center)


@pytest.mark.parametrize(
    "pt1, pt2, P, expected",
    [
        ((1, 0), (-1, 0), (0, 1), (0, 1)),
        ((1, 0), (-1, 0), (0, -1), (0, -1)),
        ((1, 0), (0, 1), (S, S), (S, S)),
        ((1, 0), (0, 1), (-1, 0), (-S, -S)),
        ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, 1)),
        ((0, 0, 1), (0, 0, -1), (1, 0, 0), (1, 0, 0)),
    ],
    ids=["upper half", "lower half", "minor", "major", "xz half", "chord along z"],
)
def test_CircleArc_goes_through_P(pt1, pt2, P, expected):
    assert CircleArc(pt1, pt2, P=P).pt3.Check(expected)


def test_CircleArc_needs_P_center_or_R():
    with pytest.raises(Exception, match="must give P, center or R"):
        CircleArc((1, 0), (0, 1))


@pytest.mark.parametrize(
    "pt2, center",
    [((1, 0), (0, 0)), ((0, 1), (1, 0)), ((0, 2), (0, 0))],
    ids=["same ends", "center on pt1", "off-center"],
)
def test_CircleArc_rejects_a_degenerate_arc(pt2, center):
    with pytest.raises(AssertionError):
        CircleArc((1, 0), pt2, center=center)


def test_CircleArc_pt1_is_the_start():
    assert CircleArc((1, 0), (0, 1), center=(0, 0)).pt1.Check((1, 0))


def test_CircleArc_pt2_is_the_end():
    assert CircleArc((1, 0), (0, 1), center=(0, 0)).pt2.Check((0, 1))


def test_CircleArc_center_is_the_given_center():
    assert CircleArc((2, 1), (1, 2), center=(1, 1)).center.Check((1, 1))


@pytest.mark.parametrize(
    "coef, expected", [(1, (S, S)), (-1, (-S, -S))], ids=["coef 1", "coef -1"]
)
def test_CircleArc_pt3_is_on_the_bisector(coef: int, expected):
    arc = CircleArc((1, 0), (0, 1), center=(0, 0), coef=coef)

    assert arc.pt3.Check(expected)


def test_CircleArc_pt3_of_a_half_arc_is_on_the_circle():
    arc = CircleArc((1, 0), (-1, 0), center=(0, 0))

    assert arc.pt3.Check((0, -1))


@pytest.mark.parametrize(
    "pt1, pt2, expected",
    [((1, 0), (0, 1), (0, 0, 1)), ((0, 1), (1, 0), (0, 0, -1))],
    ids=["counterclockwise", "clockwise"],
)
def test_CircleArc_n_follows_the_drawing_direction(pt1, pt2, expected):
    assert np.allclose(CircleArc(pt1, pt2, center=(0, 0)).n, expected)


def test_CircleArc_n_of_a_half_arc_goes_through_pt3():
    arc = CircleArc((3, 1), (-3, 1), center=(0, 1))

    assert np.allclose(arc.n, (0, 0, -1))


@pytest.mark.parametrize(
    "pt2, expected", [((0, 1), np.pi / 2), ((-1, 0), np.pi), ((S, S), np.pi / 4)]
)
def test_CircleArc_angle_spans_the_two_ends(pt2, expected):
    assert CircleArc((1, 0), pt2, center=(0, 0)).angle == pytest.approx(expected)


def test_CircleArc_r_is_the_radius():
    assert CircleArc((3, 1), (1, 3), center=(1, 1)).r == pytest.approx(2)


def test_CircleArc_length_is_r_times_the_angle():
    assert CircleArc((2, 0), (0, 2), center=(0, 0)).length == pytest.approx(np.pi)
