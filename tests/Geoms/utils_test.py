# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.Geoms import (
    Point,
    AsPoint,
    AsCoords,
    Normalize,
    Distance_To_Polyline,
    Translate,
    Rotate,
    Symmetry,
    Circle,
    Circle_Triangle,
    Circle_Coords,
    Points_Intersect_Circles,
    Angle_Between,
    Jacobian_Matrix,
    Fillet,
)

ONES = [
    pytest.param(1, id="int"),
    pytest.param(1.0, id="float"),
    pytest.param([1, 1, 1], id="list"),
    pytest.param((1, 1, 1), id="tuple"),
    pytest.param(Point(1, 1, 1), id="Point"),
]

# (start, theta [deg], center, direction, expected)
ROTATIONS = [
    ((1, 0, 0), 90, (0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), -90, (0, 0, 0), (0, 0, 1), (1, 0, 0)),
    ((1, 0, 0), 180, (0, 0, 0), (0, 0, 1), (-1, 0, 0)),
    ((1, 0, 0), 90, (0, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 0, 1), 90, (0, 0, 0), (1, 0, 0), (0, -1, 0)),
    ((1, 0, 0), 180, (0, 0, 0), (1, 0, 1), (0, 0, 1)),
    ((1, 0, 0), 90, (-1, 0, 0), (0, -1, 0), (-1, 0, 2)),
]

# (start, point on the plane, normal, expected)
SYMMETRIES = [
    ((1, 1, 0), (0, 0, 0), (0, 1, 0), (1, -1, 0)),
    ((1, 1, 0), (0, 0, 0), (-1, 0, 0), (-1, 1, 0)),
    ((-1, -1, 0), (0, 0, 0), (1, 1, 0), (1, 1, 0)),
    ((3, 1, 0), (1, 0, 0), (1, 0, 0), (-1, 1, 0)),
]


def test_Point_defaults_to_the_origin_closed_and_sharp():
    point = Point()

    assert point.Check((0, 0, 0))
    assert not point.isOpen
    assert point.r == 0


@pytest.mark.parametrize("value", ONES)
def test_Point_adds_a_scalar_or_vector_to_every_component(value):
    assert (Point(0, 0, 0) + value).Check((1, 1, 1))
    assert (value + Point(0, 0, 0)).Check((1, 1, 1))


@pytest.mark.parametrize("value", ONES)
def test_Point_subtracts_a_scalar_or_vector_from_every_component(value):
    assert (Point(1, 1, 1) - value).Check((0, 0, 0))


@pytest.mark.parametrize(
    "operation, expected",
    [
        (lambda p: p * 2, (4, 8, 12)),
        (lambda p: 2 * p, (4, 8, 12)),
        (lambda p: p * [1, 2, 3], (2, 8, 18)),
        (lambda p: p / 2, (1, 2, 3)),
        (lambda p: p // 4, (0, 1, 1)),
    ],
    ids=["times scalar", "scalar times", "times vector", "divide", "floor divide"],
)
def test_Point_arithmetic_is_componentwise(operation, expected):
    assert operation(Point(2, 4, 6)).Check(expected)


def test_Point_x_sets_only_the_first_component():
    point = Point(1, 2, 3)
    point.x = 7

    assert point.x == 7
    assert point.Check((7, 2, 3))


def test_Point_y_sets_only_the_second_component():
    point = Point(1, 2, 3)
    point.y = 7

    assert point.y == 7
    assert point.Check((1, 7, 3))


def test_Point_z_sets_only_the_third_component():
    point = Point(1, 2, 3)
    point.z = 7

    assert point.z == 7
    assert point.Check((1, 2, 7))


def test_Point_x_rejects_a_non_scalar():
    point = Point()

    with pytest.raises(AssertionError):
        point.x = (1, 2)  # type: ignore [assignment]


def test_Point_coord_is_a_copy():
    point = Point(1, 2, 3)
    coord = point.coord
    coord[0] = 9

    assert point.Check((1, 2, 3))


def test_Point_coord_setter_keeps_isOpen_and_r():
    point = Point(
        1,
        1,
        1,
        isOpen=True,
        r=2,
    )
    point.coord = (1, 3, 4)

    assert point.Check((1, 3, 4))
    assert point.isOpen
    assert point.r == 2


@pytest.mark.parametrize(
    "point, other, expected",
    [
        (Point(1, 2, 3), (1, 2, 3), True),
        (Point(1, 2, 3), (1, 2, 3.1), False),
        (Point(1e6), (1e6 + 1e-7, 0, 0), True),
        (Point(), (1e-13, 0, 0), True),
        (Point(), (1e-6, 0, 0), False),
    ],
    ids=["same", "different", "relative to the norm", "origin", "origin, off"],
)
def test_Point_Check_compares_within_a_relative_tolerance(point, other, expected):
    assert point.Check(other) == expected


def test_Point_Translate_shifts_by_the_offset():
    point = Point()
    point.Translate(-1, 2)

    assert point.Check((-1, 2, 0))


@pytest.mark.parametrize("start, theta, center, direction, expected", ROTATIONS)
def test_Point_Rotate_about_an_axis(start, theta: float, center, direction, expected):
    point = Point(*start)
    point.Rotate(theta, center, direction)

    assert point.Check(expected)


@pytest.mark.parametrize("start, plane, n, expected", SYMMETRIES)
def test_Point_Symmetry_through_a_plane(start, plane, n, expected):
    point = Point(*start)
    point.Symmetry(plane, n)

    assert point.Check(expected)


def test_Point_copy_is_a_new_point_with_the_same_attributes():
    point = Point(
        1,
        3,
        4,
        isOpen=True,
        r=2,
    )
    copy = point.copy()
    copy.Translate(1)

    assert copy is not point
    assert point.Check((1, 3, 4))
    assert copy.isOpen
    assert copy.r == 2


def test_AsPoint_returns_a_Point_as_is():
    point = Point(1, 2)

    assert AsPoint(point) is point


@pytest.mark.parametrize("coords", [(1, 2), [1, 2, 0], np.array([1.0, 2.0])])
def test_AsPoint_pads_coordinates_with_zeros(coords):
    assert AsPoint(coords).Check((1, 2, 0))


def test_AsPoint_rejects_more_than_3_coordinates():
    with pytest.raises(AssertionError):
        AsPoint((1, 2, 3, 4))


def test_AsPoint_rejects_a_scalar():
    with pytest.raises(NotImplementedError):
        AsPoint(1)  # type: ignore [arg-type]


@pytest.mark.parametrize(
    "value, expected",
    [
        (2, (2, 2, 2)),
        (2.5, (2.5, 2.5, 2.5)),
        ((1, 2), (1, 2, 0)),
        (Point(1, 2, 3), (1, 2, 3)),
    ],
    ids=["int", "float", "short tuple", "Point"],
)
def test_AsCoords_returns_a_3d_vector(value, expected):
    assert np.array_equal(AsCoords(value), expected)


def test_AsCoords_keeps_an_array_of_vectors():
    coords = np.arange(6.0).reshape(2, 3)

    assert np.array_equal(AsCoords(coords), coords)


def test_AsCoords_pads_rows_of_2d_vectors():
    assert np.array_equal(AsCoords([(1, 2), (3, 4)]), [(1, 2, 0), (3, 4, 0)])


@pytest.mark.parametrize(
    "value", [(1, 2, 3, 4), np.zeros((2, 4))], ids=["4", "4d rows"]
)
def test_AsCoords_rejects_more_than_3_components(value):
    with pytest.raises(AssertionError):
        AsCoords(value)


def test_AsCoords_rejects_an_unsupported_type():
    with pytest.raises(NotImplementedError):
        AsCoords(None)  # type: ignore [arg-type]


def test_Normalize_a_vector_has_a_unit_norm():
    assert np.allclose(Normalize([3, 4, 0]), (0.6, 0.8, 0))


def test_Normalize_leaves_zero_vectors():
    rows = Normalize(np.array([[0.0, 0, 2], [0, 0, 0]]))

    assert np.array_equal(rows, [[0, 0, 1], [0, 0, 0]])
    assert np.array_equal(Normalize([0, 0, 0]), [0, 0, 0])


def test_Normalize_rejects_a_3d_array():
    with pytest.raises(Exception, match="wrong size"):
        Normalize(np.ones((2, 2, 3)))


@pytest.mark.parametrize(
    "coord, expected",
    [((1, 1, 0), 1.0), ((-3, 4, 0), 5.0), ((2, -1, 0), 1.0), ((5, 0, 0), 3.0)],
    ids=["above a segment", "before the start", "below a segment", "past the end"],
)
def test_Distance_To_Polyline_is_the_distance_to_the_nearest_segment(coord, expected):
    polyline = [(0, 0, 0), (2, 0, 0), (2, 3, 0)]

    assert Distance_To_Polyline(coord, polyline) == pytest.approx(expected)


def test_Distance_To_Polyline_of_one_point_is_the_distance_to_it():
    distances = Distance_To_Polyline([(3, 4, 0), (0, 0, 0)], [(0, 0, 0)])

    assert np.allclose(distances, [5, 0])


def test_Distance_To_Polyline_skips_a_repeated_point():
    polyline = [(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0)]

    assert Distance_To_Polyline((2, 0.5), polyline) == pytest.approx(1)


def test_Translate_shifts_every_row():
    coord = np.array([[0.0, 0, 0], [1, 2, 3]])

    assert np.array_equal(Translate(coord, 1, -1, 2), [[1, -1, 2], [2, 1, 5]])


@pytest.mark.parametrize("start, theta, center, direction, expected", ROTATIONS)
def test_Rotate_about_an_axis(start, theta: float, center, direction, expected):
    rotated = Rotate(np.array(start, dtype=float), theta, center, direction)

    assert np.allclose(rotated, [expected])


@pytest.mark.parametrize("start, plane, n, expected", SYMMETRIES)
def test_Symmetry_through_a_plane(start, plane, n, expected):
    reflected = Symmetry(np.array(start, dtype=float), plane, n)

    assert np.allclose(reflected, [expected])


def test_Circle_Triangle_is_equidistant_from_the_three_points():
    center = np.array([1.0, 2.0, 0.5])
    angles = np.array([0.3, 2.0, 4.0])
    p1, p2, p3 = center + 3 * np.stack(
        [np.cos(angles), np.sin(angles), 0 * angles], axis=1
    )

    assert np.allclose(Circle_Triangle(p1, p2, p3), center)


@pytest.mark.parametrize(
    "coord, R, n, expected",
    [
        ([(2, 0, 0), (0, 2, 0)], 2, (0, 0, 1), (0, 0, 0)),
        ([(0, 2, 0), (2, 0, 0)], 2, (0, 0, 1), (2, 2, 0)),
        ([(2, 0, 0), (0, 2, 0)], 2, (0, 0, -1), (2, 2, 0)),
        ([(2, 0, 1), (0, 2, 1)], 2, (0, 0, 1), (0, 0, 1)),
        ([(3, 1, 0), (-3, 1, 0)], 3, (0, 0, 1), (0, 1, 0)),
    ],
    ids=["quarter", "reversed", "flipped normal", "shifted plane", "half"],
)
def test_Circle_Coords_is_the_center_seeing_the_points_counterclockwise(
    coord, R, n, expected
):
    assert np.allclose(Circle_Coords(np.array(coord, dtype=float), R, n), expected)


def test_Circle_Coords_rejects_a_radius_below_half_the_chord():
    with pytest.raises(AssertionError):
        Circle_Coords(np.array([(1.0, 0, 0), (0, 1, 0)]), 0.5, (0, 0, 1))


def test_Circle_Coords_rejects_a_normal_along_the_chord():
    with pytest.raises(AssertionError):
        Circle_Coords(np.array([(0.0, 0, 0), (0, 0, 2)]), 2, (0, 0, 1))


def test_Points_Intersect_Circles_crossing_circles_meet_twice():
    circle1 = Circle((0, 0), 2)
    circle2 = Circle((1, 0), 2)
    points = Points_Intersect_Circles(circle1, circle2)

    expected = [(0.5, -np.sqrt(3) / 2, 0), (0.5, np.sqrt(3) / 2, 0)]
    assert np.allclose(points, expected)


def test_Points_Intersect_Circles_tangent_circles_meet_once():
    circle1 = Circle((0, 0), 2)
    circle2 = Circle((2, 0), 2)

    assert np.allclose(Points_Intersect_Circles(circle1, circle2), [(1, 0, 0)])


@pytest.mark.parametrize(
    "center2, diam2",
    [((5, 0), 2), ((0.2, 0), 0.5), ((0, 0), 2)],
    ids=["separated", "nested", "same"],
)
def test_Points_Intersect_Circles_without_crossing_is_None(center2, diam2):
    circle1 = Circle((0, 0), 2)
    circle2 = Circle(center2, diam2)

    assert Points_Intersect_Circles(circle1, circle2) is None


@pytest.mark.parametrize(
    "b, expected",
    [((0, 1), np.pi / 2), ((1, 1), np.pi / 4), ((2, 0), 0), ((-1, 0), np.pi)],
    ids=["perpendicular", "diagonal", "same direction", "opposite"],
)
def test_Angle_Between_two_vectors(b, expected):
    assert Angle_Between((1, 0), b) == pytest.approx(expected)


def test_Angle_Between_rows_of_vectors():
    b = np.array([(0.0, 1, 0), (2, 0, 0), (-1, 0, 0)])

    assert np.allclose(Angle_Between(np.array([1.0, 0, 0]), b), [np.pi / 2, 0, np.pi])
    assert np.allclose(Angle_Between(b, b), 0)


def test_Jacobian_Matrix_columns_are_the_orthonormal_frame():
    i = np.array([2.0, 0, 0])
    k = np.array([0.0, 0, 3])

    assert np.allclose(Jacobian_Matrix(i, k), np.eye(3))

    tilted = Jacobian_Matrix([1, 1, 0], [0, 0, 1])
    assert np.allclose(tilted.T @ tilted, np.eye(3))
    assert np.allclose(tilted[:, 0], np.array([1, 1, 0]) / np.sqrt(2))


def test_Fillet_rounds_a_right_angle_with_its_radius():
    P0, P1, P2 = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    A, B, C = Fillet(P0, P1, P2, 1)

    assert np.allclose([A, B, C], [(1, 0, 0), (0, 1, 0), (1, 1, 0)])


def test_Fillet_with_a_negative_radius_keeps_the_corner():
    P0, P1, P2 = np.array([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    A, B, C = Fillet(P0, P1, P2, -1)

    assert np.allclose([A, B, C], [(1, 0, 0), (0, 1, 0), (0, 0, 0)])
