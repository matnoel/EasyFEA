# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.Geoms import Point

ONES = [
    pytest.param(1, id="int"),
    pytest.param(1.0, id="float"),
    pytest.param([1, 1, 1], id="list"),
    pytest.param((1, 1, 1), id="tuple"),
    pytest.param(Point(1, 1, 1), id="Point"),
]


def test_setting_coord_keeps_isOpen_and_r():
    point = Point(1, 1, 1, isOpen=True, r=2)
    point.coord = (1, 3, 4)

    assert point.Check((1, 3, 4))
    assert point.isOpen
    assert point.r == 2


def test_copy_is_a_new_point_with_the_same_attributes():
    point = Point(1, 3, 4, isOpen=True, r=2)
    copy = point.copy()

    assert copy is not point
    assert copy.Check(point)
    assert copy.isOpen
    assert copy.r == 2


@pytest.mark.parametrize("value", ONES)
def test_add_a_scalar_or_vector_to_every_component(value):
    assert (Point(0, 0, 0) + value).Check((1, 1, 1))


@pytest.mark.parametrize("value", ONES)
def test_subtract_a_scalar_or_vector_from_every_component(value):
    assert (Point(1, 1, 1) - value).Check((0, 0, 0))


@pytest.mark.parametrize(
    "operation, expected",
    [
        (lambda p: p * 2, (4, 8, 12)),
        (lambda p: p * [1, 2, 3], (2, 8, 18)),
        (lambda p: p / 2, (1, 2, 3)),
        (lambda p: p // 4, (0, 1, 1)),
    ],
    ids=["times scalar", "times vector", "divide", "floor divide"],
)
def test_arithmetic_is_componentwise(operation, expected):
    assert operation(Point(2, 4, 6)).Check(expected)


@pytest.mark.parametrize(
    "axis, expected", [("x", (7, 2, 3)), ("y", (1, 7, 3)), ("z", (1, 2, 7))]
)
def test_setting_one_component_leaves_the_others(axis: str, expected):
    point = Point(1, 2, 3)
    setattr(point, axis, 7)

    assert point.Check(expected)


def test_translate_shifts_by_the_offset():
    point = Point()
    point.Translate(-1, 2)

    assert point.Check((-1, 2, 0))


@pytest.mark.parametrize(
    "start, theta, center, direction, expected",
    [
        ((1, 0, 0), 90, (0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), -90, (0, 0, 0), (0, 0, 1), (1, 0, 0)),
        ((1, 0, 0), 180, (0, 0, 0), (0, 0, 1), (-1, 0, 0)),
        ((1, 0, 0), 90, (0, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 0, 1), 90, (0, 0, 0), (1, 0, 0), (0, -1, 0)),
        ((1, 0, 0), 180, (0, 0, 0), (1, 0, 1), (0, 0, 1)),
        ((1, 0, 0), 90, (-1, 0, 0), (0, -1, 0), (-1, 0, 2)),
    ],
)
def test_rotate_about_an_axis(start, theta: float, center, direction, expected):
    point = Point(*start)
    point.Rotate(theta, center, direction)

    assert point.Check(expected)


@pytest.mark.parametrize(
    "start, n, expected",
    [
        ((1, 1, 0), (0, 1, 0), (1, -1, 0)),
        ((1, 1, 0), (-1, 0, 0), (-1, 1, 0)),
        ((-1, -1, 0), (1, 1, 0), (1, 1, 0)),
    ],
)
def test_symmetry_through_a_plane(start, n, expected):
    point = Point(*start)
    point.Symmetry((0, 0, 0), n)

    assert point.Check(expected)


def test_symmetry_through_a_shifted_plane():
    point = Point(3, 1, 0)
    point.Symmetry((1, 0, 0), (1, 0, 0))

    assert point.Check((-1, 1, 0))
