# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.Geoms import Point, Line


def test_Line_keeps_meshSize_and_isOpen():
    line = Line((0, 0), (1, 0), 1 / 3, isOpen=True)

    assert line.meshSize == 1 / 3
    assert line.isOpen
    assert not line.isFilled


def test_Line_pt1_is_the_given_first_point():
    p1 = Point(1, 2)
    line = Line(p1, (3, 4))
    line.Translate(1)

    assert line.pt1 is p1
    assert p1.Check((2, 2))


def test_Line_pt2_is_the_given_second_point():
    p2 = Point(3, 4)
    line = Line((1, 2), p2)

    assert line.pt2 is p2


def test_Line_unitVector_points_from_pt1_to_pt2():
    line = Line((1, 1), (4, 5))
    assert np.allclose(line.unitVector, (0.6, 0.8, 0))

    line.Rotate(90, line.pt1.coord)
    assert np.allclose(line.unitVector, (-0.8, 0.6, 0))


def test_Line_length_is_kept_by_a_rotation():
    line = Line((1, 1), (4, 5))
    assert line.length == pytest.approx(5)

    line.Rotate(33, direction=(1, 2, 3))
    assert line.length == pytest.approx(5)


def test_Line_distance_between_two_points():
    assert Line.distance(Point(1, 2, 3), Point(3, 5, 9)) == pytest.approx(7)


def test_Line_get_unitVector_between_two_points():
    unitVector = Line.get_unitVector(Point(1, 2, 3), Point(3, 5, 9))

    assert np.allclose(unitVector, np.array([2, 3, 6]) / 7)
