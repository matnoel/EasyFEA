# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import plt, np
from EasyFEA.Geoms import Point, Line


class TestLine:

    def test_line(sef):

        p1 = Point()
        p2 = Point(1)

        line = Line(p1, p2, 1 / 3, isOpen=True)

        assert line.length == 1
        assert line.meshSize == 1 / 3
        assert line.isOpen == True
        assert np.linalg.norm(line.unitVector - (1, 0, 0)) == 0

        line.Translate(1, 2, 3)
        assert line.length == 1
        assert p1.Check((1, 2, 3))
        assert p2.Check((2, 2, 3))

        line.Rotate(90, p1.coord, (0, 0, 1))
        assert line.length == 1
        assert np.linalg.norm(line.unitVector - (0, 1, 0)) == 0
