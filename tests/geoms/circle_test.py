# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import plt, np
from EasyFEA.Geoms import Point, Circle, CircleArc

class TestCircle:

    def test_circle(self):

        center = Point()
        circle = Circle(center, 1., 1/5)
        assert np.linalg.norm(circle.n - (0,0,1)) == pytest.approx(0)

        assert circle.length == np.pi * circle.diam

        circle.Translate(1, 2, 3)
        assert circle.center.Check((1,2,3))
        assert circle.length == np.pi * circle.diam
        assert circle.pt1.Check((1.5,2,3))
        assert circle.pt2.Check((1,2.5,3))
        assert circle.pt3.Check((.5,2,3))
        assert circle.pt4.Check((1,1.5,3))
        assert np.linalg.norm(circle.n - (0,0,1)) == pytest.approx(0)

        circle.Rotate(-90, center, (1,0,0))
        assert circle.length == np.pi * circle.diam
        assert circle.pt1.Check((1.5,2,3))
        assert circle.pt2.Check((1,2,2.5))
        assert circle.pt3.Check((.5,2,3))
        assert circle.pt4.Check((1,2,3.5))
        assert np.linalg.norm(circle.n - (0,1,0)) == pytest.approx(0)

class TestCircleArc:

    def test_circle_arc(self):

        circleArc = CircleArc(Point(3,1), Point(-3,1), center=Point(0,1))       
        
        circleArc2 = CircleArc(Point(3,1), Point(-3,1), R=3)
        
        circleArc3 = CircleArc(Point(3,1), Point(-3,1), P=Point(0,4))
        print(circleArc3.center)
        
        assert circleArc.center.Check(circleArc2.center)
        assert circleArc.center.Check(circleArc3.center)

        assert circleArc.r == circleArc2.r == circleArc3.r == 3
        
        assert np.linalg.norm(circleArc.n - (0,0,-1)) == pytest.approx(0)
        assert np.linalg.norm(circleArc2.n - (0,0,-1)) == pytest.approx(0)

        assert  np.abs(circleArc.angle) == pytest.approx(np.pi)
        assert circleArc.length == pytest.approx(np.abs(circleArc.angle * circleArc.r))