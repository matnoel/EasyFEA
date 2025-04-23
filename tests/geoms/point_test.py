# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import plt, np
from EasyFEA.Geoms import _Geom, Point, Line, Circle, CircleArc, Points, Domain, Contour

class TestPoint:
    
    def test_point_properties(self):

        point = Point(1,1,1, isOpen=True, r=2)

        point.coord = (1,3,4)
        assert point.isOpen == True
        assert point.r == 2

        copy = point.copy()
        assert copy.isOpen == True
        assert copy.r == 2

        new = copy + 2
        assert new.Check((3,5,6))

        new = copy + (0,0,1)
        assert new.Check((1,3,5))

    def test_point_operations(self):
        """test on points"""
        
        # operations + and -
        p0 = Point(0,0,0)

        for val in [1,1.0,[1,1,1],(1,1,1),Point(1,1,1)]:
            # + and -            
            p1: Point = p0 + val
            assert p1.Check([1]*3)

            p2: Point = p1 - val
            assert p2.Check([0]*3)
        
        # operations *, / and //
        p3 = Point(1, 1, 1)

        p4 = p3 * 2
        assert p4.Check([2]*3)

        p5 = p4 / 2
        assert p5.Check([1]*3)

        p6 = p4 // 2
        assert p6.Check([1]*3)

        p7 = p3 * [1,2,3]
        assert p7.Check([1,2,3])

        # new coord
        p3.coord = [0,0.4,0.5]
        assert p3.Check([0,0.4,0.5])

        # new coord
        p3.x = 1
        assert p3.Check([1,0.4,0.5])
        p3.y = 2.
        assert p3.Check([1,2.,0.5])
        p3.z = -1
        assert p3.Check([1,2.,-1])

        # copy
        pC1 = Point()
        pC2 = pC1.copy()
        assert pC1 is not pC2

        # translate
        pT1 = Point()
        pT1.Translate(-1,2)
        assert pT1.Check((-1,2,0))

        # rotate
        pR1 = Point(1)

        pR1.Rotate(90, (0,0,0), (0,0,1))
        assert pR1.Check((0,1,0))

        pR1.Rotate(-90, (0,0,0), (0,0,1))
        assert pR1.Check((1,0,0))

        pR1.Rotate(180, (0,0,0), (0,0,1))
        assert pR1.Check((-1,0,0))

        pR1.Rotate(-180, (0,0,0), (0,0,1))
        assert pR1.Check((1,0,0))

        pR1.Rotate(90, (0,0,0), (0,-1,0))
        assert pR1.Check((0,0,1))

        pR1.Rotate(90, (0,0,0), (1,0,0))
        assert pR1.Check((0,-1,0))

        pR2 = Point(1)
        pR2.Rotate(180, (0,0,0), (1,0,1))
        assert pR2.Check((0,0,1))

        pR3 = Point(1)
        pR3.Rotate(90, (-1,0,0), (0,-1,0))
        assert pR3.Check((-1,0,2))

        # symmetry
        pS1 = Point(1, 1)

        pS1.Symmetry((0,0,0), (0,1,0))
        assert pS1.Check((1,-1,0))

        pS1.Symmetry((0,0,0), (1,0,0))
        assert pS1.Check((-1,-1,0))

        pS1.Symmetry((0,0,0), (1,1,0))
        assert pS1.Check((1,1,0))

        pS1.Symmetry((0,0,0), (-1,0,0))
        assert pS1.Check((-1,1,0))