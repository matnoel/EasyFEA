# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import plt, np
from EasyFEA.Geoms import _Geom, Point, Line, Circle, CircleArc, Points, Domain, Contour

class TestPoints:
    
    def test_points_properties(self):

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

    def test_points_operations(self):
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

class TestLine:

    def test_line(sef):

        p1 = Point()
        p2 = Point(1)
        
        line = Line(p1, p2, 1/3, isOpen=True)

        assert line.length == 1
        assert line.meshSize == 1/3
        assert line.isOpen == True
        assert np.linalg.norm(line.unitVector - (1,0,0)) == 0

        line.Translate(1, 2, 3)
        assert line.length == 1
        assert p1.Check((1, 2, 3))
        assert p2.Check((2, 2, 3))

        line.Rotate(90, p1.coord, (0,0,1))
        assert line.length == 1
        assert np.linalg.norm(line.unitVector - (0,1,0)) == 0

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

class TestGeoms:
    
    def test_move_and_plot_geom_objects(self):

        line = Line(Point(), Point(5,1))

        x = np.linspace(0, 5, 10)
        y = np.sin(x)

        points = Points([Point(x[i],y[i]) for i in range(x.size)])

        domain = Domain(Point(), Point(1,1,1))

        circle = Circle(Point(), 5, n=(1,1,1))

        circleArc = CircleArc(Point(3,1,3), Point(-3,1,3), center=Point(0,1))       
        
        circleArc2 = CircleArc(Point(3,1,3), Point(-3,1,3), R=3)

        assert circleArc.center.Check(circleArc2.center)

        contour1 = Contour([Line(Point(), Point(5,0)),
                           CircleArc(Point(5), Point(-5), P=Point(0,5)),
                           Line(Point(-5), Point())])
        
        assert contour1.geoms[1].center.Check((0,0,0))
        
        points2 = Points([Point(), Point(5,0), Point(5,5,r=2), Point(0,5,r=-3)])
        contour2 = points2.Get_Contour()

        dec = (10,0,0)

        geoms: list[_Geom] = [line, points, domain,
                              circle, circleArc, contour1,
                              points2, contour2]

        for geom in geoms:

            ax = geom.Plot()

            geom.Translate(*dec)

            geom.Plot(ax)

            geom.Rotate(90)
            geom.Plot(ax)

            geom.Rotate(90, direction=(1,0,0))
            geom.Plot(ax)

            cop = geom.copy()
            cop.Translate(-10)
            cop.Plot(ax)

            cop.Symmetry()
            cop.Plot(ax)

            cop.Symmetry(cop.points[0],(0,0,1))
            cop.Plot(ax)

            cop.Symmetry(n=(0,np.cos(180/6),np.sin(180/6)))
            cop.Plot(ax)

            ax.legend()
        
        plt.close('all')