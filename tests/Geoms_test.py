# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.txt for more information.

import unittest

from EasyFEA import plt, np
from EasyFEA.Geoms import *
from EasyFEA.Geoms import _Geom

class Test_Geom(unittest.TestCase):
    
    def test_Points(self):
        """test on points"""
        
        # opertionts
        p0 = Point(0,0,0)

        for val in [1,1.0,[1,1,1],(1,1,1),Point(1,1,1)]:
            # + and -            
            p1: Point = p0 + val
            self.assertTrue(p1.Check([1]*3))

            p2: Point = p1 - val
            self.assertTrue(p2.Check([0]*3))
        
        # mul, /, //        
        p3 = Point(1, 1, 1)

        p4 = p3 * 2
        self.assertTrue(p4.Check([2]*3))

        p5 = p4 / 2
        self.assertTrue(p5.Check([1]*3))

        p6 = p4 // 2
        self.assertTrue(p6.Check([1]*3))

        p7 = p3 * [1,2,3]
        self.assertTrue(p7.Check([1,2,3]))

        # new coord
        p3.coord = [0,0.4,0.5]
        self.assertTrue(p3.Check([0,0.4,0.5]))

        # copy
        pC1 = Point()
        pC2 = pC1.Copy()        
        self.assertTrue(pC1 is not pC2)

        # translate
        pT1 = Point()
        pT1.Translate(-1,2)
        self.assertTrue(pT1.Check((-1,2,0)))

        # rotate
        pR1 = Point(1)

        pR1.Rotate(90, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((0,1,0)))

        pR1.Rotate(-90, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((1,0,0)))

        pR1.Rotate(180, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((-1,0,0)))

        pR1.Rotate(-180, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((1,0,0)))

        pR1.Rotate(90, (0,0,0), (0,-1,0))
        self.assertTrue(pR1.Check((0,0,1)))

        pR1.Rotate(90, (0,0,0), (1,0,0))
        self.assertTrue(pR1.Check((0,-1,0)))

        pR2 = Point(1)
        pR2.Rotate(180, (0,0,0), (1,0,1))
        self.assertTrue(pR2.Check((0,0,1)))

        pR3 = Point(1)
        pR3.Rotate(90, (-1,0,0), (0,-1,0))
        # self.assertTrue(pR3.Check((-1,0,2)))

    def test_Geoms(self):

        line = Line(Point(), Point(5,1))

        x = np.linspace(0, 5, 10)
        y = np.sin(x)

        points = Points([Point(x[i],y[i]) for i in range(x.size)])

        domain = Domain(Point(), Point(1,1,1))

        circle = Circle(Point(), 5, n=(1,1,1))

        circleArc = CircleArc(Point(3,1,3), Point(-3,1,3), center=Point(0,1))       
        
        circleArc2 = CircleArc(Point(3,1,3), Point(-3,1,3), R=3)

        self.assertTrue(circleArc.center.Check(circleArc2.center))

        contour1 = Contour([Line(Point(), Point(5,0)),
                           CircleArc(Point(5), Point(-5), P=Point(0,5)),
                           Line(Point(-5), Point())])
        
        self.assertTrue(contour1.geoms[1].center.Check((0,0,0)))
        
        points2 = Points([Point(), Point(5,0), Point(5,5,r=2), Point(0,5,r=-3)])
        contour2 = points2.Get_Contour()

        dec = (10,0,0)

        geoms: list[_Geom] = [line, points, domain, circle, circleArc, contour1, points2, contour2]

        for geom in geoms:

            ax = geom.Plot()

            geom.Translate(*dec)

            geom.Plot(ax)

            geom.Rotate(90)
            geom.Plot(ax)

            geom.Rotate(90, direction=(1,0,0))
            geom.Plot(ax)

            cop = geom.Copy()
            cop.Translate(-10)
            cop.Plot(ax)

            cop.Symmetry()
            cop.Plot(ax)

            cop.Symmetry(cop.points[0],(0,0,1))
            cop.Plot(ax)

            cop.Symmetry(n=(0,np.cos(180/6),np.sin(180/6)))
            cop.Plot(ax)

            ax.legend()
            pass        
        
        plt.close('all')
        
if __name__ == '__main__':
    unittest.main(verbosity=2)