import unittest
import os

import numpy as np
from Geom import *

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
        p3.coordo = [0,0.4,0.5]
        self.assertTrue(p3.Check([0,0.4,0.5]))

        # copy
        pC1 = Point()
        pC2 = pC1.copy()        
        self.assertTrue(pC1 is not pC2)

        # translate
        pT1 = Point()
        pT1.translate(-1,2)
        self.assertTrue(pT1.Check((-1,2,0)))

        # rotate
        pR1 = Point(1)

        pR1.rotate(np.pi/2, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((0,1,0)))

        pR1.rotate(-np.pi/2, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((1,0,0)))

        pR1.rotate(np.pi, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((-1,0,0)))

        pR1.rotate(-np.pi, (0,0,0), (0,0,1))
        self.assertTrue(pR1.Check((1,0,0)))

        pR1.rotate(np.pi/2, (0,0,0), (0,-1,0))
        self.assertTrue(pR1.Check((0,0,1)))

        pR1.rotate(np.pi/2, (0,0,0), (1,0,0))
        self.assertTrue(pR1.Check((0,-1,0)))

        pR2 = Point(1)
        pR2.rotate(np.pi, (0,0,0), (1,0,1))
        self.assertTrue(pR2.Check((0,0,1)))

        pR3 = Point(1)
        pR3.rotate(np.pi/2, (-1,0,0), (0,-1,0))
        # self.assertTrue(pR3.Check((-1,0,2)))

    def test_Geoms(self):

        line = Line(Point(), Point(5,1))

        x = np.linspace(0, 5, 10)
        y = np.sin(x)        

        points = PointsList([Point(x[i],y[i]) for i in range(x.size)])

        domain = Domain(Point(), Point(1,1,1))

        circle = Circle(Point(), 5, n=(1,1,1))

        circleArc = CircleArc(Point(3,1,3), Point(-3,1,3), center=Point(0,1))       
        
        circleArc2 = CircleArc(Point(3,1,3), Point(-3,1,3), R=3)

        self.assertTrue(circleArc.center.Check(circleArc2.center))

        contour1 = Contour([Line(Point(), Point(5,0)),
                           CircleArc(Point(5), Point(-5), P=Point(0,5)),
                           Line(Point(-5), Point())])
        
        self.assertTrue(contour1.geoms[1].center.Check((0,0,0)))
        
        points2 = PointsList([Point(), Point(5,0), Point(5,5,r=2), Point(0,5,r=-3)])
        contour2 = points2.Get_Contour()

        dec = (10,0,0)

        geoms: list[Geom] = [line, points, domain, circle, circleArc, contour1, points2, contour2]

        for geom in geoms:

            ax = geom.Plot()

            geom.translate(*dec)

            geom.Plot(ax)

            geom.rotate(np.pi/2)
            geom.Plot(ax)

            geom.rotate(np.pi/2, direction=(1,0,0))
            geom.Plot(ax)

            cop = geom.copy()
            cop.translate(-10)
            cop.Plot(ax)


            cop.symmetry()
            cop.Plot(ax)


            cop.symmetry(cop.points[0],(0,0,1))
            cop.Plot(ax)

            cop.symmetry(n=(0,np.cos(np.pi/6),np.sin(np.pi/6)))
            cop.Plot(ax)



            ax.legend()
            pass

        Display.plt.close('all')


        
if __name__ == '__main__':        
    try:
        import Display
        Display.Clear()
        unittest.main(verbosity=2)
    except:
        print("")