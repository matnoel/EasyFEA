# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import plt, np
from EasyFEA.Geoms import _Geom, Point, Line, Circle, CircleArc, Points, Domain, Contour

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