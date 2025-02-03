# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a hydraulic dam."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Points

if __name__ == '__main__':

    Display.Clear()

    h = 180
    N = 5

    pt1 = Point()
    pt2 = Point(x=h)
    pt3 = Point(y=h)    

    contour = Points([pt1, pt2, pt3], h/N)
    contour.Get_Contour().Plot()
    
    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"
    elemType = "PRISM6"
    mesh = Mesher().Mesh_Extrude(contour, [], [0,0,2*h], 10, elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()