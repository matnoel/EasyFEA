# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a 3D bracket."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Points

if __name__ == "__main__":

    Display.Clear()

    L = 120
    h = L * 0.3
    N = 8

    pt1 = Point(isOpen=True, r=-10)
    pt2 = Point(x=L)
    pt3 = Point(x=L, y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)

    contour = Points([pt1, pt2, pt3, pt4, pt5, pt6], h / N)
    contour.Get_Contour().Plot()

    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"
    elemType = "PRISM15"
    mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, -h], 3, elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
