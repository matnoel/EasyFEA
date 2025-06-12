# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a triangle."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Points

if __name__ == "__main__":

    Display.Clear()

    h = 180
    N = 5

    pt1 = Point()
    pt2 = Point(x=h)
    pt3 = Point(y=h)

    contour = Points([pt1, pt2, pt3], h / N)
    contour.Get_Contour().Plot()

    # "TRI3", "TRI6", "TRI10", "TRI15"
    elemType = "TRI3"
    mesh = Mesher().Mesh_2D(contour, [], elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
