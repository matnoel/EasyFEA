# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a 2D domain with hole."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Domain, Circle

if __name__ == '__main__':

    Display.Clear()

    contour = Domain(Point(), Point(1,1))
    circle = Circle(Point(1/2, 1/2), 1/3, 1/10, isHollow=True)
    contour.Plot()

    # "TRI3", "TRI6", "TRI10", "QUAD4", "QUAD8"
    elemType = "TRI3"
    mesh = Mesher().Mesh_2D(contour, [circle], elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()