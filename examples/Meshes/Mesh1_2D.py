# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh1_2D
========

Meshing a 2D domain.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Domain

if __name__ == "__main__":

    Display.Clear()

    contour = Domain(Point(), Point(1, 1))
    contour.Plot()

    # "TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8",  "QUAD9"
    elemType = "TRI3"
    mesh = Mesher().Mesh_2D(contour, [], elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
