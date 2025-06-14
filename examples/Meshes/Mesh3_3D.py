# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh3_3D
========

Meshing a 3D domain with hole.
"""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Domain, Circle

if __name__ == "__main__":

    Display.Clear()

    contour = Domain(Point(), Point(1, 1), 1 / 10)
    circle = Circle(Point(1 / 2, 1 / 2), 1 / 3, 1 / 10, isHollow=True)
    contour.Plot()

    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"
    elemType = "PRISM15"
    mesh = Mesher().Mesh_Extrude(
        contour, [circle], [0, 0, 0.5], 3, elemType, isOrganised=True
    )
    Display.Plot_Mesh(mesh)

    Display.plt.show()
