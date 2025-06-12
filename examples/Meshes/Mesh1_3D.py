# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a 3D domain."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Domain

if __name__ == "__main__":

    Display.Clear()

    contour = Domain(Point(), Point(1, 1))
    contour.Plot()

    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"
    elemType = "HEXA8"
    mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, 0.5], 3, elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
