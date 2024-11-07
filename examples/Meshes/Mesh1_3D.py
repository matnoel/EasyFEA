# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Meshing a 3D domain."""

from EasyFEA import Display, Mesher
from EasyFEA.Geoms import Point, Domain

if __name__ == '__main__':

    Display.Clear()

    contour = Domain(Point(), Point(1,1))
    contour.Plot()

    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "PRISM6", "PRISM15"
    elemType = "TETRA4"
    mesh = Mesher().Mesh_Extrude(contour, [], [0,0,0.5], 3, elemType, isOrganised=True)
    Display.Plot_Mesh(mesh)

    Display.plt.show()