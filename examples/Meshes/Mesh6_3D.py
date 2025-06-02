# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Refined 3D mesh in zones."""

from EasyFEA import Display, Mesher, ElemType
from EasyFEA.Geoms import Point, Circle, Domain

if __name__ == "__main__":

    Display.Clear()

    L = 1
    meshSize = L / 4

    contour = Domain(Point(), Point(L, L), meshSize)
    circle = Circle(Point(L / 2, L / 2), L / 3, meshSize)
    inclusions = [circle]

    refine1 = Domain(Point(0, L), Point(L, L * 0.8), meshSize / 8)
    refine2 = Circle(circle.center, L / 2, meshSize / 8)
    refine3 = Circle(Point(), L / 2, meshSize / 8)
    refineGeoms = [refine1, refine2, refine3]

    geoms = [contour, circle, refine1, refine2, refine3]
    contour.Plot_Geoms(geoms)

    mesh = Mesher().Mesh_Extrude(
        contour, inclusions, [0, 0, -L], [3], ElemType.PRISM15, refineGeoms=refineGeoms
    )
    Display.Plot_Mesh(mesh)

    Display.plt.show()
