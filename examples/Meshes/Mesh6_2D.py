# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh6_2D
========

Refined 2D mesh in zones.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType
from EasyFEA.Geoms import Circle, Domain

if __name__ == "__main__":
    Display.Clear()

    L = 1
    meshSize = L / 4

    contour = Domain((0, 0), (L, L), meshSize)
    circle = Circle((L / 2, L / 2), L / 3, meshSize)
    inclusions = [circle]

    refine1 = Domain((0, L), (L, L * 0.8), meshSize / 8)
    refine2 = Circle(circle.center, L / 2, meshSize / 8)
    refine3 = Circle((0, 0), L / 2, meshSize / 8)
    refineGeoms = [refine1, refine2, refine3]

    geoms = [contour, circle, refine1, refine2, refine3]
    contour.Plot_Geoms(geoms)

    mesh = contour.Mesh_2D(inclusions, ElemType.QUAD4, refineGeoms=refineGeoms)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
