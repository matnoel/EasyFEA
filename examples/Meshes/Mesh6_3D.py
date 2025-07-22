# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh6_3D
========

Refined 3D mesh in zones.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, PyVista
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

    PyVista.Plot_Geoms([contour, circle, *refineGeoms]).show()

    mesh = contour.Mesh_Extrude(
        inclusions, [0, 0, -L], [5], ElemType.PRISM15, refineGeoms=refineGeoms
    )
    PyVista.Plot_Mesh(mesh).show()
