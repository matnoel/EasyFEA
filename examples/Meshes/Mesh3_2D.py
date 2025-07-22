# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh3_2D
========

Meshing a 2D domain with hole.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, PyVista
from EasyFEA.Geoms import Domain, Circle

if __name__ == "__main__":
    Display.Clear()

    contour = Domain((0, 0), (1, 1), 1 / 10)
    circle = Circle((1 / 2, 1 / 2), 1 / 3, 1 / 10, isHollow=True)
    PyVista.Plot_Geoms([contour, circle]).show()

    # "TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8",  "QUAD9"
    elemType = ElemType.TRI6
    mesh = contour.Mesh_2D([circle], elemType)
    PyVista.Plot_Mesh(mesh).show()
