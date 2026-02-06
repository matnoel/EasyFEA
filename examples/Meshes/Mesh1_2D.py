# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh1_2D
========

Meshing a 2D domain.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, PyVista
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    contour = Domain((0, 0), (1, 1))
    PyVista.Plot_Geoms(contour).show()

    # "TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8",  "QUAD9"
    elemType = ElemType.TRI3
    mesh = contour.Mesh_2D([], elemType)
    PyVista.Plot_Mesh(mesh).show()
