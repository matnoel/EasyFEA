# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh3_3D
========

Meshing a 3D domain with hole.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, PyVista
from EasyFEA.Geoms import Domain, Circle

if __name__ == "__main__":
    Display.Clear()

    contour = Domain((0, 0), (1, 1), 1 / 10)
    circle = Circle((1 / 2, 1 / 2), 1 / 3, 1 / 10, isHollow=True)
    PyVista.Plot_Geoms([contour, circle]).show()

    # "TETRA4", "TETRA10", "HEXA8", "HEXA20", "HEXA27", "PRISM6", "PRISM15", "PRISM18"
    elemType = ElemType.PRISM15
    mesh = contour.Mesh_Extrude([circle], [0, 0, 0.5], [3], elemType, isOrganised=True)
    PyVista.Plot_Mesh(mesh).show()
