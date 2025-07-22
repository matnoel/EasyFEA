# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh2_3D
========

Meshing a hydraulic dam.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType, PyVista
from EasyFEA.Geoms import Points

if __name__ == "__main__":
    Display.Clear()

    h = 180
    N = 5

    contour = Points([(0, 0), (h, 0), (0, h)], h / N)
    PyVista.Plot_Geoms(contour).show()

    # "TETRA4", "TETRA10", "PRISM6", "PRISM15", "PRISM18"
    elemType = ElemType.PRISM6
    mesh = contour.Mesh_Extrude([], [0, 0, 2 * h], [10], elemType)
    PyVista.Plot_Mesh(mesh).show()
