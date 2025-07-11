# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh2_2D
========

Meshing a triangle.
"""
# sphinx_gallery_thumbnail_number = 2

from EasyFEA import Display, ElemType
from EasyFEA.Geoms import Points

if __name__ == "__main__":
    Display.Clear()

    h = 180
    N = 5

    contour = Points([(0, 0), (h, 0), (0, h)], h / N)
    contour.Get_Contour().Plot()

    # "TRI3", "TRI6", "TRI10", "TRI15"
    elemType = ElemType.TRI3
    mesh = contour.Mesh_2D([], elemType)
    Display.Plot_Mesh(mesh)

    Display.plt.show()
