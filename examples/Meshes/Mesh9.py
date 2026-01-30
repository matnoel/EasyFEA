# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh9
=====

Meshing of a perforated plate with a structured mesh.
"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np

from EasyFEA import Display, Mesher, ElemType, PyVista
from EasyFEA.Geoms import Point, Circle, Points, Line, CircleArc, Contour

if __name__ == "__main__":
    dim = 2

    if dim == 2:
        elemType = ElemType.QUAD4
    else:
        elemType = ElemType.HEXA8

    Display.Clear()

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------
    H = 90
    L = 45
    D = 10
    e = 20

    N = 5
    mS = (np.pi / 4 * D / 2) / N

    # PI for Points
    # pi for gmsh points
    PC = Point(L / 2, H / 2, 0)
    circle = Circle(PC, D, mS)

    P1 = Point()
    P2 = Point(L, 0)
    P3 = Point(L, H)
    P4 = Point(0, H)
    contour1 = Points(
        [
            (P3 + P2) / 2,
            P3,
            (P3 + P4) / 2,
            P4,
            (P4 + P1) / 2,
            P1,
            (P1 + P2) / 2,
            P2,
            (P3 + P2) / 2,
        ],
        mS,
    )

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    mesher = Mesher(False, True, True)
    factory = mesher._factory

    contours1: list[Contour] = []

    for c in range(4):
        pc = circle.center
        contour = circle.Get_Contour()
        pc1 = contour.geoms[c].pt1
        pc2 = contour.geoms[c].pt2
        pc3 = contour.geoms[c].pt3

        p1, p2, p3 = contour1.points[c * 2 : c * 2 + 3]

        cont1 = Contour(
            [Line(pc1, p1), Line(p1, p2), Line(p2, pc3), CircleArc(pc3, pc1, pc)]
        )
        loop1, lines1, points1 = mesher._Loop_From_Geom(cont1)

        cont2 = Contour(
            [Line(pc3, p2), Line(p2, p3), Line(p3, pc2), CircleArc(pc2, pc3, pc)]
        )
        loop2, lines2, points2 = mesher._Loop_From_Geom(cont2)

        surf1 = factory.addSurfaceFilling(loop1)
        surf2 = factory.addSurfaceFilling(loop2)

        mesher._Surfaces_Organize([surf1, surf2], elemType, True, [N] * 4)

        contours1.extend([cont1, cont2])

    PyVista.Plot_Geoms(contours1).show()

    if dim == 3:
        for cont1 in contours1:
            cont2 = cont1.copy()
            cont2.Translate(dz=e)
            # cont2.rotate(np.pi/8, PC.coord)
            mesher._Link_Contours(cont1, cont2, elemType, 3, [N] * 4)

    mesher._Set_PhysicalGroups()

    mesher._Mesh_Generate(dim, elemType)

    mesh = mesher._Mesh_Get_Mesh()

    if len(mesh.orphanNodes) > 0:
        plotter = PyVista.Plot_Nodes(mesh, mesh.orphanNodes)
        plotter.add_title("Orphan nodes detected")
        plotter.show()

    PyVista.Plot_Mesh(mesh).show()
