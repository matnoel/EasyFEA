# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh5_2D
========

Mesh of a 2D cracked part.
"""
# sphinx_gallery_thumbnail_number = 3

from EasyFEA import Display, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Point, Line, Points, Domain

if __name__ == "__main__":
    Display.Clear()

    L = 1
    openCrack = True

    contour = Domain(Point(), Point(L, L))

    # ----------------------------------------------
    # CRACK
    # ----------------------------------------------

    crack1 = Line(Point(L / 4, L / 2), Point(3 * L / 4, L / 2), isOpen=openCrack)
    crack2 = Line(
        Point(0, L / 3, isOpen=openCrack), Point(L / 2, L / 3), isOpen=openCrack
    )
    crack3 = Line(
        Point(0, 2 * L / 3, isOpen=openCrack), Point(L / 2, 2 * L / 3), isOpen=openCrack
    )
    crack4 = Line(Point(0, 4 * L / 5), Point(L, 4 * L / 5), isOpen=False)
    crack5 = Points(
        [Point(L / 2, L / 5), Point(2 * L / 3, L / 5), Point(L, L / 10, isOpen=True)],
        isOpen=True,
    )

    cracks = [crack1, crack2, crack3, crack4, crack5]

    PyVista.Plot_Geoms([contour, *cracks]).show()

    mesh = contour.Mesh_2D([], ElemType.TRI6, cracks)

    PyVista.Plot_Tags(mesh).show()

    # ----------------------------------------------
    # SIMU
    # ----------------------------------------------

    material = Models.Elastic.Isotropic(2)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: y == 0), [0] * 2, simu.Get_unknowns()
    )
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: y == L), [L * 0.05], ["y"])
    simu.Solve()
    PyVista.Plot(simu, "uy", 1, plotMesh=True).show()
