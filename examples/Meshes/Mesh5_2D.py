# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh5_2D
========

Mesh of a 2D cracked part.
"""

from EasyFEA import Display, Mesher, ElemType, Materials, Simulations
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

    mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI6, cracks)

    Display.Plot_Tags(mesh, alpha=0.1, showId=True)

    # ----------------------------------------------
    # SIMU
    # ----------------------------------------------

    material = Materials.ElasIsot(2)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: y == 0), [0] * 2, simu.Get_unknowns()
    )
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: y == L), [L * 0.05], ["y"])
    simu.Solve()
    Display.Plot_Result(simu, "uy", 1, plotMesh=True)

    Display.plt.show()
