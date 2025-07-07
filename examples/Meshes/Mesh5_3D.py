# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Mesh5_3D
========

Mesh of a 3D cracked part.
"""

from EasyFEA import Display, ElemType, Models, Simulations
from EasyFEA.Geoms import Point, Line, Points, Domain, Contour

if __name__ == "__main__":
    Display.Clear()

    L = 1
    openCrack = True

    contour = Domain(Point(), Point(L, L))

    # ----------------------------------------------
    # CRACK
    # ----------------------------------------------

    line1 = Line(Point(L / 4, L / 2), Point(3 * L / 4, L / 2), isOpen=openCrack)
    line2 = Line(line1.pt2, line1.pt2 + [0, 0.25, L])
    line3 = Line(line2.pt2, line1.pt1 + [0, 0.25, L], isOpen=openCrack)
    line4 = Line(line3.pt2, line1.pt1)
    crack1 = Points(
        [
            Point(L / 2, L / 5, L),
            Point(2 * L / 3, L / 5, L),
            Point(L, L / 2, L, isOpen=True),
        ],
        isOpen=True,
    )

    cracks = [Contour([line1, line2, line3, line4], isOpen=openCrack), crack1]

    # WARNING:
    # only works with TETRA4 and TETRA10
    # only works with nLayers = []
    mesh = contour.Mesh_Extrude([], [0, 0, L], [], ElemType.TETRA4, cracks)
    Display.Plot_Tags(mesh, alpha=0.1, showId=True)

    # ----------------------------------------------
    # SIMU
    # ----------------------------------------------

    material = Models.ElasIsot(3)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(
        mesh.Nodes_Conditions(lambda x, y, z: y == 0), [0] * 3, simu.Get_unknowns()
    )
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x, y, z: y == L), [L * 0.05], ["y"])
    simu.Solve()
    Display.Plot_Result(simu, "uy", 1, plotMesh=True)

    Display.plt.show()
