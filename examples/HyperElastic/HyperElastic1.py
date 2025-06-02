# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import (
    Display,
    Folder,
    Mesher,
    ElemType,
    np,
    Materials,
    Simulations,
    PyVista,
)
from EasyFEA.Geoms import Domain

folder = Folder.Dir(__file__)

if __name__ == "__main__":

    Display.Clear()

    L = 120
    h = 13
    meshSize = h / 3

    contour = Domain((0, 0), (L, h), h / 3)

    mesh = Mesher().Mesh_Extrude(
        contour, [], [0, 0, h], [h / meshSize], ElemType.HEXA20, isOrganised=True
    )
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    lmbda = 121153.84615384616  # Mpa
    mu = 80769.23076923077
    rho = 7850 * 1e-9  # kg/mm3

    mat = Materials.SaintVenantKirchhoff(3, lmbda, mu)

    simuHyper = Simulations.HyperElasticSimu(mesh, mat)

    def Apply_Bc():
        simuHyper.Bc_Init()
        simuHyper.add_dirichlet(nodesX0, [0, 0, 0], simuHyper.Get_unknowns())
        # simuHyper.add_dirichlet(nodesXL, [-10], ["y"])
        simuHyper.add_volumeLoad(mesh.nodes, [-rho * 9.81], ["y"])
        simuHyper.add_surfLoad(nodesXL, [-800 / h / h], ["y"])

    simuHyper.Solve(Apply_Bc, maxIter=50)

    PyVista.Plot(simuHyper, "uy", 1, show_edges=True).show()
