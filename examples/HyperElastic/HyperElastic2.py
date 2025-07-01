# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
HyperElastic2
=============

A hyper elastic cube in compression.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, ElemType, Materials, Simulations, PyVista
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    L = 1
    h = 1
    meshSize = h / 10

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (L, h), h / 10)

    mesh = contour.Mesh_Extrude(
        [], [0, 0, h], [h / meshSize], ElemType.HEXA8, isOrganised=True
    )

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    isot = Materials.ElasIsot(3, E=1, v=0.3)
    lmbda = isot.get_lambda()
    mu = isot.get_mu()
    mat = Materials.SaintVenantKirchhoff(3, lmbda, mu)

    simuHyper = Simulations.HyperElasticSimu(mesh, mat)

    uc = -0.3
    simuHyper.add_dirichlet(nodesX0, [0, 0, 0], simuHyper.Get_unknowns())
    simuHyper.add_dirichlet(nodesXL, [uc, 0, 0], simuHyper.Get_unknowns())

    simuHyper.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    PyVista.Plot_BoundaryConditions(simuHyper).show()
    PyVista.Plot(simuHyper, "ux", 1, show_edges=True).show()
