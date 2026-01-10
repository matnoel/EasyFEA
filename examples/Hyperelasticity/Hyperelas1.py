# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas1
==========

A cantilever beam undergoing bending deformation.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 120
    h = 13

    # model
    lmbda = 121153.84615384616  # Mpa
    mu = 80769.23076923077
    rho = 7850 * 1e-9  # kg/mm3

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = h / 2

    contour = Domain((0, 0), (L, h), h / 3)

    mesh = contour.Mesh_Extrude(
        [], [0, 0, h], [h / meshSize], ElemType.HEXA20, isOrganised=True
    )
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    mat = Models.HyperElastic.SaintVenantKirchhoff(3, lmbda, mu)

    simu = Simulations.HyperElasticSimu(mesh, mat)

    simu.add_dirichlet(nodesX0, [0, 0, 0], simu.Get_unknowns())
    simu.add_volumeLoad(mesh.nodes, [-rho * 9.81], ["y"])
    simu.add_surfLoad(nodesXL, [-800 / h / h], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    PyVista.Plot_BoundaryConditions(simu).show()
    PyVista.Plot(simu, "uy", 1, plotMesh=True).show()
