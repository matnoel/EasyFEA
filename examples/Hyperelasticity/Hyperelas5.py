# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas5
==========

A perfectly incompressible beam clamped at one end and sheared by an imposed displacement of the other, the setup of `MFEM's Example 19 <https://github.com/mfem/mfem/blob/v4.10/examples/ex19.cpp>`_.
"""

# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Terminal, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 8
    h = 1

    # model
    mu = 1.0

    # load
    uyMax = -0.25 * L
    Nstep = 8

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = h / 2

    contour = Domain((0, 0), (L, h), meshSize)

    # u on the HEXA27 nodes, p on their HEXA8 vertices
    mesh = contour.Mesh_Extrude(
        [],
        [0, 0, h],
        [h / meshSize],
        ElemType.HEXA27,
        isOrganised=True,
    )

    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # no bulk term: the pressure enforces J = 1
    mat = Models.HyperElastic.NeoHookean(3, K=mu / 2)

    simu = Simulations.IncompressibleHyperElastic(mesh, mat)
    unknowns = ["x", "y", "z"]

    for step in range(1, Nstep + 1):
        simu.Bc_Init()
        simu.add_dirichlet(nodesLeft, [0, 0, 0], unknowns)
        simu.add_dirichlet(nodesRight, [0, uyMax * step / Nstep, 0], unknowns)

        simu.Solve()
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    print(simu)

    PyVista.Plot_BoundaryConditions(simu).show()
    PyVista.Plot(
        simu,
        simu.Result("J") - 1,
        1,
        plotMesh=True,
        colorbarTitle="J - 1",
    ).show()
    PyVista.Plot(simu, "p", 1, plotMesh=True).show()
    PyVista.Plot(simu, "uy", 1, plotMesh=True).show()
