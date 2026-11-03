# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas6
==========

Cook's membrane: a perfectly incompressible tapered panel, clamped on the left and sheared on the right, solved for the displacement and the pressure.
"""

# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Terminal, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Points

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 48
    H1 = 44
    H2 = 60

    # model
    mu = 1.0

    # load
    traction = 0.2 * mu
    Nstep = 5

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = (H2 - H1) / 5

    contour = Points(
        [
            (0, 0),
            (L, H1),
            (L, H2),
            (0, H1),
        ],
        meshSize=meshSize,
    )

    # u on the QUAD9 nodes, p on their QUAD4 vertices
    mesh = contour.Mesh_2D([], ElemType.QUAD9, isOrganised=True)

    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    nodesTip = mesh.Nodes_Point(contour.points[2])

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # no bulk term: the pressure enforces J = 1
    mat = Models.HyperElastic.NeoHookean(2, K=mu / 2)

    simu = Simulations.IncompressibleHyperElastic(mesh, mat)

    for step in range(1, Nstep + 1):
        simu.Bc_Init()
        simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"])
        simu.add_surfLoad(nodesRight, [traction * step / Nstep], ["y"])

        simu.Solve()
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    print(simu)

    PyVista.Plot_BoundaryConditions(simu).show()
    PyVista.Plot(simu, "p", 1, plotMesh=True, nodeValues=False).show()
    PyVista.Plot(simu, "J", 1, plotMesh=True, nodeValues=True).show()
    PyVista.Plot(simu, "uy", 1, plotMesh=True).show()
