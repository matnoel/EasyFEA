# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Hyperelas4
==========

A cantilever beam undergoing bending deformation in dynamic.
"""
# sphinx_gallery_thumbnail_number = 1

from EasyFEA import (
    Display,
    Folder,
    ElemType,
    Models,
    Simulations,
    PyVista,
)
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "Hyperelasticity")
    makeMovie = True
    result = "uy"

    # geom
    L = 120
    h = 13

    # model
    lmbda = 121153.84615384616  # Mpa
    mu = 80769.23076923077

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = h / 3

    contour = Domain((0, 0), (L, h), h / 3)

    mesh = contour.Mesh_Extrude(
        [], [0, 0, h], [h / meshSize], ElemType.HEXA8, isOrganised=True
    )
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    mat = Models.SaintVenantKirchhoff(3, lmbda, mu)

    simu = Simulations.HyperElasticSimu(mesh, mat)

    simu.add_dirichlet(nodesX0, [0, 0, 0], simu.Get_unknowns())
    simu.add_dirichlet(nodesXL, [-h], ["y"])

    # static
    simu.Solve()
    simu.Save_Iter()

    # dynamic
    T = 7
    dt = T / 7
    simu.Bc_Init()
    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.add_dirichlet(nodesX0, [0, 0, 0], simu.Get_unknowns())

    for _ in range(int(T / dt)):
        simu.Solve()
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    PyVista.Plot_BoundaryConditions(simu).show()

    if makeMovie:
        PyVista.Movie_simu(
            simu, "uy", folder, "Hyperelas4.gif", deformFactor=1, plotMesh=True
        )

    PyVista.Plot(simu, "uy", 1, plotMesh=True).show()
