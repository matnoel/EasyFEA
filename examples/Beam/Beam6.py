# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam6
=====

A cantilever beam undergoing bending deformation in dynamic.
"""

import matplotlib.pyplot as plt

from EasyFEA import (
    Folder,
    Display,
    Models,
    Mesher,
    ElemType,
    Simulations,
    Paraview,
    PyVista,
)
from EasyFEA.Geoms import Domain, Line

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Results_Dir()
    makeParaview = False
    makeMovie = True
    result = "uy"

    # geom
    L = 120
    nL = 10
    h = 13
    b = 13
    e = 3

    # model
    E = 210000
    uy = 0.3

    # load
    load = 800

    # time
    Tmax = 2.0
    N = 50
    dt = Tmax / N

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG3
    beamDim = 2  # must be >= 2

    # Create a section object for the beam mesh
    domain1 = Domain((0, 0), (b, h), h / 10)
    domain2 = Domain((e, e), (b - e, h - e), h / 10)
    section = domain1.Mesh_2D([domain2])

    p1 = (0, 0)
    p2 = (L, 0)
    line = Line(p1, p2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, uy)

    mesh = Mesher().Mesh_Beams([beam], elemType=elemType)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # Initialize the beam structure with the defined beam segments
    beamStructure = Models.Beam.BeamStructure([beam])

    # Create the beam simulation
    simu = Simulations.Beam(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(p2), [-load], ["y"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.Bc_Init()
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())

    for _ in range(N):
        simu.Solve()
        simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_BoundaryConditions(simu)

    if makeParaview:
        Paraview.Save_simu(simu, folder)

    deform = 10
    if makeMovie:
        PyVista.Movie_simu(
            simu,
            f"{result}",
            folder,
            f"{result}.gif",
            N=20,
            deformFactor=deform,
            plotMesh=True,
        )

    print(simu)

    Display.Plot_Result(simu, result, deformFactor=deform)
    ax = Display.Plot_Mesh(section)
    ax.set_title("Section")
    Display.Plot_Mesh(simu, deformFactor=deform)
    Display.Plot_BoundaryConditions(simu)

    Display.Plot_Mesh(simu, deform)

    plt.show()
