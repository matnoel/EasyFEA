# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam7
=====

Frame with two beams in dynamic.
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
from EasyFEA.Geoms import Circle, Line

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
    h = 13
    b = 13

    # model
    E = 210000
    v = 0.3

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

    # Create a section object for the beam mesh
    mesher = Mesher()
    section = mesher.Mesh_2D(Circle((0, 0), h))

    p1 = (0, 0)
    p2 = (0, L)
    p3 = (L / 2, L)
    line1 = Line(p1, p2, L / 9)
    line2 = Line(p2, p3, L / 9)
    beam1 = Models.Beam.Isotropic(3, line1, section, E, v)
    beam2 = Models.Beam.Isotropic(3, line2, section, E, v)
    beams = [beam1, beam2]

    mesh = mesher.Mesh_Beams(beams=beams, elemType=elemType)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # Initialize the beam structure with the defined beam segments
    beamStructure = Models.Beam.BeamStructure(beams)

    # Create the beam simulation
    simu = Simulations.Beam(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(p3), [-load, load], ["y", "z"])
    if beamStructure.nBeam > 1:
        simu.add_connection_fixed(mesh.Nodes_Point(p2))

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.Bc_Init()
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    if beamStructure.nBeam > 1:
        simu.add_connection_fixed(mesh.Nodes_Point(p2))

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

    plt.show()
