# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Thermal3
========

Transient thermal simulation.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, Folder, Models, np, Mesher, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Line, Domain, Point

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Results_Dir()
    makeMovie = True
    result = "thermal"

    # geom
    R = 10
    e = 2
    h = 10
    a = 1

    # load
    Tmax = 5
    N = 50
    dt = Tmax / N

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    domain = Domain(Point(R), Point(R + e, h), e / 2)
    axis = Line(Point(), Point(0, 1, 0))

    # Generate the mesh based on the specified dimension
    angle = 360 * 3 / 4
    mesh = Mesher().Mesh_Revolve(
        domain,
        [],
        axis,
        angle,
        [angle * np.pi / 180 * R / domain.meshSize],
        elemType=ElemType.HEXA8,
        isOrganised=True,
    )

    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodesYH = mesh.Nodes_Conditions(lambda x, y, z: y == h)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    thermalModel = Models.Thermal(k=1, c=1)
    simu = Simulations.ThermalSimu(mesh, thermalModel, False)
    simu.rho = 1

    simu.add_surfLoad(nodesY0, [5], ["t"])
    simu.add_surfLoad(nodesYH, [5], ["t"])

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

    simu._Set_solutions(simu.problemType, np.ones(mesh.Nn) * -10)

    print()
    t = -dt  # init time
    while t < Tmax:
        t += dt

        simu.Solve()
        simu.Save_Iter()

        print(f"{t:.3f} s", end="\r")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    PyVista.Plot(simu, result, plotMesh=True, nodeValues=True)

    if makeMovie:
        PyVista.Movie_simu(simu, result, folder, f"{result}.gif", plotMesh=True)
