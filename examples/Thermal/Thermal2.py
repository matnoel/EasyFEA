# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Thermal2
========

Transient thermal simulation.
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, Folder, Models, Mesher, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Circle, Domain, Point

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 3  # Set the simulation dimension (2D or 3D)

    # outputs
    folder = Folder.Results_Dir() + f"{dim}D"
    makeMovie = True
    result = "thermal"

    # geom
    a = 1

    # load
    Tmax = 0.5
    N = 50
    dt = Tmax / N

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    domain = Domain(Point(), Point(a, a), a / 4)
    circle = Circle(Point(a / 2, a / 2), diam=a / 3, isHollow=True, meshSize=a / 4)

    if dim == 2:
        mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI6)
    else:
        mesh = Mesher().Mesh_Extrude(
            domain, [circle], [0, 0, -a], [4], ElemType.PRISM18
        )

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXa = mesh.Nodes_Conditions(lambda x, y, z: x == a)
    nodesCircle = mesh.Nodes_Cylinder(circle, direction=[0, 0, -1])

    if dim == 3:
        print(f"Volume: {mesh.volume:.3}")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    thermalModel = Models.Thermal(k=1, c=1, thickness=1)
    simu = Simulations.Thermal(mesh, thermalModel, False)
    simu.rho = 1

    simu.add_dirichlet(nodesX0, [0], ["t"])
    simu.add_dirichlet(nodesXa, [40], ["t"])

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

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

    PyVista.Plot(simu, result, plotMesh=True, nodeValues=True).show()

    if makeMovie:
        PyVista.Movie_simu(simu, result, folder, f"{result}.gif", plotMesh=True)
