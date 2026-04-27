# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam1
=====

Beam subjected to pure tensile loading.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Models, Mesher, ElemType, Simulations
from EasyFEA.Geoms import Domain, Line

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 10
    nL = 10
    h = 0.1
    b = 0.1

    # model
    E = 200000e6
    ro = 7800
    v = 0.3

    # load
    g = 10
    q = ro * g * (h * b)
    F = 5000

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 1

    # Create a section for the beam
    mesher = Mesher()
    section = mesher.Mesh_2D(Domain((0, 0), (b, h)))

    p1 = (0, 0)
    p2 = (L, 0)
    line = Line(p1, p2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)

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
    simu.add_lineLoad(mesh.nodes, [q], ["x"])
    simu.add_neumann(mesh.Nodes_Point(p2), [F], ["x"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Mesh(simu, deformFactor=L / 10 / sol.max())
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "ux")

    # ------------------------
    # ux
    # ------------------------

    A = section.area
    x = np.linspace(0, L, 100)
    ux_x = lambda x: (F * x / (E * A)) + (ro * g * x / 2 / E * (2 * L - x))

    ux = simu.Result("ux")
    x_n = mesh.coord[:, 0]
    err_ux = np.abs(ux_x(x_n) - ux).max() / np.abs(ux_x(L))
    Display.MyPrint(f"\nerr ux: {err_ux * 100:.2e}%")

    axUx = Display.Init_Axes()
    axUx.plot(x, ux_x(x), label="Analytical", c="blue")
    axUx.scatter(x_n, ux, label="FE", c="red", marker="x", zorder=2)
    axUx.set_title("$u_x(x)$")
    axUx.legend()

    # ------------------------
    # N
    # ------------------------

    N_x = lambda x: F + q * (L - x)

    x_e = x_n[mesh.connect].mean(1)  # element centroid x-coords
    N = simu.Result("N", nodeValues=False)
    err_N = np.abs(N_x(x_e) - N).max() / np.abs(N_x(x_e)).max()
    Display.MyPrint(f"\nerr N: {err_N * 100:.2e}%")

    axN = Display.Init_Axes()
    axN.plot(x, N_x(x), label="Analytical", c="blue")
    axN.scatter(x_e, N, label="FE", c="red", marker="x", zorder=2)
    axN.set_title("$N(x)$")
    axN.legend()

    print(simu)

    plt.show()
