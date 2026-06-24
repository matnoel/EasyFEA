# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam2
=====

A cantilever beam undergoing bending deformation.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Terminal, Matplotlib, Models, Mesher, ElemType, Simulations
from EasyFEA.Geoms import Domain, Line

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 120
    nL = 10
    h = 13
    b = 13

    # model
    E = 210000
    v = 0.3
    beamDim = 2  # must be >= 2
    useTimoshenko = True

    # load
    F = -800  # applied tip force in y (negative = downward)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Create a section object for the beam mesh
    mesher = Mesher()
    section = mesher.Mesh_2D(Domain((0, 0), (b, h)), elemType=ElemType.TRI6)

    p1 = (0, 0)
    p2 = (L, 0)
    line = Line(p1, p2, L / nL)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam])

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # Initialize the beam structure with the defined beam segments
    beamStructure = Models.Beam.BeamStructure([beam])

    # Create the beam simulation
    simu = Simulations.Beam(mesh, beamStructure, useTimoshenko=useTimoshenko)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(p2), [F], ["y"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Matplotlib.Plot_Mesh(simu, deformFactor=-L / 10 / sol.min())
    Matplotlib.Plot_BoundaryConditions(simu)
    Matplotlib.Plot(simu, "uy")

    # beam properties
    G = beam.mu
    A = section.area
    Iy = beam.Iy
    Iz = beam.Iz

    # ------------------------
    # uy
    # ------------------------

    x = np.linspace(0, L, 100)
    uy_x_eb = lambda x: F * (L * x**2 / 2 - x**3 / 6) / (E * Iz)
    if simu.useTimoshenko:
        kappa = beam._Get_shear_correction_factor()
        uy_x = lambda x: uy_x_eb(x) + F * x / (kappa * G * A)
    else:
        uy_x = uy_x_eb

    uy = simu.Result("uy")
    x_n = mesh.coord[:, 0]
    err_uy = np.abs(uy_x(x_n) - uy).max() / np.abs(uy_x(L))
    Terminal.MyPrint(f"\nerr uy: {err_uy * 100:.2e}%")

    axUy = Matplotlib.Init_Axes()
    axUy.plot(x, uy_x(x), label="Analytical", c="blue")
    axUy.scatter(x_n, uy, label="FE", c="red", marker="x", zorder=2)
    axUy.set_title("$u_y(x)$")
    axUy.legend()

    # ------------------------
    # rz
    # ------------------------

    rz_x = lambda x: F / E / Iz * (L * x - x**2 / 2)

    rz = simu.Result("rz")
    err_rz = np.abs(rz_x(x_n) - rz).max() / np.abs(rz_x(L))
    Terminal.MyPrint(f"\nerr rz: {err_rz * 100:.2e}%")

    axRz = Matplotlib.Init_Axes()
    axRz.plot(x, rz_x(x), label="Analytical", c="blue")
    axRz.scatter(x_n, rz, label="FE", c="red", marker="x", zorder=2)
    axRz.set_title("$r_z(x)$")
    axRz.legend()

    # ------------------------
    # Mz
    # ------------------------

    Mz_x = lambda x: F * (L - x)

    x_e = x_n[mesh.connect].mean(1)  # element centroid x-coords
    Mz = simu.Result("Mz", nodeValues=False)
    err_Mz = np.abs(Mz_x(x_e) - Mz).max() / np.abs(Mz_x(x_e)).max()
    Terminal.MyPrint(f"\nerr Mz: {err_Mz * 100:.2e}%")

    axMz = Matplotlib.Init_Axes()
    axMz.plot(x, Mz_x(x), label="Analytical", c="blue")
    axMz.scatter(x_e, Mz, label="FE", c="red", marker="x", zorder=2)
    axMz.set_title("$M_z(x)$")
    axMz.legend()

    # ------------------------
    # Ty
    # ------------------------

    Ty = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(F - Ty).max() / np.abs(F)
    Terminal.MyPrint(f"\nerr Ty: {err_Ty * 100:.2e}%")

    axTy = Matplotlib.Init_Axes()
    axTy.axhline(F, label="Analytical", c="blue")
    axTy.scatter(x_e, Ty, label="FE", c="red", marker="x", zorder=2)
    axTy.set_ylim(min(1.5 * F, -0.5 * F), max(1.5 * F, -0.5 * F))
    axTy.set_title("$T_y(x)$")
    axTy.legend()

    print(simu)

    plt.show()
