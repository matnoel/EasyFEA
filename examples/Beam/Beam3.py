# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam3
=====

A bi-fixed beam undergoing bending deformation.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Models, Mesher, ElemType, Simulations
from EasyFEA.Geoms import Line, Point, Points

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 120
    h = 20
    b = 13
    e = 2

    # model
    E = 210000
    v = 0.3
    useTimoshenko = False

    # load
    F = -800

    # ----------------------------------------------
    # Section
    # ----------------------------------------------

    def DoSym(p: Point, n: np.ndarray) -> Point:
        pc = p.copy()
        pc.Symmetry(n=n)
        return pc

    p1 = Point(-b / 2, -h / 2)
    p2 = Point(b / 2, -h / 2)
    p3 = Point(b / 2, -h / 2 + e)
    p4 = Point(e / 2, -h / 2 + e, r=e)
    p5 = DoSym(p4, (0, 1))
    p6 = DoSym(p3, (0, 1))
    p7 = DoSym(p2, (0, 1))
    p8 = DoSym(p1, (0, 1))
    p9 = DoSym(p6, (1, 0))
    p10 = DoSym(p5, (1, 0))
    p11 = DoSym(p4, (1, 0))
    p12 = DoSym(p3, (1, 0))
    contour = Points([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], e / 6)
    section = Mesher().Mesh_2D(contour)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 2  # must be >= 2

    p1 = Point()
    pL = Point(x=L * 0.75)
    p2 = Point(x=L)
    line = Line(p1, p2, L / 9)
    beam = Models.Beam.Isotropic(beamDim, line, section, E, v)

    mesh = Mesher().Mesh_Beams([beam], additionalPoints=[pL], elemType=elemType)

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
    simu.add_dirichlet(mesh.Nodes_Point(p2), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(pL), [F], ["y"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Mesh(simu, L / 20 / sol.min())
    ax = Display.Plot_Mesh(section)
    ax.set_title("Section")
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy", L / 20 / sol.min())

    # ------------------------
    # uy
    # ------------------------

    # beam properties
    Iz = beam.Iz
    G = beam.mu
    A = section.area

    # general reactions and fixed-end moment (valid for arbitrary pL.x)
    a = pL.x
    b = L - a
    Ra = -F * b**2 * (3 * a + b) / L**3  # upward reaction at x=0
    Rb = -F * a**2 * (a + 3 * b) / L**3  # upward reaction at x=L
    Ma = F * a * b**2 / L**2  # fixed-end moment at x=0

    x = np.linspace(0, L, 100)
    x_n = mesh.coord[:, 0]
    x_e = x_n[mesh.connect].mean(1)  # element centroid x-coords

    def uy_x(x):
        x = np.asarray(x)
        macaulay = np.where(x > a, (x - a) ** 3 / 6, 0.0)
        return (Ma / 2 * x**2 + Ra / 6 * x**3 + F * macaulay) / (E * Iz)

    uy = simu.Result("uy")
    err_uy = np.abs(uy_x(x_n) - uy).max() / np.abs(uy_x(a))
    Display.MyPrint(f"\nerr uy: {err_uy * 100:.2e}%")

    ax_uy = Display.Init_Axes()
    ax_uy.plot(x, uy_x(x), label="Analytical", c="blue")
    ax_uy.scatter(x_n, uy, label="FE", c="red", marker="x", zorder=2)
    ax_uy.set_title("$u_y(x)$")
    ax_uy.legend()

    # ------------------------
    # Mz
    # ------------------------

    def Mz_x(x):
        x = np.asarray(x)
        Ma_plus_Ra_x = Ma + Ra * x
        return np.where(x <= a, Ma_plus_Ra_x, Ma_plus_Ra_x + F * (x - a))

    Mz = simu.Result("Mz", nodeValues=False)
    err_Mz = np.abs(Mz_x(x_e) - Mz).max() / np.abs(Mz_x(x_e)).max()
    Display.MyPrint(f"\nerr Mz : {err_Mz * 100:.2e} %")

    axMz = Display.Init_Axes()
    axMz.plot(x, Mz_x(x), label="Analytical", c="blue")
    axMz.scatter(x_e, Mz, label="FE", c="red", marker="x", zorder=2)
    axMz.set_title("$M_z(x)$")
    axMz.legend()

    # ------------------------
    # Ty
    # ------------------------

    def Ty_x(x):
        x = np.asarray(x)
        return np.where(x < a, -Ra, Rb)

    Ty = simu.Result("Ty", nodeValues=False)
    err_Ty = np.abs(Ty_x(x_e) - Ty).max() / max(Ra, Rb)
    Display.MyPrint(f"\nerr Ty : {err_Ty * 100:.2e} %")

    ax_Ty = Display.Init_Axes()
    ax_Ty.step(x, Ty_x(x), label="Analytical", c="blue", where="mid")
    ax_Ty.scatter(x_e, Ty, label="FE", c="red", marker="x", zorder=2)
    ax_Ty.set_title("$T_y(x)$")
    ax_Ty.legend()

    print(simu)

    plt.show()
