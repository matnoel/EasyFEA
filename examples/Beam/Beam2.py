# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam2
=====

A cantilever beam undergoing bending deformation.
"""

from EasyFEA import Display, plt, np, Mesher, ElemType, Materials, Simulations
from EasyFEA.Geoms import Domain, Line

if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    L = 120
    nL = 10
    h = 13
    b = 13
    E = 210000
    uy = 0.3
    load = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 2  # must be >= 2

    # Create a section object for the beam mesh
    mesher = Mesher()
    section = mesher.Mesh_2D(Domain((0, 0), (b, h)))

    p1 = (0, 0)
    p2 = (L, 0)
    line = Line(p1, p2, L / nL)
    beam = Materials.BeamElasIsot(beamDim, line, section, E, uy)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    Iy = beam.Iy
    Iz = beam.Iz

    # Initialize the beam structure with the defined beam segments
    beamStructure = Materials.BeamStructure([beam])

    # Create the beam simulation
    simu = Simulations.BeamSimu(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(p2), [-load], ["y"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Mesh(simu, deformFactor=-L / 10 / sol.min())
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy")

    rz = simu.Result("rz")
    uy = simu.Result("uy")

    x = np.linspace(0, L, 100)
    uy_x = load / (E * Iz) * (x**3 / 6 - (L * x**2) / 2)

    flecheanalytique = load * L**3 / (3 * E * Iz)
    err_uy = np.abs(flecheanalytique + uy.min()) / flecheanalytique
    Display.MyPrint(f"err uy: {err_uy*100:.2e} %")

    # Plot the analytical and finite element solutions for vertical displacement (v)
    axUy = Display.Init_Axes()
    axUy.plot(x, uy_x, label="Analytical", c="blue")
    axUy.scatter(mesh.coord[:, 0], uy, label="FE", c="red", marker="x", zorder=2)
    axUy.set_title("$u_y(x)$")
    axUy.legend()

    rz_x = load / E / Iz * (x**2 / 2 - L * x)
    rotalytique = load * L**2 / (2 * E * Iz)
    err_rz = np.abs(rotalytique + rz.min()) / rotalytique
    Display.MyPrint(f"err rz: {err_rz*100:.2e} %")

    # Plot the analytical and finite element solutions for rotation (rz)
    axRz = Display.Init_Axes()
    axRz.plot(x, rz_x, label="Analytical", c="blue")
    axRz.scatter(mesh.coord[:, 0], rz, label="FE", c="red", marker="x", zorder=2)
    axRz.set_title("$r_z(x)$")
    axRz.legend()

    print(simu)

    plt.show()
