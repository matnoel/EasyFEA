# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Beam subjected to pure tensile loading."""

from EasyFEA import (Display, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Domain, Line, Point

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    L = 10
    nL = 10
    h = 0.1
    b = 0.1
    E = 200000e6
    ro = 7800
    v = 0.3
    g = 10
    q = ro * g * (h * b)
    load = 5000

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 1

    # Create a section for the beam
    mesher = Mesher()
    section = mesher.Mesh_2D(Domain(Point(), Point(b, h)))

    p1 = Point()    
    p2 = Point(x=L)
    line = Line(p1, p2, L/nL)
    beam = Materials.Beam_Elas_Isot(beamDim, line, section, E, v)

    mesh = mesher.Mesh_Beams([beam], elemType=elemType)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # Initialize the beam structure with the defined beam segments
    beamStructure = Materials.BeamStructure([beam])

    # Create the beam simulation
    simu = Simulations.BeamSimu(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0]*dof_n, simu.Get_dofs())
    simu.add_lineLoad(mesh.nodes, [q], ["x"])
    simu.add_neumann(mesh.Nodes_Point(p2), [load], ["x"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, deformFactor=L/10/sol.max())
    Display.Plot_Result(simu, "ux")

    ux = simu.Result('ux')

    x_array = np.linspace(0, L, 100)
    u_x = (load * x_array / (E * (section.area))) + (ro * g * x_array / 2 / E * (2 * L - x_array))
    err_ux = np.abs(u_x[-1] - ux.max()) / ux.max()
    Display.MyPrint(f"err ux: {err_ux*100:.2e} %")

    # Plot the analytical and finite element solutions for displacement (u)
    ax = Display.Init_Axes()
    ax.plot(x_array, u_x, label='Analytical', c='blue')
    ax.scatter(mesh.coord[:, 0], ux, label='FE', c='red', marker='x', zorder=2)
    ax.set_title(fr"$u_x(x)$")
    ax.legend()

    print(simu)

    plt.show()