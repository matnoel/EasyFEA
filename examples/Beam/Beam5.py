# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Beam5
=====

Frame with six beams
"""

from EasyFEA import Display, Models, np, Mesher, ElemType, Simulations
from EasyFEA.Geoms import Domain, Line

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    elemType = ElemType.SEG2
    dim = 2  # must be >= 2

    L = 100  # mm

    pA = (2 * L, 0)
    pB = (L, 0)
    pC = (L, L)
    pD = (0, 0)
    pE = (0, L)

    line1 = Line(pA, pC)
    line2 = Line(pA, pB)
    line3 = Line(pB, pC)
    line4 = Line(pC, pE)
    line5 = Line(pB, pD)
    line6 = Line(pB, pE)
    lines = [line1, line2, line3, line4, line5, line6]

    section = Mesher().Mesh_2D(Domain((-4 / 2, -8 / 2), (4 / 2, 8 / 2)))

    E = 276  # MPa
    v = 0.3

    beams = [Models.BeamElasIsot(dim, line, section, E, v) for line in lines]
    structure = Models.BeamStructure(beams)

    mesh = Mesher().Mesh_Beams(beams, elemType)
    # Display.Plot_Mesh(mesh)
    # Display.Plot_Tags(mesh)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    simu = Simulations.BeamSimu(mesh, structure)

    nodesRigi = mesh.Nodes_Point(pE)
    nodesRigi = np.append(nodesRigi, mesh.Nodes_Point(pD))
    nodesA = mesh.Nodes_Point(pA)

    # link beams at specified points
    for point in [pA, pB, pC]:
        nodes = mesh.Nodes_Point(point)
        firstNodes = nodes[0]
        others = nodes[1:]
        [simu.add_connection_hinged([firstNodes, n]) for n in others]

    simu.add_dirichlet(nodesRigi, [0, 0], ["x", "y"])
    simu.add_neumann(nodesA, [-40 * 9.81], ["y"])

    simu.Solve()

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    matrixDep = simu.Results_displacement_matrix()
    depMax = np.max(np.linalg.norm(matrixDep, axis=1))

    Display.Plot_Mesh(simu, deformFactor=10 / depMax)
    Display.Plot_Mesh(section, title="Cross section")
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "ux", deformFactor=5 / depMax)
    Display.Plot_Result(simu, "uy", deformFactor=5 / depMax)
    Display.Plot_Result(simu, "rz", deformFactor=5 / depMax)
    Display.Plot_Result(simu, "fx", deformFactor=5 / depMax)
    Display.Plot_Result(simu, "fy", deformFactor=5 / depMax)

    Epsilon_e_pg = simu._Calc_Epsilon_e_pg(simu.displacement)
    Internal_e = simu._Calc_InternalForces_e_pg(Epsilon_e_pg).mean(1)
    Sigma_e = simu._Calc_Sigma_e_pg(Epsilon_e_pg).mean(1)
    Display.Plot_Result(simu, Sigma_e[:, 0], title="Sxx")
    Display.Plot_Result(simu, Internal_e[:, 0], title="N")

    ux, uy, rz = simu.Result("ux"), simu.Result("uy"), simu.Result("rz")
    fx, fy, cz = simu.Result("fx"), simu.Result("fy"), simu.Result("cz")

    for i in range(5):
        print(f"\nNode {i} at {simu.mesh.coord[i]}")
        print(f"  ux={ux[i]:.2e} mm, uy={uy[i]:.2e} mm, rz={rz[i]:.2e} rad")
        print(f"  fx={fx[i]:.2e} N, fy={fy[i]:.2e} N, cz={cz[i]:.2e} N.mm")

    Display.plt.show()
