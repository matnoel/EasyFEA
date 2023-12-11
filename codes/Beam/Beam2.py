"""Bending beam"""

import matplotlib.pyplot as plt
import numpy as np
from Interface_Gmsh import Interface_Gmsh, ElemType, Domain, Line, Point, Section
import Display
import Materials
import Simulations
import Folder
import PostProcessing

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------------------------------------------

    L = 120
    nL = 1
    h = 13
    b = 13
    E = 210000
    uy = 0.3
    load = 800

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 3

    # Create a section object for the beam mesh
    interfGmsh = Interface_Gmsh()
    meshSection = interfGmsh.Mesh_2D(Domain(Point(x=-b / 2, y=-h / 2), Point(x=b / 2, y=h / 2)))
    section = Section(meshSection)

    point1 = Point()
    point2 = Point(x=L / 2)
    point3 = Point(x=L)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    line = Line(point1, point3)
    beam1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, uy)
    beam2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, uy)
    beams = [beam1, beam2]

    mesh = interfGmsh.Mesh_Beams(beams=beams, elemType=elemType)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------

    # Initialize the beam structure with the defined beam segments
    beamStructure = Materials.Beam_Structure(beams)

    # Create the beam simulation
    simu = Simulations.Simu_Beam(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(point1), [0]*dof_n, simu.Get_directions())
    simu.add_neumann(mesh.Nodes_Point(point3), [-load], ["y"])
    if beamStructure.nBeam > 1:
        simu.add_connection_fixed(mesh.Nodes_Point(point2))

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------

    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, deformFactor=-L/10/sol.min())
    Display.Plot_Result(simu, "ux")

    rz = simu.Result('rz')
    uy = simu.Result('uy')

    x = np.linspace(0, L, 100)
    uy_x = load / (E * section.Iz) * (x ** 3 / 6 - (L * x ** 2) / 2)

    flecheanalytique = load * L ** 3 / (3 * E * section.Iz)
    erreur = np.abs(flecheanalytique + uy.min()) / flecheanalytique

    # Plot the analytical and finite element solutions for vertical displacement (v)
    fig, axUy = plt.subplots()
    axUy.plot(x, uy_x, label='Analytique', c='blue')
    axUy.scatter(mesh.coordo[:, 0], uy, label='EF', c='red', marker='x', zorder=2)
    axUy.set_title(fr"$u_y(x)$")
    axUy.legend()

    rz_x = load / E / section.Iz * (x ** 2 / 2 - L * x)
    rotalytique = -load * L ** 2 / (2 * E * section.Iz)

    # Plot the analytical and finite element solutions for rotation (rz)
    fig, axRz = plt.subplots()
    axRz.plot(x, rz_x, label='Analytique', c='blue')
    axRz.scatter(mesh.coordo[:, 0], rz, label='EF', c='red', marker='x', zorder=2)
    axRz.set_title(fr"$r_z(x)$")
    axRz.legend()

    print(simu)

    plt.show()