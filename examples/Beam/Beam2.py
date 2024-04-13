"""A cantilever beam undergoing bending deformation."""

from EasyFEA import (Display, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Domain, Line, Point

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Dimensions
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
    beamDim = 2 # must be >= 2

    # Create a section object for the beam mesh
    mesher = Mesher()
    section = mesher.Mesh_2D(Domain(Point(), Point(b, h)))

    point1 = Point()
    point2 = Point(x=L / 2)
    point3 = Point(x=L)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    line = Line(point1, point3)
    beam1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, uy)
    beam2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, uy)
    beams = [beam1, beam2]

    mesh = mesher.Mesh_Beams(beams=beams, elemType=elemType)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    Iy = beam1.Iy
    Iz = beam1.Iz

    # Initialize the beam structure with the defined beam segments
    beamStructure = Materials.Beam_Structure(beams)

    # Create the beam simulation
    simu = Simulations.BeamSimu(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(point1), [0]*dof_n, simu.Get_dofs())
    simu.add_neumann(mesh.Nodes_Point(point3), [-load], ["y"])
    if beamStructure.nBeam > 1:
        simu.add_connection_fixed(mesh.Nodes_Point(point2))

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, deformFactor=-L/10/sol.min())
    Display.Plot_Result(simu, "uy")

    rz = simu.Result('rz')
    uy = simu.Result('uy')

    x = np.linspace(0, L, 100)
    uy_x = load / (E * Iz) * (x ** 3 / 6 - (L * x ** 2) / 2)

    flecheanalytique = load * L ** 3 / (3 * E * Iz)
    err_uy = np.abs(flecheanalytique + uy.min()) / flecheanalytique
    Display.myPrint(f"err uy: {err_uy*100:.2e} %")

    # Plot the analytical and finite element solutions for vertical displacement (v)
    axUy = Display.init_Axes()
    axUy.plot(x, uy_x, label='Analytique', c='blue')
    axUy.scatter(mesh.coord[:, 0], uy, label='EF', c='red', marker='x', zorder=2)
    axUy.set_title(fr"$u_y(x)$")
    axUy.legend()

    rz_x = load / E / Iz * (x ** 2 / 2 - L * x)
    rotalytique = load * L ** 2 / (2 * E * Iz)
    err_rz = np.abs(rotalytique + rz.min()) / rotalytique
    Display.myPrint(f"err rz: {err_rz*100:.2e} %")


    # Plot the analytical and finite element solutions for rotation (rz)
    axRz = Display.init_Axes()
    axRz.plot(x, rz_x, label='Analytique', c='blue')
    axRz.scatter(mesh.coord[:, 0], rz, label='EF', c='red', marker='x', zorder=2)
    axRz.set_title(fr"$r_z(x)$")
    axRz.legend()

    print(simu)

    plt.show()