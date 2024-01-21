"""Frame with two beams"""

import matplotlib.pyplot as plt
import numpy as np
from Interface_Gmsh import Interface_Gmsh, ElemType, Domain, Line, Point
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
    nL = 10
    h = 13
    b = 13
    E = 210000
    v = 0.3
    load = 800

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    elemType = ElemType.SEG2
    beamDim = 3

    # Create a section object for the beam mesh
    interfGmsh = Interface_Gmsh()
    section = interfGmsh.Mesh_2D(Domain(Point(x=-b / 2, y=-h / 2), Point(x=b / 2, y=h / 2)))

    point1 = Point()
    point2 = Point(y=L)
    point3 = Point(y=L, x=L / 2)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    beam1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, v)
    beam2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, v)
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
    simu.add_neumann(mesh.Nodes_Point(point3), [-load, load], ["y", "z"])
    if beamStructure.nBeam > 1:
        simu.add_connection_fixed(mesh.Nodes_Point(point2))

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------

    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, L/10/sol.max())
    Display.Plot_Result(simu, "ux", L/10/sol.max())
    Display.Plot_Result(simu, "uy", L/10/sol.max())    

    print(simu)

    plt.show()