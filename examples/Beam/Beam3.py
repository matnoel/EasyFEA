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

    # load
    load = 800

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
    pL = Point(x=L / 2)
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
    simu = Simulations.Beam(mesh, beamStructure)
    dof_n = simu.Get_dof_n()

    # Apply boundary conditions
    simu.add_dirichlet(mesh.Nodes_Point(p1), [0] * dof_n, simu.Get_unknowns())
    simu.add_dirichlet(mesh.Nodes_Point(p2), [0] * dof_n, simu.Get_unknowns())
    simu.add_neumann(mesh.Nodes_Point(pL), [-load], ["y"])

    # Solve the beam problem and get displacement results
    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    u_an = load * L**3 / (192 * E * beam.Iz)

    uy_1d = np.abs(simu.Result("uy").min())

    Display.MyPrint(f"err uy : {np.abs(u_an - uy_1d) / u_an * 100:.2f} %")

    Display.Plot_Mesh(simu, L / 20 / sol.min())
    ax = Display.Plot_Mesh(section)
    ax.set_title("Section")
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy", L / 20 / sol.min())

    print(simu)

    plt.show()
