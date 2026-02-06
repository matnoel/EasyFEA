# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas6
=====

A bi-fixed beam undergoing bending deformation.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Models, Mesher, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Point, Points, Line

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    L = 500  # mm
    hx = 13
    hy = 20
    e = 2

    # model
    E = 210000  # MPa (Young's modulus)
    v = 0.3  # Poisson's ratio
    coef = 1

    # load
    load = 800  # N

    # ----------------------------------------------
    # Section
    # ----------------------------------------------

    elemType = ElemType.TETRA4

    def DoSym(p: Point, n: np.ndarray) -> Point:
        pc = p.copy()
        pc.Symmetry(n=n)
        return pc

    p1 = Point(-hx / 2, -hy / 2)
    p2 = Point(hx / 2, -hy / 2)
    p3 = Point(hx / 2, -hy / 2 + e)
    p4 = Point(e / 2, -hy / 2 + e, r=e)
    p5 = DoSym(p4, (0, 1))
    p6 = DoSym(p3, (0, 1))
    p7 = DoSym(p2, (0, 1))
    p8 = DoSym(p1, (0, 1))
    p9 = DoSym(p6, (1, 0))
    p10 = DoSym(p5, (1, 0))
    p11 = DoSym(p4, (1, 0))
    p12 = DoSym(p3, (1, 0))
    section = Points([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], e / 1)
    meshSection = Mesher().Mesh_2D(section)

    section.Get_Contour().Plot()

    section.Rotate(-90, direction=(0, 1))

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    mesher = Mesher()
    fact = mesher._factory
    surfaces = mesher._Surfaces(section, [], elemType)[0]

    layers = [50]
    # layers = [2*L/meshSize/10]

    mesher._Extrude(surfaces, [-L / 2, 0, 0], elemType, layers)
    mesher._Extrude(surfaces, [L / 2, 0, 0], elemType, layers)

    mesher._Set_PhysicalGroups()
    mesher._Mesh_Generate(3, elemType)
    mesh = mesher._Mesh_Get_Mesh()

    nodes_fixed = mesh.Nodes_Conditions(lambda x, y, z: (x == -L / 2) | (x == L / 2))
    nodes_load = mesh.Nodes_Line(Line(p7, p8))

    # ----------------------------------------------
    # Beam simulation
    # ----------------------------------------------

    beam = Models.Beam.Isotropic(
        2, Line(Point(-L / 2), Point(L / 2), L / 10), meshSection, E, v
    )

    mesh_beam = Mesher().Mesh_Beams([beam], ElemType.SEG3)

    beams = Models.Beam.BeamStructure([beam])
    simu_beam = Simulations.Beam(mesh_beam, beams)

    simu_beam.add_dirichlet(
        mesh_beam.Nodes_Conditions(lambda x, y, z: (x == -L / 2) | (x == L / 2)),
        [0] * simu_beam.Get_dof_n(),
        simu_beam.Get_unknowns(),
    )
    simu_beam.add_neumann(
        mesh_beam.Nodes_Conditions(lambda x, y, z: (x == 0)), [-load], ["y"]
    )
    simu_beam.Solve()

    Display.Plot_Result(simu_beam, "uy")

    u_an = load * L**3 / (192 * E * beam.Iz)

    uy_1d = np.abs(simu_beam.Result("uy").min())

    print(f"err beam model : {np.abs(u_an - uy_1d) / u_an * 100:.2f} %")

    # ----------------------------------------------
    # Elastic simulation
    # ----------------------------------------------

    material = Models.Elastic.Isotropic(3, E, v)
    simu = Simulations.Elastic(mesh, material)

    simu.add_dirichlet(nodes_fixed, [0] * 3, simu.Get_unknowns())
    simu.add_lineLoad(nodes_load, [-load / hx], ["y"])
    sol = simu.Solve()

    uy_3d = np.abs(simu.Result("uy").min())

    print(f"err 3d model : {np.abs(u_an - uy_3d) / u_an * 100:.2f} %")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    plotter = PyVista._Plotter(shape=(2, 1))
    PyVista.Plot(
        simu,
        "ux",
        coef=1 / coef,
        nColors=20,
        plotMesh=True,
        plotter=plotter,
        verticalColobar=False,
    )
    PyVista.Plot_BoundaryConditions(simu, plotter=plotter)
    plotter.subplot(1, 0)
    PyVista.Plot(
        simu, "uy", coef=1 / coef, nColors=20, plotter=plotter, verticalColobar=False
    )
    plotter.show()

    plt.show()
