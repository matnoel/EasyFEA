# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas2
=====

Bending bracket component.
"""

from EasyFEA import Display, Models, plt, np, ElemType, Simulations
from EasyFEA.Geoms import Point, Points

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    dim = 3
    L = 120  # mm
    h = L * 0.3

    # model
    E = 210000  # MPa (Young's modulus)
    v = 0.3  # Poisson's ratio
    coef = 1

    # load
    load = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Define points and crack geometry for the mesh
    pt1 = Point(isOpen=True, r=-10)
    pt2 = Point(x=L)
    pt3 = Point(x=L, y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)

    contour = Points([pt1, pt2, pt3, pt4, pt5, pt6], h / 3)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI3)
    elif dim == 3:
        mesh = contour.Mesh_Extrude([], [0, 0, -h], [4], elemType=ElemType.TETRA10)

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_xL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Models.ElasIsot(dim, E, v, planeStress=True, thickness=h)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodes_x0, [0] * dim, simu.Get_unknowns())
    simu.add_surfLoad(nodes_xL, [-800 / (h * h)], ["y"])

    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Plot_Mesh(simu, h / 2 / np.abs(sol).max())
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1 / coef, ncolors=20)

    plt.show()
