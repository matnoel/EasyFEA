# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Bending bracket component."""

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points

if __name__ == '__main__':

    Display.Clear()

    # Define dimension and mesh size parameters
    dim = 3
    N = 20 if dim == 2 else 10

    # Define material properties
    E = 210000  # MPa (Young's modulus)
    v = 0.3     # Poisson's ratio
    coef = 1

    L = 120 # mm
    h = L*0.3
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
    contour = Points([pt1, pt2, pt3, pt4, pt5, pt6], h/N)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3)
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, -h], [4], elemType=ElemType.TETRA4)

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_xL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.Elas_Isot(dim, E, v, planeStress=True, thickness=h)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodes_x0, [0]*dim, simu.Get_dofs())
    simu.add_surfLoad(nodes_xL, [-800/(h*h)], ["y"])

    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, h/2/np.abs(sol).max())
    Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1/coef, ncolors=20)

    Tic.Plot_History(details=False)

    plt.show()