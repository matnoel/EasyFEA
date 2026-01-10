# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas4
=====

Plate with a hole subjected to uniform tensile loading.
"""

from EasyFEA import Display, ElemType, Models, Simulations
from EasyFEA.Geoms import Point, Points, Domain, Circle

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 2
    isSymmetric = True

    # geom
    a = 10
    l = 50
    h = 20
    thickness = 1

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    meshSize = h / 4

    if isSymmetric:
        p0 = Point(0, 0, r=-a)
        p1 = Point(l, 0)
        p2 = Point(l, h)
        p3 = Point(0, h)
        contour = Points([p0, p1, p2, p3], meshSize)
        inclusions = []
    else:
        p0 = Point(-l, -h)
        p1 = Point(l, h)
        contour = Domain(p0, p1, meshSize)
        inclusions = [Circle(Point(), 2 * a, meshSize, isHollow=True)]

    if dim == 2:
        mesh = contour.Mesh_2D(inclusions, elemType=ElemType.TRI10)
    else:
        mesh = contour.Mesh_Extrude(
            inclusions, [0, 0, thickness], [1], ElemType.PRISM18
        )

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=thickness
    )
    simu = Simulations.Elastic(mesh, material)

    if isSymmetric:
        nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_xl = mesh.Nodes_Conditions(lambda x, y, z: x == l)
        simu.add_dirichlet(nodes_x0, [0], ["x"])
        simu.add_dirichlet(nodes_y0, [0], ["y"])
        simu.add_surfLoad(nodes_xl, [800 / 20], ["x"])
    else:
        nodes_pl = mesh.Nodes_Conditions(lambda x, y, z: x == l)
        nodes_ml = mesh.Nodes_Conditions(lambda x, y, z: x == -l)
        nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        simu.add_dirichlet(nodes_y0, [0], ["y"])
        simu.add_surfLoad(nodes_pl, [800 / 20], ["x"])
        simu.add_surfLoad(nodes_ml, [-800 / 20], ["x"])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    uxMax = simu.Result("ux").max()
    Display.Plot_Mesh(simu, deformFactor=2 / uxMax)
    Display.Plot_BoundaryConditions(simu)

    Display.Plot_Result(simu, "ux", ncolors=10)
    Display.Plot_Result(simu, "uy", ncolors=10)
    Display.Plot_Result(simu, "Svm", ncolors=10, nodeValues=False, plotMesh=True)

    print(simu)

    Display.plt.show()
