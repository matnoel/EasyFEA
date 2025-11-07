# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Static0
=======

Attempt to implement hyperelasticity within an Eulerian framework.

Mesh node coordinates are updated at each loading iteration.
WARNING: Implementation not validated.
"""

from EasyFEA import Display, Folder, Models, ElemType, Simulations, Paraview
from EasyFEA.Geoms import Point, Points

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "HyperElastic", "HyperElastic0")
    makeParaview = False
    useHyperElastic = True  # eulerian approch

    # geom
    L = 250
    thickness = 50
    w = 50

    # load
    sigMax = 8 * 1e5 / (w * thickness)
    uMax = 50

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = L / 10

    p1 = Point(0, 0)
    p2 = Point(L, 0)
    p3 = Point(L, L, r=50)
    p4 = Point(2 * L - w, L)
    p5 = Point(2 * L, L)
    p6 = Point(2 * L, 2 * L)
    p7 = Point(2 * L - w, 2 * L)
    p8 = Point(0, 2 * L)

    contour = Points([p1, p2, p3, p4, p5, p6, p7, p8], meshSize)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI6)
    else:
        mesh = contour.Mesh_Extrude([], [0, 0, -thickness], [3], ElemType.PRISM6)

    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodes_Load = mesh.Nodes_Conditions(lambda x, y, z: x == 2 * L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.ElasIsot(
        dim, E=210000, v=0.25, planeStress=True, thickness=thickness
    )

    simu = Simulations.ElasticSimu(mesh, material)

    N = 20
    iter = 0

    while iter < N:
        iter += 1

        print(f"{iter / N * 100:2.2f} %", end="\r")

        simu.Bc_Init()
        simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
        # simu.add_dirichlet(nodes_Load, [uMax*iter/N], ['y'])
        simu.add_surfLoad(nodes_Load, [sigMax * iter / N], ["y"])

        simu.Solve()

        simu.Save_Iter()

        if useHyperElastic and iter != N:
            # update the nodes coordinates

            newMesh = simu.mesh.copy()
            newMesh.coordGlob += simu.Results_displacement_matrix()

            simu.mesh = newMesh

            pass

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_Mesh(simu, deformFactor=1)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "ux")
    Display.Plot_Result(simu, "uy")
    Display.Plot_Result(simu, "Svm", nodeValues=False)
    Display.Plot_Result(simu, "Evm", nodeValues=False)

    print(simu)

    if makeParaview:
        Paraview.Save_simu(simu, folder, elementFields=["Strain"])

    Display.plt.show()
