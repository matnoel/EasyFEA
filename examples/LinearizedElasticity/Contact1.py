# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Contact1
========

Performing a 'Hertz contact problem' with the assumption of frictionless contact
The master mesh is considered non-deformable.

WARNING
-------
The assumption of small displacements is highly questionable for this simulation.
"""
# TODO: Compare results with analytical values ?

from EasyFEA import Display, Folder, Models, plt, np, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Point, Domain, Points

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # outputs
    folder = Folder.Results_Dir()
    pltIter = False
    makeMovie = True
    result = "uy"

    # geom
    R = 10
    height = R
    thickness = R / 3

    # load
    N = 30
    inc = 1e-0 / N
    cx, cy = 0, -1

    # ----------------------------------------------
    # Meshes
    # ----------------------------------------------

    meshSize = R / 20

    # slave mesh
    contour_slave = Domain(Point(-R / 2, 0), Point(R / 2, height), meshSize)
    if dim == 2:
        mesh_slave = contour_slave.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        mesh_slave = contour_slave.Mesh_Extrude(
            [], [0, 0, -thickness], [4], ElemType.HEXA8, isOrganised=True
        )

    # nodes_slave = mesh_slave.Get_list_groupElem(dim-1)[0].nodes
    nodes_slave = mesh_slave.Nodes_Conditions(lambda x, y, z: y == height)
    nodes_y0 = mesh_slave.Nodes_Conditions(lambda x, y, z: y == 0)

    # master mesh
    r = R / 2
    p0 = Point(-R / 2, height, r=r)
    p1 = Point(R / 2, height, r=r)
    p2 = Point(R / 2, height + R)
    p3 = Point(-R / 2, height + R)
    contour_master = Points([p0, p1, p2, p3])

    yMax = height + np.abs(r)
    if dim == 2:
        master_mesh = contour_master.Mesh_2D([], ElemType.TRI3)
    else:
        master_mesh = contour_master.Mesh_Extrude(
            [], [0, 0, -thickness - 2], [4], ElemType.TETRA4
        )
        groupMaster = master_mesh.Get_list_groupElem(dim - 1)[0]
        if len(master_mesh.Get_list_groupElem(dim - 1)) > 1:
            Display.MyPrintError(
                f"The {groupMaster.elemType.name} element group is used. In 3D, TETRA AND HEXA elements are recommended."
            )
    master_mesh.Translate(dz=-(master_mesh.center[2] - mesh_slave.center[2]))

    # Display.Plot_Tags(mesh_master, alpha=0.1, showId=True)

    # get master nodes
    # nodes_master = mesh_master.Get_list_groupElem(dim-1)[0].nodes
    if dim == 2:
        nodes_master = master_mesh.Nodes_Tags(["L0", "L1"])
    else:
        nodes_master = master_mesh.Nodes_Tags(["S1", "S2"])

    # # plot meshes
    # ax = Display.Plot_Mesh(master_mesh, alpha=0)
    # Display.Plot_Mesh(mesh_slave, ax=ax, alpha=0)
    # # add nodes interface
    # ax.scatter(*mesh_slave.coord[nodes_slave, :dim].T, label="slave nodes")
    # ax.scatter(*master_mesh.coord[nodes_master, :dim].T, label="master nodes")
    # ax.legend()
    # ax.set_title("Contact nodes")

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Models.Elastic.Isotropic(
        dim, E=210000, v=0.3, planeStress=True, thickness=thickness
    )
    simu = Simulations.ElasticSimu(mesh_slave, material)

    list_master_mesh = [master_mesh]

    if pltIter:
        ax = Display.Plot_Result(simu, result, deformFactor=1)

    for i in range(N):
        master_mesh = master_mesh.copy()
        master_mesh.Translate(cx * inc, cy * inc)

        list_master_mesh.append(master_mesh)

        convergence = False

        coordo_old = simu.Results_displacement_matrix() + simu.mesh.coord

        while not convergence:
            # apply new boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())

            nodes, newU = simu.Get_contact(master_mesh, nodes_slave, nodes_master)

            if nodes.size > 0:
                simu.add_dirichlet(nodes, [newU[:, 0], newU[:, 1]], ["x", "y"])

            simu.Solve()

            # check if there is no new nodes in the master mesh
            oldSize = nodes.size
            nodes, __ = simu.Get_contact(master_mesh, nodes_slave, nodes_master)
            convergence = oldSize == nodes.size

        simu.Save_Iter()

        print(f"Eps max = {simu.Result('Strain').max() * 100:3.2f} %")

        if pltIter:
            Display.Plot_Result(simu, result, plotMesh=True, deformFactor=1, ax=ax)
            Display.Plot_Mesh(master_mesh, alpha=0, ax=ax)
            ax.set_title(result)
            if dim == 3:
                Display._Axis_equal_3D(
                    ax, np.concatenate((master_mesh.coord, mesh_slave.coord), 0)
                )

            # # Plot arrows
            # if nodes.size >0:
            #     # get the nodes coordinates on the interface
            #     coordinates = groupMaster.Get_GaussCoordinates_e_p('mass').reshape(-1,3)
            #     ax.scatter(*coordinates[:,:dim].T)

            #     coordo_new = simu.Results_displacement_matrix() + simu.mesh.coord
            #     ax.scatter(*coordo_old[nodes,:dim].T)
            #     incU = coordo_new - coordo_oldq
            #     [ax.arrow(*coordo_old[node, :dim], *incU[node,:dim],length_includes_head=True) for node in nodes]

            plt.pause(1e-12)

    print(simu)

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    if makeMovie:

        def DoAnim(plotter, n):
            simu.Set_Iter(n)
            PyVista.Plot(
                simu,
                "Svm",
                1,
                style="surface",
                color="k",
                plotter=plotter,
                nColors=10,
                show_grid=True,
            )
            PyVista.Plot(list_master_mesh[n], plotter=plotter, plotMesh=True, alpha=0.2)

        PyVista.Movie_func(DoAnim, N, folder=folder, filename=f"{result}.gif")

    if not pltIter:
        plotter = PyVista.Plot(simu, result, plotMesh=True, deformFactor=1)
        PyVista.Plot_Mesh(master_mesh, alpha=0.4, plotter=plotter)
        plotter.show()

    Display.plt.show()
