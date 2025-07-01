# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
LShape
======

Damage simulation for a L-part.
"""

from EasyFEA import (
    Display,
    Folder,
    plt,
    np,
    Tic,
    ElemType,
    Materials,
    Simulations,
    Paraview,
    PyVista,
)
from EasyFEA.Geoms import Point, Points, Domain, Circle

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    doSimu = True
    meshTest = True
    optimMesh = True

    pltIter = False
    pltLoad = False
    makeMovie = False
    makeParaview = False

    # geom
    dim = 2
    L = 250  # mm
    nL = 50
    ep = 100

    # material
    E = 2e4  # MPa
    v = 0.18

    # phase field
    split = "Miehe"
    # split = "AnisotStress"
    regu = "AT1"
    Gc = 130  # J/m2
    Gc *= 1000 / 1e6  # mJ/mm2
    tolConv = 1e-0
    convOption = 2

    # loading
    # uMax = 1.2 # mm
    uMax = 1  # mm
    inc0 = uMax / 25
    inc1 = inc0 / 2

    config = f"""
    uMax = {uMax}

    inc0 = {inc0}
    inc1 = {inc1}

    while ud <= uMax:
    
    ud += inc0 if simu.damage.max() < 0.6 else inc1

    simu.add_dirichlet(nodes_circle, [0], ['d'], "damage")
    simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_dofs())
    simu.add_dirichlet(nodes_load, [ud], ['y'])
    """

    # folder
    name = "L_Shape_Benchmark"
    if dim == 3:
        name += "_3D"
    folder = Folder.Join(Folder.RESULTS_DIR, name, mkdir=True)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    l0 = L / nL

    if meshTest:
        hC = l0
    else:
        hC = 0.5
        # hC = 0.25

    p1 = Point()
    p2 = Point(L, 0)
    p3 = Point(L, L)
    p4 = Point(2 * L - 30, L)
    p5 = Point(2 * L, L)
    p6 = Point(2 * L, 2 * L)
    p7 = Point(0, 2 * L)
    if optimMesh:
        # hauteur zone rafinée
        h = 100
        refineDomain = Domain(Point(0, L - h / 3), Point(L + h / 3, L + h), hC)
        hD = hC * 5
    else:
        refineDomain = None
        hD = hC

    contour = Points([p1, p2, p3, p4, p5, p6, p7], hD)

    circle = Circle(p5, 100)

    if dim == 2:
        mesh = contour.Mesh_2D([], ElemType.TRI3, refineGeoms=[refineDomain])
    else:
        mesh = contour.Mesh_Extrude(
            [], [0, 0, -ep], [3], ElemType.HEXA8, refineGeoms=[refineDomain]
        )

    # Display.Plot_Mesh(mesh)
    # Display.Plot_Tags(mesh)
    # from EasyFEA import PyVista
    # PyVista.Plot_Mesh(mesh).show()

    nodes_y0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodes_load = mesh.Nodes_Conditions(lambda x, y, z: (y == L) & (x >= 2 * L - 30))
    node3 = mesh.Nodes_Point(p3)
    node4 = mesh.Nodes_Point(p4)
    nodes_circle = mesh.Nodes_Cylinder(circle, [0, 0, ep])
    nodes_edges = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) | (y == 0))

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.ElasIsot(dim, E, v, True, ep)
    pfm = Materials.PhaseField(material, split, regu, Gc, l0)

    folder_save = Folder.PhaseField_Folder(
        folder,
        "",
        pfm.split,
        pfm.regularization,
        "CP",
        tolConv,
        "",
        meshTest,
        optimMesh,
        nL=nL,
    )

    Display.MyPrint(folder_save, "green")

    if doSimu:
        simu = Simulations.PhaseFieldSimu(mesh, pfm)
        simu.Results_Set_Bc_Summary(config)

        dofsY_load = simu.Bc_dofs_nodes(nodes_load, ["y"])

        if pltIter:
            axIter = Display.Plot_Result(simu, "damage")

            axLoad = Display.Init_Axes()
            axLoad.set_xlabel("displacement [mm]")
            axLoad.set_ylabel("load [kN]")

        displacement = []
        force = []
        ud = -inc0
        iter = -1

        while ud <= uMax:
            # update displacement
            iter += 1
            ud += inc0 if simu.damage.max() < 0.6 else inc1

            # update boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_circle, [0], ["d"], "damage")
            simu.add_dirichlet(nodes_y0, [0] * dim, simu.Get_unknowns())
            simu.add_dirichlet(nodes_load, [ud], ["y"])

            # solve
            u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)
            # calc load
            fr = np.sum(Kglob[dofsY_load, :] @ u)

            # saves load and displacement
            displacement.append(ud)
            force.append(fr)

            # print iter
            simu.Results_Set_Iteration_Summary(iter, ud, "mm", ud / uMax, True)

            # saves iteration
            simu.Save_Iter()

            if pltIter:
                plt.figure(axIter.figure)
                Display.Plot_Result(simu, "damage", ax=axIter)
                plt.pause(1e-12)

                plt.figure(axLoad.figure)
                axLoad.scatter(ud, fr / 1000, c="black")
                plt.pause(1e-12)

            if not convergence or np.max(d[nodes_edges]) >= 1:
                # stops simulation if damage occurs on edges or convergence has not been reached
                break

        # saves load and displacement
        displacement = np.asarray(displacement)
        force = np.asarray(force)
        Simulations.Save_pickle(
            (force, displacement), folder_save, "force-displacement"
        )

        # saves the simulation
        simu.Save(folder_save)

    else:
        simu = Simulations.Load_Simu(folder_save)
        mesh = simu.mesh

    force, displacement = Simulations.Load_pickle(folder_save, "force-displacement")

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    Display.Plot_Result(simu, "damage", folder=folder_save)
    Display.Plot_Mesh(simu)
    Display.Plot_BoundaryConditions(simu)

    axLoad = Display.Init_Axes()
    axLoad.set_xlabel("displacement [mm]")
    axLoad.set_ylabel("load [kN]")
    axLoad.plot(displacement, force / 1000, c="blue")
    Display.Save_fig(folder_save, "forcedep")

    Display.Plot_Iter_Summary(simu, folder_save)

    if makeMovie:
        depMax = simu.Result("displacement_norm").max()
        deformFactor = L * 0.1 / depMax
        PyVista.Movie_simu(
            simu,
            "damage",
            folder_save,
            "damage.mp4",
            show_edges=True,
            deformFactor=deformFactor,
            clim=(0, 1),
        )

    if makeParaview:
        Paraview.Save_simu(simu, folder_save)

    if doSimu:
        Tic.Plot_History(folder_save, False)

    plt.show()
