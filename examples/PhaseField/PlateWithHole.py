# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
PlateWithHole
=============

Damage simulation for a plate with a hole subjected to compression.
"""

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import (
    Display,
    Folder,
    Models,
    Tic,
    ElemType,
    Simulations,
    PyVista,
    Paraview,
)
from EasyFEA.Geoms import Domain, Circle

if __name__ == "__main__":

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # simu options
    doSimu = True
    meshTest = True
    optimMesh = True

    # outputs
    folder = Folder.Results_Dir()
    plotMesh = False
    plotIter = False
    plotEnergy = False

    makeParaview = False
    makeMovie = True

    # Available splits: Bourdin, Amor, Miehe, Stress (isotropic)
    #                   He, AnisotStrain, AnisotStress, Zhang (anisotropic)
    split = Models.PhaseField.SplitType.Miehe

    # Available regus: AT1, AT2
    regu = Models.PhaseField.ReguType.AT1

    l0 = 0.12e-3  # m

    # convergence
    solver = (
        Models.PhaseField.SolverType.History
    )  # History, HistoryDamage, BoundConstrain
    maxIter = 1000
    tolConv = 1e-0

    # load units
    unitU = "μm"
    unitF = "kN/mm"
    unit = 1e6

    # ----------------------------------------------
    # Geometry
    # ----------------------------------------------
    L = 15e-3  # m
    h = 30e-3  # m
    diam = 6e-3  # m
    thickness = 1

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    E = 12e9  # Pa
    v = 0.3
    Gc = 1.4  # J/m2

    folder_save = Simulations.PhaseField.Folder(
        folder, "", split, regu, "", tolConv, solver, meshTest, optimMesh
    )
    Display.MyPrint(folder_save, "green", end="\n")

    if doSimu:
        # ----------------------------------------------
        # Mesh
        # ----------------------------------------------
        clC = l0 * 2 if meshTest else l0 / 2
        if optimMesh:
            clD = l0 * 4
            refineZone = diam * 1.5 / 2
            if split in (
                Models.PhaseField.SplitType.Bourdin,
                Models.PhaseField.SplitType.Amor,
            ):
                refineGeom = Domain(
                    (0, h / 2 - refineZone), (L, h / 2 + refineZone), clC
                )
            else:
                refineGeom = Domain(
                    (L / 2 - refineZone, 0), (L / 2 + refineZone, h), clC
                )
        else:
            clD = l0 if meshTest else l0 / 2
            refineGeom = None

        domain = Domain((0, 0), (L, h), clD)
        circle = Circle((L / 2, h / 2), diam, clD)
        mesh = domain.Mesh_2D([circle], ElemType.TRI3, refineGeoms=[refineGeom])

        # Nodes
        nodes_lower = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == h)
        nodes_x0y0 = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y == 0))
        nodes_edges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        material = Models.Elastic.Isotropic(
            2, E, v, planeStress=False, thickness=thickness
        )
        pfm = Models.PhaseField(material, split, regu, Gc, l0, solver=solver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        threshold = 0.6
        u_max = 25e-6
        uinc0 = 8e-7 if meshTest else 8e-8
        uinc1 = 2e-7 if meshTest else 2e-8

        config = f"""
        E = {E:.2e} Pa;  v = {v};  Gc = {Gc} J/m2;  l0 = {l0:.2e} m

        while ud <= u_max = {u_max:.2e}:

        ud += uinc0 if simu.damage.max() < {threshold} else uinc1
        uinc0 = {uinc0:.1e};  uinc1 = {uinc1:.1e}

        simu.add_dirichlet(nodes_lower, [0], ["y"])
        simu.add_dirichlet(nodes_x0y0, [0], ["x"])
        simu.add_dirichlet(nodes_upper, [-ud], ["y"])
        """

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseField(mesh, pfm, verbosity=False)
        simu.Results_Set_Bc_Summary(config)

        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        def Apply_BC(ud: float):
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodes_x0y0, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])

        list_dep = []
        list_f = []
        ud = -uinc0
        iter = 0
        nDetect = 0

        if plotIter:
            axIter = Display.Plot_Result(simu, "damage", nodeValues=True)
            _, axLoad = plt.subplots()
            axLoad.set_xlabel(f"ud [{unitU}]")
            axLoad.set_ylabel(f"f [{unitF}]")
            axLoad.grid()

        while ud <= u_max:
            iter += 1
            ud += uinc0 if simu.damage.max() < threshold else uinc1

            Apply_BC(ud)

            u, d, convergence = simu.Solve(tolConv, maxIter)
            simu.Save_Iter()

            if not convergence:
                break

            f = np.sum(simu.Calc_Reaction(dofsY_upper, "elastic"))
            simu.Results_Set_Iteration_Summary(iter, ud * unit, unitU, ud / u_max, True)

            if simu.Detect_Damage(nodes_edges, 1):
                nDetect += 1
                if nDetect == 10:
                    break

            list_dep.append(ud)
            list_f.append(f)

            if plotIter:
                Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)
                plt.figure(axIter.figure)
                plt.pause(1e-12)
                axLoad.clear()
                axLoad.plot(np.abs(list_dep * unit), np.abs(list_f / unit), c="blue")
                axLoad.set_xlabel(f"ud [{unitU}]")
                axLoad.set_ylabel(f"f [{unitF}]")
                axLoad.grid()
                plt.figure(axLoad.figure)
                plt.pause(1e-12)

        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        print()
        Simulations.Save_pickle((list_f, list_dep), folder_save, "force-displacement")
        simu.Save(folder_save)

    else:
        simu: Simulations.PhaseField = Simulations.Load_Simu(folder_save)
        list_f, list_dep = Simulations.Load_pickle(folder_save, "force-displacement")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    if plotEnergy:
        Display.Plot_Energy(simu, list_f, list_dep, N=400, folder=folder_save)

    Display.Plot_Result(
        simu,
        "damage",
        nodeValues=True,
        colorbarIsClose=True,
        folder=folder_save,
        filename="damage",
    )
    Display.Plot_Mesh(simu)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Iter_Summary(simu, folder_save, None, None)

    ax = Display.Init_Axes()
    ax.plot(np.abs(list_dep) * unit, np.abs(list_f) / unit, c="blue")
    ax.set_xlabel(f"ud [{unitU}]")
    ax.set_ylabel(f"f [{unitF}]")
    ax.grid()
    Display.Save_fig(folder_save, "force-displacement")

    if plotMesh:
        Display.Plot_Mesh(simu.mesh)

    if makeParaview:
        Paraview.Save_simu(simu, folder_save)

    if makeMovie:
        simu.Set_Iter(-1)
        deformFactor = L * 0.05 / simu.Result("displacement_norm").max()

        iterations = np.arange(0, simu.Niter, simu.Niter // 20)

        def Func(plotter, iter):
            simu.Set_Iter(iterations[iter])
            thresh = PyVista._pvMesh(simu, "damage", deformFactor).threshold((0, 0.8))
            PyVista.Plot(thresh, "damage", plotMesh=True, plotter=plotter, clim=(0, 1))

        PyVista.Movie_func(Func, iterations.size, folder_save, "damage.gif")

    if doSimu:
        Tic.Plot_History(folder_save, details=False)

    plt.show()
