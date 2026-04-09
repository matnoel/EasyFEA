# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Tension
=======

Damage simulation for a plate subjected to tension.
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
from EasyFEA.Geoms import Point, Points, Domain, Line, Contour

if __name__ == "__main__":

    # ----------------------------------------------
    # Configurations
    # ----------------------------------------------
    dim = 2

    # simu options
    doSimu = True
    meshTest = True
    openCrack = True
    optimMesh = True

    # outputs
    folder = Folder.Results_Dir() + f"{dim}D"
    plotMesh = False
    plotEnergy = False
    makeParaview = False
    makeMovie = True

    # phasefield
    maxIter = 1000
    tolConv = 1e-0  # 1e-1, 1e-2, 1e-3
    pfmSolver = Models.PhaseField.SolverType.History

    # Available splits: Bourdin, Amor, Miehe, Stress (isotropic)
    #                   He, AnisotStrain, AnisotStress, Zhang (anisotropic)
    split = Models.PhaseField.SplitType.Bourdin

    # Available regus: AT1, AT2
    regu = Models.PhaseField.ReguType.AT1

    # ----------------------------------------------
    # Geometry
    # ----------------------------------------------
    L = 1e-3  # m
    l0 = 8.5e-6
    thickness = 1 if dim == 2 else 0.1 / 1000

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    E = 210e9  # Pa
    v = 0.3
    Gc = 2.7e3  # J/m2

    folder_save = Simulations.PhaseField.Folder(
        folder,
        "",
        split,
        regu,
        "",
        tolConv,
        pfmSolver,
        meshTest,
        optimMesh,
        not openCrack,
    )
    Display.MyPrint(folder_save, "green", end="\n")

    if doSimu:
        # ----------------------------------------------
        # Mesh
        # ----------------------------------------------
        clC = l0 * 2 if meshTest else l0 / 2
        if optimMesh:
            clD = clC * 4
            gap = L * 0.05
            refineDomain = Domain(
                Point(L / 2 - gap, L / 2 - gap),
                Point(L, L / 2 + gap, thickness),
                clC,
            )
        else:
            clD = clC
            refineDomain = None

        pt1 = Point()
        pt2 = Point(L)
        pt3 = Point(L, L)
        pt4 = Point(0, L)
        contour = Points([pt1, pt2, pt3, pt4], clD)

        if dim == 2:
            ptC1 = Point(0, L / 2, isOpen=openCrack)
            ptC2 = Point(L / 2, L / 2)
            cracks = [Line(ptC1, ptC2, clC, isOpen=openCrack)]
        elif dim == 3:
            meshSize = clD if optimMesh else clC
            ptC1 = Point(0, L / 2, 0, isOpen=openCrack)
            ptC2 = Point(L / 2, L / 2, 0)
            ptC3 = Point(L / 2, L / 2, thickness)
            ptC4 = Point(0, L / 2, thickness, isOpen=openCrack)
            l1 = Line(ptC1, ptC2, meshSize, openCrack)
            l2 = Line(ptC2, ptC3, meshSize, False)
            l3 = Line(ptC3, ptC4, meshSize, openCrack)
            l4 = Line(ptC4, ptC1, meshSize, openCrack)
            cracks = [Contour([l1, l2, l3, l4])]

        if dim == 2:
            mesh = contour.Mesh_2D([], ElemType.TRI3, cracks, [refineDomain])
        elif dim == 3:
            mesh = contour.Mesh_Extrude(
                [], [0, 0, thickness], [3], ElemType.TETRA4, cracks, [refineDomain]
            )

        # Nodes
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == L)
        nodes_lower = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_right = mesh.Nodes_Conditions(
            lambda x, y, z: (x == L) & (y > 0) & (y < L)
        )
        nodes_crack = mesh.Nodes_Conditions(lambda x, y, z: (y == L / 2) & (x <= L / 2))
        if openCrack:
            nodes_detect = mesh.nodes.copy()
        else:
            nodes_detect = np.array(list(set(mesh.nodes) - set(nodes_crack)))

        # Builds edge nodes
        nodes_edges = []
        for nodes in [nodes_lower, nodes_right, nodes_upper]:
            nodes_edges.extend(nodes)

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        material = Models.Elastic.Isotropic(
            dim, E=E, v=v, planeStress=False, thickness=thickness
        )
        pfm = Models.PhaseField(material, split, regu, Gc=Gc, l0=l0, solver=pfmSolver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        uinc0 = 1e-7 if meshTest else 1e-8
        N0 = 40 if meshTest else 400
        uinc1 = 1e-8 if meshTest else 1e-9
        N1 = 400 if meshTest else 4000
        threshold = uinc0 * N0
        dep0 = threshold
        dep1 = dep0 + uinc1 * N1
        config = f"""
        E = {E:.2e} Pa;  v = {v};  Gc = {Gc:.2e} J/m2;  l0 = {l0:.2e} m

        while True:

        uinc0 = {uinc0:.1e} (dep < threshold={threshold:.2e})
        uinc1 = {uinc1:.1e}

        if not openCrack:
            simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
        simu.add_dirichlet(nodes_upper, [0, dep], ["x", "y"])
        simu.add_dirichlet(nodes_lower, [0], ["y"])
        """

        def Loading(dep):
            simu.Bc_Init()
            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
            if dim == 2:
                simu.add_dirichlet(nodes_upper, [0, dep], ["x", "y"])
            elif dim == 3:
                simu.add_dirichlet(nodes_upper, [0, dep, 0], ["x", "y", "z"])
            simu.add_dirichlet(nodes_lower, [0], ["y"])

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseField(mesh, pfm, folder=folder_save)
        simu.Results_Set_Bc_Summary(config)

        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        nDetect = 0
        list_dep = []
        list_f = []
        dep = -uinc0
        iter = -1
        while True:
            iter += 1
            dep += uinc0 if dep < threshold else uinc1

            Loading(dep)

            u, _, converg = simu.Solve(tolConv, maxIter, convOption=1)
            simu.Save_Iter()

            simu.Results_Set_Iteration_Summary(iter, dep * 1e6, "µm", 0, True)

            if not converg:
                break

            f = np.sum(simu.Calc_Reaction(dofsY_upper, "elastic"))
            list_dep.append(dep)
            list_f.append(f)

            if simu.Detect_Damage(nodes_edges, 1):
                nDetect += 1
                if nDetect == 10:
                    break

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
    Display.Plot_Result(
        simu,
        "damage",
        nodeValues=True,
        plotMesh=False,
        folder=folder_save,
        filename="damage",
    )
    Display.Plot_Mesh(simu)
    Display.Plot_Iter_Summary(simu, folder_save, None, None)
    Display.Plot_BoundaryConditions(simu)

    # ax = Display.Init_Axes()
    # ax.plot(np.abs(list_dep) * 1e6, np.abs(list_f) * 1e-6, c="blue")
    # ax.set_xlabel("ud [µm]")
    # ax.set_ylabel("f [kN/mm]")
    # ax.grid()
    # Display.Save_fig(folder_save, "force-displacement")

    if plotMesh:
        Display.Plot_Mesh(simu.mesh)

    if plotEnergy:
        Display.Plot_Energy(simu, N=400, folder=folder_save)

    if makeParaview:
        Paraview.Save_simu(simu, folder_save, 400)

    if makeMovie:
        simu.Set_Iter(-1)
        deformFactor = L * 0.05 / simu.Result("displacement_norm").max()

        iterations = np.arange(0, simu.Niter, simu.Niter // 20)

        def Func(plotter, iter):
            simu.Set_Iter(iterations[iter])
            thresh = PyVista._pvMesh(simu, "damage", deformFactor).threshold((0, 0.8))
            PyVista.Plot(thresh, "damage", plotMesh=True, plotter=plotter, clim=(0, 1))

        PyVista.Movie_func(Func, iterations.size, folder_save, "damage.gif")

    Tic.Resume()

    if doSimu:
        Tic.Plot_History(folder_save, False)

    plt.show()
