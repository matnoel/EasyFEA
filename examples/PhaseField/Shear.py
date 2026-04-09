# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


"""
Shear
=====

Damage simulation for a plate subjected to shear.
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
    Paraview,
    PyVista,
)
from EasyFEA.Geoms import Point, Points, Domain, Line, Contour

if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Configuration
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
    split = Models.PhaseField.SplitType.Miehe

    # Available regus: AT1, AT2
    regu = Models.PhaseField.ReguType.AT1

    # ----------------------------------------------
    # Geometry
    # ----------------------------------------------
    L = 1e-3  # m
    l0 = 1e-5  # m
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
            h = L if split == Models.PhaseField.SplitType.Bourdin else L / 2 + gap
            refineDomain = Domain(Point(L / 2 - gap, 0), Point(L, h, thickness), clC)
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
            cracks = [Contour([l1, l2, l3, l4], isOpen=openCrack)]

        if dim == 2:
            mesh = contour.Mesh_2D([], ElemType.TRI3, cracks, [refineDomain])
        elif dim == 3:
            mesh = contour.Mesh_Extrude(
                [],
                [0, 0, thickness],
                [4],
                ElemType.PRISM6,
                cracks,
                [refineDomain],
                additionalLines=[l1],
            )

        # Nodes
        nodes_crack = mesh.Nodes_Conditions(lambda x, y, z: (y == L / 2) & (x <= L / 2))
        nodes_upper = mesh.Nodes_Conditions(lambda x, y, z: y == L)
        nodes_lower = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
        nodes_left = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (y > 0) & (y < L))
        nodes_right = mesh.Nodes_Conditions(
            lambda x, y, z: (x == L) & (y > 0) & (y < L)
        )
        nodes_edges = np.concatenate([nodes_lower, nodes_right, nodes_upper])

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        material = Models.Elastic.Isotropic(
            dim, E=E, v=v, planeStress=False, thickness=thickness
        )
        pfm = Models.PhaseField(material, split, regu, Gc, l0, pfmSolver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        u_inc = 5e-8 if meshTest else 1e-8
        N = 400 if meshTest else 2000
        loadings = np.linspace(u_inc, u_inc * N, N, endpoint=True)

        config = f"""
        u_inc = {u_inc:.1e};  N = {N}
        loadings = np.linspace(u_inc, u_inc*N, N, endpoint=True)

        for iter, dep in enumerate(loadings):

            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
            simu.add_dirichlet(nodes_left, [0], ["y"])
            simu.add_dirichlet(nodes_right, [0], ["y"])
            simu.add_dirichlet(nodes_upper, [dep, 0], ["x", "y"])
            simu.add_dirichlet(nodes_lower, [0]*dim, simu.Get_unknowns())
        """

        def Loading(dep):
            simu.Bc_Init()
            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
            simu.add_dirichlet(nodes_left, [0], ["y"])
            simu.add_dirichlet(nodes_right, [0], ["y"])
            simu.add_dirichlet(nodes_upper, [dep, 0], ["x", "y"])
            simu.add_dirichlet(nodes_lower, [0] * dim, simu.Get_unknowns())

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseField(mesh, pfm, folder=folder_save)
        simu.Results_Set_Bc_Summary(config)

        dofsX_upper = simu.Bc_dofs_nodes(nodes_upper, ["x"])

        N = len(loadings)
        nDetect = 0
        list_dep = []
        list_f = []
        for iter, dep in enumerate(loadings):
            Loading(dep)

            u, _, converg = simu.Solve(tolConv, maxIter, convOption=2)
            simu.Save_Iter()

            simu.Results_Set_Iteration_Summary(iter, dep * 1e6, "µm", iter / N, True)

            if not converg:
                break

            f = np.sum(simu.Calc_Reaction(dofsX_upper, problemType="elastic"))
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
        ncolors=25,
    )
    Display.Plot_Mesh(simu)
    Display.Plot_Iter_Summary(simu, folder_save, None, None)
    Display.Plot_BoundaryConditions(simu)

    ax = Display.Init_Axes()
    ax.plot(np.abs(list_dep) * 1e6, np.abs(list_f) * 1e-6, c="blue")
    ax.set_xlabel("ud [µm]")
    ax.set_ylabel("f [kN/mm]")
    ax.grid()
    Display.Save_fig(folder_save, "force-displacement")

    if plotMesh:
        ax = Display.Plot_Mesh(simu.mesh, lw=0.3, facecolors="white")
        ax.axis("off")
        ax.set_title("")
        Display.Save_fig(folder_save, "mesh", transparent=True)

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

    if plotEnergy:
        Display.Plot_Energy(simu, N=400, folder=folder_save)

    Tic.Resume()

    if doSimu:
        Tic.Plot_History(folder_save, False)

    plt.show()
