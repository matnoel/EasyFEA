# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Dyna1
=====

A cantilever beam undergoing dynamic bending deformation.
"""

from EasyFEA import (
    Display,
    Folder,
    Tic,
    plt,
    ElemType,
    Materials,
    Simulations,
    Paraview,
)
from EasyFEA.Geoms import Domain

if __name__ == "__main__":

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2
    folder = Folder.Join(Folder.RESULTS_DIR, "Dynamics", "Dyna1", mkdir=True)
    plotResult = True

    initSimu = True
    depInit = -7
    load = -800  # N

    makeParaview = False
    makeMovie = False
    plotIter = True
    resultToPlot = "uy"

    # Dumping
    # coefM = 1e-2
    # coefK = 1e-2*2
    coefM = 1e-3
    coefK = 1e-3

    tic_Tot = Tic()

    # geom
    L = 120
    # mm
    h = 13
    b = 13

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = h / 5

    if dim == 2:
        # TRI3, TRI6, TRI10, TRI15, QUAD4, QUAD8, QUAD9
        elemType = ElemType.QUAD4
        domain = Domain((0, -h / 2), (L, h / 2), meshSize)
        mesh = domain.Mesh_2D(elemType=elemType, isOrganised=True)
        area = mesh.area - L * h
    elif dim == 3:
        # TETRA4, TETRA10, HEXA8, HEXA20, HEXA27, PRISM6, PRISM15, PRISM18
        elemType = ElemType.HEXA8
        domain = Domain((0, -h / 2, -b / 2), (L, h / 2, -b / 2), meshSize=meshSize)
        mesh = domain.Mesh_Extrude(
            [], [0, 0, b], elemType=elemType, layers=[3], isOrganised=True
        )

        volume = mesh.volume - L * b * h
        area = mesh.area - (L * h * 4 + 2 * b * h)

    nodes_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_L = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    nodes_h = mesh.Nodes_Conditions(lambda x, y, z: y == h / 2)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.ElasIsot(dim, thickness=b)
    simu = Simulations.ElasticSimu(mesh, material, useIterativeSolvers=initSimu)
    simu.rho = 8100 * 1e-9

    if initSimu:
        simu.Bc_Init()
        simu.add_dirichlet(nodes_0, [0] * dim, simu.Get_unknowns(), description="Fixed")
        simu.add_dirichlet(nodes_L, [depInit], ["y"], description="dep")
        simu.Solve()
        simu.Save_Iter()

        Display.Plot_Mesh(simu, deformFactor=1)

    factorDef = 1
    if plotIter:
        ax = Display.Plot_Result(
            simu, resultToPlot, nodeValues=True, plotMesh=True, deformFactor=factorDef
        )

    Tmax = 0.5
    N = 100
    dt = Tmax / N
    time = -dt

    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

    while time <= Tmax:

        time += dt

        simu.Bc_Init()
        simu.add_dirichlet(nodes_0, [0] * dim, simu.Get_unknowns(), description="Fixed")
        if not initSimu:
            simu.add_surfLoad(nodes_L, [load * time / Tmax / (h * b)], ["y"])
        simu.Solve()
        simu.Save_Iter()

        if plotIter:
            ax = Display.Plot_Result(
                simu,
                resultToPlot,
                nodeValues=True,
                plotMesh=True,
                ax=ax,
                deformFactor=factorDef,
            )
            plt.pause(1e-12)

        print(f"{time:.3f} s", end="\r")

    tic_Tot.Tac("Time", "total time", True)

    # ----------------------------------------------
    # Post processing
    # ----------------------------------------------
    Display.Section("Post processing")

    Display.Plot_BoundaryConditions(simu)

    # folder=""

    if makeParaview:
        Paraview.Save_simu(simu, folder)

    if makeMovie:
        Display.Movie_Simu(
            simu,
            resultToPlot,
            folder,
            f"{resultToPlot}.mp4",
            deformFactor=1,
            plotMesh=True,
            N=400,
            nodeValues=True,
        )

    if plotResult:

        tic = Tic()
        print(simu)
        Display.Plot_Result(simu, "uy", deformFactor=factorDef, nodeValues=False)
        Display.Plot_Result(simu, "Svm", plotMesh=False, nodeValues=False)

        tic.Tac("Display", "Results", plotResult)

    plt.show()
