# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas8
=====

A cantilever beam undergoing dynamic bending deformation.
"""

from EasyFEA import (
    Display,
    Folder,
    Models,
    plt,
    ElemType,
    Simulations,
    PyVista,
    Paraview,
)
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "LinearizedElasticity", "Dynamic1")
    makeParaview = False
    makeMovie = True
    result = "uy"

    # geom
    L = 120  # mm
    h = 13
    b = 13

    # time
    Tmax = 0.5
    N = 50
    dt = Tmax / N
    time = -dt

    # Dumping
    coefM = 1e-3
    coefK = 1e-3

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = h / 5

    if dim == 2:
        domain = Domain((0, -h / 2), (L, h / 2), meshSize)
        mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)

        area = mesh.area - L * h

    elif dim == 3:
        domain = Domain((0, -h / 2, -b / 2), (L, h / 2, -b / 2), meshSize=meshSize)
        mesh = domain.Mesh_Extrude([], [0, 0, b], [3], ElemType.HEXA8, isOrganised=True)

        volume = mesh.volume - L * b * h
        area = mesh.area - (L * h * 4 + 2 * b * h)

    nodes_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_L = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    nodes_h = mesh.Nodes_Conditions(lambda x, y, z: y == h / 2)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Models.ElasIsot(dim, thickness=b)
    simu = Simulations.ElasticSimu(mesh, material)
    simu.rho = 8100 * 1e-9

    # static simulation
    simu.Bc_Init()
    simu.add_dirichlet(nodes_0, [0] * dim, simu.Get_unknowns(), description="Fixed")
    simu.add_dirichlet(nodes_L, [-10], ["y"], description="dep")
    simu.Solve()
    simu.Save_Iter()
    Display.Plot_Mesh(simu, deformFactor=1)

    # dynamic simulation
    simu.Solver_Set_Hyperbolic_Algorithm(dt)
    simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

    simu.Bc_Init()
    simu.add_dirichlet(nodes_0, [0] * dim, simu.Get_unknowns(), description="Fixed")

    while time <= Tmax:
        time += dt

        simu.Solve()
        simu.Save_Iter()

        print(f"{time:.3f} s", end="\r")

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    Display.Plot_BoundaryConditions(simu)

    if makeParaview:
        Paraview.Save_simu(simu, folder)

    if makeMovie:
        PyVista.Movie_simu(
            simu,
            f"{result}",
            folder,
            f"{result}.gif",
            deformFactor=1,
            plotMesh=True,
        )

    print(simu)
    Display.Plot_Result(simu, f"{result}", deformFactor=1, nodeValues=False)
    Display.Plot_Result(simu, "Svm", plotMesh=False, nodeValues=False)

    plt.show()
