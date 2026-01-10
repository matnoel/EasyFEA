# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Elas1
=====

A cantilever beam undergoing bending deformation.
"""

from EasyFEA import Display, Models, plt, np, ElemType, Simulations
from EasyFEA.Geoms import Domain

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # geom
    dim = 2
    L = 120  # mm
    h = 13
    I = h**4 / 12  # mm4

    # model
    E = 210000  # MPa (Young's modulus)
    v = 0.3  # Poisson's ratio
    coef = 1

    # load
    load = 800  # N

    # expected results
    W_an = 2 * load**2 * L / E / h**2 * (L**2 / h**2 + (1 + v) * 3 / 5)  # mJ
    uy_an = load * L**3 / (3 * E * I)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    N = 3
    meshSize = h / N

    domain = Domain((0, 0), (L, h), meshSize)

    if dim == 2:
        mesh = domain.Mesh_2D([], ElemType.QUAD9, isOrganised=True)
    else:
        mesh = domain.Mesh_Extrude(
            [], [0, 0, -h], [N], ElemType.HEXA27, isOrganised=True
        )

    nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodes_xL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Models.Elastic.Isotropic(dim, E, v, planeStress=True, thickness=h)
    simu = Simulations.Elastic(mesh, material)

    simu.add_dirichlet(nodes_x0, [0] * dim, simu.Get_unknowns())
    simu.add_surfLoad(nodes_xL, [-load / h**2], ["y"])

    sol = simu.Solve()
    simu.Save_Iter()

    uy_num = -simu.Result("uy").min()
    W_num = simu._Calc_Psi_Elas()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Section("Result")

    print(f"err W : {np.abs(W_an - W_num) / W_an * 100:.2f} %")

    print(f"err uy : {np.abs(uy_an - uy_num) / uy_an * 100:.2f} %")

    Display.Plot_Mesh(simu, h / 2 / np.abs(sol).max())
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy", nodeValues=True, coef=1 / coef, ncolors=20)
    Display.Plot_Result(simu, "Svm", plotMesh=True, ncolors=11)

    plt.show()
