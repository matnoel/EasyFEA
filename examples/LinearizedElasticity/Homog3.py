# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog3
======

Conduct full-field homogenization.
"""
# sphinx_gallery_thumbnail_number = -4

from EasyFEA import Display, Models, plt, np, Geoms, ElemType, Simulations
from EasyFEA.fem import FeArray

from Homog1 import Compute_ukl

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    # use Periodic boundary conditions ?
    usePER = True

    # geom
    L = 120  # mm
    h = 13
    b = 13

    # inclusions
    nL = 40  # number of inclusions following x
    nH = 4  # number of inclusions following y
    cL = L / (2 * nL)
    cH = h / (2 * nH)
    isHollow = True  # hollow inclusions

    # model
    E = 210000
    v = 0.3

    # load
    load = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    elemType = ElemType.TRI6
    meshSize = h / 10

    pt1 = Geoms.Point()
    pt2 = Geoms.Point(L, 0)
    pt3 = Geoms.Point(L, h)
    pt4 = Geoms.Point(0, h)

    domain = Geoms.Domain(pt1, pt2, meshSize)

    inclusions = []
    for i in range(nL):
        x = cL + cL * (2 * i)
        for j in range(nH):
            y = cH + cH * (2 * j)

            ptd1 = Geoms.Point(x - cL / 2, y - cH / 2)
            ptd2 = Geoms.Point(x + cL / 2, y + cH / 2)

            inclusion = Geoms.Domain(ptd1, ptd2, meshSize, isHollow)

            inclusions.append(inclusion)

    inclusion = Geoms.Domain(ptd1, ptd2, meshSize)
    area_inclusion = inclusion.Mesh_2D().area

    points = Geoms.Points([pt1, pt2, pt3, pt4], meshSize)

    # mesh with inclusions
    mesh_inclusions = points.Mesh_2D(inclusions, elemType)

    # mesh without inclusions
    mesh = points.Mesh_2D([], elemType)

    ptI1 = Geoms.Point(-cL, -cH)
    ptI2 = Geoms.Point(cL, -cH)
    ptI3 = Geoms.Point(cL, cH)
    ptI4 = Geoms.Point(-cL, cH)

    pointsI = Geoms.Points([ptI1, ptI2, ptI3, ptI4], meshSize / 4)

    mesh_RVE = pointsI.Mesh_2D(
        [
            Geoms.Domain(
                Geoms.Point(-cL / 2, -cH / 2),
                Geoms.Point(cL / 2, cH / 2),
                meshSize / 4,
                isHollow,
            )
        ],
        elemType,
    )

    Display.Plot_Mesh(mesh_inclusions, title="non hom")
    Display.Plot_Mesh(mesh_RVE, title="RVE")
    Display.Plot_Mesh(mesh, title="hom")

    # ----------------------------------------------
    # Material
    # ----------------------------------------------

    # elastic behavior of the beam
    material_inclsuion = Models.ElasIsot(2, E=E, v=v, planeStress=True, thickness=b)
    CMandel = material_inclsuion.C

    material = Models.ElasAnisot(2, CMandel, False)
    testC = np.linalg.norm(material_inclsuion.C - material.C) / np.linalg.norm(
        material_inclsuion.C
    )
    assert testC < 1e-12, "the matrices are different"

    # ----------------------------------------------
    # Homogenization
    # ----------------------------------------------
    simu_inclusions = Simulations.ElasticSimu(mesh_inclusions, material_inclsuion)
    simu_VER = Simulations.ElasticSimu(mesh_RVE, material_inclsuion)
    simu = Simulations.ElasticSimu(mesh, material)

    r2 = np.sqrt(2)
    E11 = np.array([[1, 0], [0, 0]])
    E22 = np.array([[0, 0], [0, 1]])
    E12 = np.array([[0, 1 / r2], [1 / r2, 0]])

    if usePER:
        nodes_border = mesh_RVE.Nodes_Tags(["P0", "P1", "P2", "P3"])
        paired_nodes = mesh_RVE.Get_Paired_Nodes(nodes_border, True)
    else:
        nodes_border = mesh_RVE.Nodes_Tags(["L0", "L1", "L2", "L3"])
        paired_nodes = None

    u11 = Compute_ukl(simu_VER, nodes_border, E11, paired_nodes)
    u22 = Compute_ukl(simu_VER, nodes_border, E22, paired_nodes)
    u12 = Compute_ukl(simu_VER, nodes_border, E12, paired_nodes, True)

    u11_e = mesh_RVE.Locates_sol_e(u11, asFeArray=True)
    u22_e = mesh_RVE.Locates_sol_e(u22, asFeArray=True)
    u12_e = mesh_RVE.Locates_sol_e(u12, asFeArray=True)

    U_e = FeArray.zeros(*u11_e.shape, 3)

    U_e[..., 0] = u11_e
    U_e[..., 1] = u22_e
    U_e[..., 2] = u12_e

    matrixType = "rigi"
    wJ_e_pg = mesh_RVE.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = mesh_RVE.Get_B_e_pg(matrixType)

    C_hom = (wJ_e_pg * CMandel @ B_e_pg @ U_e).sum((0, 1)) / mesh_RVE.area

    if isHollow:
        coef = 1 - area_inclusion / mesh_RVE.area
        C_hom *= coef

    # print(np.linalg.eigvals(C_hom))

    # ----------------------------------------------
    # Comparison
    # ----------------------------------------------
    def Simulation(simu: Simulations._Simu, title=""):
        simu.Bc_Init()

        simu.add_dirichlet(simu.mesh.Nodes_Tags(["L3"]), [0, 0], ["x", "y"])
        simu.add_surfLoad(simu.mesh.Nodes_Tags(["L1"]), [-load / (b * h)], ["y"])

        simu.Solve()

        # Display.Plot_BoundaryConditions(simu)
        Display.Plot_Result(simu, "uy", title=f"{title} uy")
        # Display.Plot_Result(simu, "Eyy")

        print(
            f"{title}: dy = {np.max(simu.Result('uy')[simu.mesh.Nodes_Point(Geoms.Point(L, 0))]):.3f}"
        )

    Simulation(simu_inclusions, "inclusions")
    Simulation(simu, "non hom")

    testSym = np.linalg.norm(C_hom.T - C_hom) / np.linalg.norm(C_hom)

    if testSym >= 1e-12 and testSym <= 1e-3:
        C_hom = 1 / 2 * (C_hom.T + C_hom)

    material.Set_C(C_hom, False)
    Simulation(simu, "hom")

    # ax = Display.Plot_Result(simu, "uy")
    # Display.Plot_Result(simuInclusions, "uy", ax=ax)

    plt.show()
