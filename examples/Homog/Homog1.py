# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog1
======

Conduct homogenization using an example outlined in 'Computational Homogenization of Heterogeneous Materials with Finite Elements'.

Reference: https://doi.org/10.1007/978-3-030-18383-7
Section 4.7 with corrected values on page 89 (Erratum).
"""
# sphinx_gallery_thumbnail_number = -1

from EasyFEA import Display, plt, np, ElemType, Materials, Simulations
from EasyFEA.Geoms import Points, Circle
from EasyFEA.fem import LagrangeCondition, FeArray
from typing import Optional


def Calc_ukl(
    simu: Simulations.ElasticSimu,
    nodes_border: np.ndarray,
    Ekl: np.ndarray,
    paired_nodes: Optional[np.ndarray] = None,
    pltSol=False,
    useMean0=False,
):

    simu.Bc_Init()
    mesh = simu.mesh
    coord = mesh.coordGlob

    usePER = paired_nodes is not None

    def func_ux(x, y, z):
        return Ekl.dot([x, y])[0]

    def func_uy(x, y, z):
        return Ekl.dot([x, y])[1]

    simu.add_dirichlet(nodes_border, [func_ux, func_uy], ["x", "y"])

    if usePER:

        for n0, n1 in paired_nodes:

            nodes = np.array([n0, n1])

            for direction in ["x", "y"]:
                dofs = simu.Bc_dofs_nodes(nodes, [direction])

                values = Ekl @ [
                    coord[n0, 0] - coord[n1, 0],
                    coord[n0, 1] - coord[n1, 1],
                ]
                value = values[0] if direction == "x" else values[1]

                condition = LagrangeCondition(
                    "elastic", nodes, dofs, [direction], [value], [1, -1]
                )
                simu._Bc_Add_Lagrange(condition)

    if useMean0:

        nodes = mesh.nodes
        vect = np.ones(mesh.Nn) * 1 / mesh.Nn

        # sum u_i / Nn = 0
        dofs = simu.Bc_dofs_nodes(nodes, ["x"])
        condition = LagrangeCondition("elastic", nodes, dofs, ["x"], [0], [vect])
        simu._Bc_Add_Lagrange(condition)

        # sum v_i / Nn = 0
        dofs = simu.Bc_dofs_nodes(nodes, ["y"])
        condition = LagrangeCondition("elastic", nodes, dofs, ["y"], [0], [vect])
        simu._Bc_Add_Lagrange(condition)

    ukl = simu.Solve()

    simu.Save_Iter()

    if pltSol:
        Display.Plot_Result(simu, "ux", deformFactor=0.3)
        Display.Plot_Result(simu, "uy", deformFactor=0.3)

        Display.Plot_Result(simu, "Sxx", deformFactor=0.3)
        Display.Plot_Result(simu, "Syy", deformFactor=0.3)
        Display.Plot_Result(simu, "Sxy", deformFactor=0.3)

    return ukl


if __name__ == "__main__":

    Display.Clear()

    # use Periodic boundary conditions ?
    usePER = True  # FALSE mean KUBC

    # ----------------------------------------------------------------------------
    # Mesh
    # ----------------------------------------------------------------------------
    p0 = (-1 / 2, -1 / 2)
    p1 = (1 / 2, -1 / 2)
    p2 = (1 / 2, 1 / 2)
    p3 = (-1 / 2, 1 / 2)
    pts = [p0, p1, p2, p3]

    meshSize = 1 / 15

    contour = Points(pts, meshSize)

    f = 0.4

    r = 1 * np.sqrt(f / np.pi)

    inclusion = Circle((0, 0), 2 * r, meshSize, isHollow=False)

    mesh = contour.Mesh_2D([inclusion], ElemType.TRI6)

    Display.Plot_Mesh(mesh, title="RVE")

    if usePER:
        nodes_border = mesh.Nodes_Tags(["P0", "P1", "P2", "P3"])
        paired_nodes = mesh.Get_Paired_Nodes(nodes_border, True)
    else:
        nodes_border = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
        paired_nodes = None

    # ----------------------------------------------------------------------------
    # Material and Simulation
    # ----------------------------------------------------------------------------
    elements_inclusion = mesh.Elements_Tags(["S1"])
    elements_matrix = mesh.Elements_Tags(["S0"])

    E = np.zeros_like(mesh.groupElem.elements, dtype=float)
    v = np.zeros_like(mesh.groupElem.elements, dtype=float)

    E[elements_matrix] = 1  # MPa
    v[elements_matrix] = 0.45

    if elements_inclusion.size > 0:
        E[elements_inclusion] = 50
        v[elements_inclusion] = 0.3

    material = Materials.ElasIsot(2, E, v, planeStress=False)

    simu = Simulations.ElasticSimu(mesh, material, useNumba=True)

    Display.Plot_Result(simu, E, nodeValues=False, title="E [MPa]")
    Display.Plot_Result(simu, v, nodeValues=False, title="v")

    # ----------------------------------------------------------------------------
    # Homogenization
    # ----------------------------------------------------------------------------
    r2 = np.sqrt(2)
    E11 = np.array([[1, 0], [0, 0]])
    E22 = np.array([[0, 0], [0, 1]])
    E12 = np.array([[0, 1 / r2], [1 / r2, 0]])

    u11 = Calc_ukl(simu, nodes_border, E11, paired_nodes)
    u22 = Calc_ukl(simu, nodes_border, E22, paired_nodes)
    u12 = Calc_ukl(simu, nodes_border, E12, paired_nodes, True)

    u11_e = mesh.Locates_sol_e(u11, asFeArray=True)
    u22_e = mesh.Locates_sol_e(u22, asFeArray=True)
    u12_e = mesh.Locates_sol_e(u12, asFeArray=True)

    # ----------------------------------------------------------------------------
    # Effective elasticity tensor (C_hom)
    # ----------------------------------------------------------------------------
    U_e = FeArray.zeros(*u11_e.shape, 3)

    U_e[..., 0] = u11_e
    U_e[..., 1] = u22_e
    U_e[..., 2] = u12_e

    matrixType = "mass"
    weightedJacobian_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = mesh.Get_B_e_pg(matrixType)

    C_Mat = Materials.Reshape_variable(material.C, *B_e_pg.shape[:2])

    # Be careful here you have to use all the area even if there are holes
    # if you use the mesh area, multiply C_hom by the porosity (1-f)
    C_hom = (weightedJacobian_e_pg * C_Mat @ B_e_pg @ U_e).sum((0, 1)) / mesh.area

    if inclusion.isHollow and mesh.area != 1:
        C_hom *= 1 - f

    # Display.Plot_BoundaryConditions(simu)

    print(f"f = {f}")
    print(f"c1111 = {C_hom[0,0]}")
    print(f"c1122 = {C_hom[0,1]}")
    print(f"c1212 = {C_hom[2,2]/2}")

    plt.show()
