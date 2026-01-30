# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog4
======

Conduct 3d homogenization.

WARNING
-------
Verify that the periodic boundary conditions have been correctly applied.
"""
# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Models, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Points, Circle
from EasyFEA.FEM import Mesh, LagrangeCondition, FeArray
from typing import Optional


def Compute_ukl(
    simu: Simulations.Elastic,
    nodes_border: np.ndarray,
    Ekl: np.ndarray,
    paired_nodes: Optional[np.ndarray] = None,
    pltSol=False,
):
    simu.Bc_Init()
    mesh = simu.mesh
    coord = mesh.coord

    usePER = paired_nodes is not None

    def func_ux(x, y, z):
        return Ekl.dot([x, y, z])[0]

    def func_uy(x, y, z):
        return Ekl.dot([x, y, z])[1]

    def func_uz(x, y, z):
        return Ekl.dot([x, y, z])[2]

    directions = ["x", "y", "z"]

    simu.add_dirichlet(nodes_border, [func_ux, func_uy, func_uz], directions)

    if usePER:
        for n0, n1 in paired_nodes:
            nodes = np.array([n0, n1])

            for d, direction in enumerate(directions):
                dofs = simu.Bc_dofs_nodes(nodes, [direction])

                values = Ekl @ [
                    coord[n0, 0] - coord[n1, 0],
                    coord[n0, 1] - coord[n1, 1],
                    coord[n0, 2] - coord[n1, 2],
                ]
                value = values[d]

                condition = LagrangeCondition(
                    "elastic", nodes, dofs, [direction], [value], [1, -1]
                )
                simu._Bc_Add_Lagrange(condition)

    ukl = simu.Solve()

    simu.Save_Iter()

    if pltSol:
        PyVista.Plot(simu, "ux", deformFactor=0.3).show()
        PyVista.Plot(simu, "uy", deformFactor=0.3).show()

        PyVista.Plot(simu, "Sxx", deformFactor=0.3).show()
        PyVista.Plot(simu, "Syy", deformFactor=0.3).show()
        PyVista.Plot(simu, "Sxy", deformFactor=0.3).show()

    return ukl


def Get_nodes(mesh: Mesh, dimElem: int):
    """Returns\n
    nodesLeft, nodesRight, nodesUpper, nodesLower, nodesFront, nodesBack"""

    coord = mesh.coord

    conditions = {
        "left": lambda x, y, z: x == coord[:, 0].min(),
        "right": lambda x, y, z: x == coord[:, 0].max(),
        "lower": lambda x, y, z: y == coord[:, 1].min(),
        "upper": lambda x, y, z: y == coord[:, 1].max(),
        "back": lambda x, y, z: z == coord[:, 2].min(),
        "front": lambda x, y, z: z == coord[:, 2].max(),
    }

    # get nodes
    nodes = {key: [] for key in conditions}
    for groupElem in mesh.Get_list_groupElem(dimElem):
        for key, condition in conditions.items():
            nodes[key].extend(groupElem.Get_Nodes_Conditions(condition))

    return tuple(np.asarray(nodes[key]) for key in conditions)


def Get_kubc_nodes(mesh: Mesh, dimElem: int = 0):
    """Returns nodes_kubc"""

    coord = mesh.coord

    # conditions to get nodes on x and y faces
    conditions = [
        lambda x, y, z: x == coord[:, 0].min(),
        lambda x, y, z: x == coord[:, 0].max(),
        lambda x, y, z: y == coord[:, 1].min(),
        lambda x, y, z: y == coord[:, 1].max(),
        # lambda x, y, z: z == coord[:, 2].min(),
        # lambda x, y, z: z == coord[:, 2].max(),
    ]

    # get nodes
    nodes_kubc: set[int] = set()
    for groupElem in mesh.Get_list_groupElem(dimElem):
        for condition in conditions:
            nodes = groupElem.Get_Nodes_Conditions(condition)
            nodes_kubc = nodes_kubc.union(nodes)

    return np.asarray(list(nodes_kubc), dtype=int)


if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # use Periodic boundary conditions ?
    usePBC = True
    plotPBC = True
    plotSurfaces = False

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = 1 / 10

    p0 = (-1 / 2, -1 / 2)
    p1 = (1 / 2, -1 / 2)
    p2 = (1 / 2, 1 / 2)
    p3 = (-1 / 2, 1 / 2)
    pts = [p0, p1, p2, p3]
    contour = Points(pts, meshSize)

    # inclusion
    f = 0.4
    r = 1 * np.sqrt(f / np.pi)
    inclusion = Circle((0, 0), 2 * r, meshSize, isHollow=False)
    # contour.Plot_Geoms([contour, inclusion])

    elemType = ElemType.PRISM6
    mesh = contour.Mesh_Extrude([inclusion], [0, 0, 1], [1 / meshSize], elemType)

    plotter = PyVista.Plot_Mesh(mesh)
    plotter.add_title("RVE")
    plotter.show()

    # ----------------------------------------------
    # Get paired nodes
    # ----------------------------------------------

    # PyVista.Plot_Tags(mesh).show()

    nodesLeft, nodesRight, nodesLower, nodesUpper, nodesFront, nodesBack = Get_nodes(
        mesh, dimElem=2
    )

    if plotSurfaces:
        plotter = PyVista.Plot(mesh, alpha=0.1)
        colors = plt.get_cmap("tab10").colors

        dict_nodes = {
            f"{nodesLeft=}".split("=")[0]: nodesLeft,
            f"{nodesRight=}".split("=")[0]: nodesRight,
            f"{nodesLower=}".split("=")[0]: nodesLower,
            f"{nodesUpper=}".split("=")[0]: nodesUpper,
            f"{nodesFront=}".split("=")[0]: nodesFront,
            f"{nodesBack=}".split("=")[0]: nodesBack,
        }

        for i, (name, nodes) in enumerate(dict_nodes.items()):
            PyVista.Plot_Elements(
                mesh, nodes, 2, color=colors[i], label=name, plotter=plotter
            )
        plotter.add_legend()
        plotter.show()

    if usePBC:
        # Hybrid PBC setup
        nodes_kubc = Get_kubc_nodes(mesh, dimElem=0)
        # PyVista.Plot_Nodes(mesh, nodes_kubc).show()

        # get nodes along x, y and z directions
        nodes_x = set(nodesLeft).union(nodesRight)
        nodes_y = set(nodesLower).union(nodesUpper)
        nodes_z = set(nodesFront).union(nodesBack)

        # get nodes along ij directions
        nodes_xy = nodes_x.union(nodes_y)
        nodes_xz = nodes_x.union(nodes_z)
        nodes_yz = nodes_y.union(nodes_z)

        # Get nodes along the y-axis without nodes along x axis.
        nodesLower = np.array([node for node in nodesLower if node not in nodes_x])
        nodesUpper = np.array([node for node in nodesUpper if node not in nodes_x])

        # Get nodes along the z-axis without nodes along x and y axis.
        nodesFront = np.array([node for node in nodesFront if node not in nodes_xy])
        nodesBack = np.array([node for node in nodesBack if node not in nodes_xy])

        paired_surfaces = [
            (nodesLeft, nodesRight),  # x
            (nodesLower, nodesUpper),  # y
            (nodesFront, nodesBack),  # z
        ]

        # get paired nodes
        for i, (nodes1, nodes2) in enumerate(paired_surfaces):

            # remove kubc nodes
            nodes1 = np.array(list(set(nodes1) - set(nodes_kubc)))
            nodes2 = np.array(list(set(nodes2) - set(nodes_kubc)))

            # sort nodes
            sorted_nodes2 = np.zeros_like(nodes1)
            coord = mesh.coord
            for n, node in enumerate(nodes1):
                dist = np.linalg.norm(coord[nodes2] - coord[node], axis=1)
                sorted_nodes2[n] = nodes2[dist.argmin()]
            nodes2 = sorted_nodes2

            if plotPBC:
                plotter = PyVista.Plot_Mesh(mesh, alpha=0.1)
                PyVista.Plot_Nodes(mesh, nodes1, plotter=plotter)
                PyVista.Plot_Nodes(mesh, nodes2, plotter=plotter)
                plotter.add_lines(
                    np.concatenate(
                        (mesh.coord[nodes1], mesh.coord[nodes2]), axis=1
                    ).reshape(-1, 3),
                    color="k",
                )
                plotter.add_title(f"paired nodes along {['x','y','z'][i]} axis.")
                plotter.show()

            # set new paired nodes
            newPairedNodes = np.array([nodes1, nodes2]).T
            if i == 0:
                paired_nodes = newPairedNodes
            else:
                paired_nodes = np.concatenate((paired_nodes, newPairedNodes))

    else:
        nodes_kubc = set(
            np.concatenate(
                [nodesLeft, nodesRight, nodesUpper, nodesLower, nodesBack, nodesFront]
            )
        )
        nodes_kubc = list(nodes_kubc)
        paired_nodes = None

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    elements_inclusion = mesh.Elements_Tags(["V1"])
    elements_matrix = mesh.Elements_Tags(["V0"])

    E = np.zeros_like(mesh.groupElem.elements, dtype=float)
    v = np.zeros_like(mesh.groupElem.elements, dtype=float)

    E[elements_matrix] = 1  # MPa
    v[elements_matrix] = 0.45

    if elements_inclusion.size > 0:
        E[elements_inclusion] = 50
        v[elements_inclusion] = 0.3

    material = Models.Elastic.Isotropic(3, E, v)

    simu = Simulations.Elastic(mesh, material)

    PyVista.Plot(simu, E, nodeValues=False, colorbarTitle="E [MPa]").show()
    PyVista.Plot(simu, v, nodeValues=False, colorbarTitle="v").show()

    # ----------------------------------------------
    # Homogenization
    # ----------------------------------------------
    r2 = np.sqrt(2)
    E11 = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    E22 = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    E33 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]
    )
    E12 = np.array(
        [
            [0, 1 / r2, 0],
            [1 / r2, 0, 0],
            [0, 0, 0],
        ]
    )
    E13 = np.array(
        [
            [0, 0, 1 / r2],
            [0, 0, 0],
            [1 / r2, 0, 0],
        ]
    )
    E23 = np.array(
        [
            [0, 0, 0],
            [0, 0, 1 / r2],
            [0, 1 / r2, 0],
        ]
    )

    u11 = Compute_ukl(simu, nodes_kubc, E11, paired_nodes, True)
    u22 = Compute_ukl(simu, nodes_kubc, E22, paired_nodes)
    u33 = Compute_ukl(simu, nodes_kubc, E33, paired_nodes)
    u12 = Compute_ukl(simu, nodes_kubc, E12, paired_nodes, True)
    u13 = Compute_ukl(simu, nodes_kubc, E13, paired_nodes)
    u23 = Compute_ukl(simu, nodes_kubc, E23, paired_nodes)

    u11_e = mesh.Locates_sol_e(u11, asFeArray=True)
    u22_e = mesh.Locates_sol_e(u22, asFeArray=True)
    u33_e = mesh.Locates_sol_e(u33, asFeArray=True)
    u12_e = mesh.Locates_sol_e(u12, asFeArray=True)
    u13_e = mesh.Locates_sol_e(u13, asFeArray=True)
    u23_e = mesh.Locates_sol_e(u23, asFeArray=True)

    # ----------------------------------------------
    # Effective elasticity tensor (C_hom)
    # ----------------------------------------------
    U_e = FeArray.zeros(*u11_e.shape, 6)

    U_e[..., 0] = u11_e
    U_e[..., 1] = u22_e
    U_e[..., 2] = u33_e
    U_e[..., 3] = u23_e
    U_e[..., 4] = u13_e
    U_e[..., 5] = u12_e

    matrixType = "mass"
    wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = mesh.Get_B_e_pg(matrixType)

    C_Mat = Models.Reshape_variable(material.C, *B_e_pg.shape[:2])

    # Be careful here you have to use all the volume even if there are holes
    # if you use the mesh volume, multiply C_hom by the porosity (1-f)
    C_hom = (wJ_e_pg * C_Mat @ B_e_pg @ U_e).sum((0, 1)) / mesh.volume

    if inclusion.isHollow:
        C_hom *= 1 - f

    formatted_array = ""
    for i in range(6):
        formatted_array += "\n"
        for j in range(6):
            formatted_array += f"{C_hom[i,j]:10.3e} "

    print("C_hom =", formatted_array)

    plt.show()
