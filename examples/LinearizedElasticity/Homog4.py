# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog4
======

Conduct 3d homogenization on a simple RVE.
"""
# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Display, Models, Mesh, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Points, Circle
from EasyFEA.FEM import LagrangeCondition, FeArray
from typing import Optional


def Compute_ukl(
    simu: Simulations.Elastic,
    Ekl: np.ndarray,
    nodesKUBC: Optional[np.ndarray] = None,
    pairedNodes: Optional[np.ndarray] = None,
    pltSol=False,
):
    simu.Bc_Init()
    mesh = simu.mesh
    coord = mesh.coord

    directions = ["x", "y", "z"]

    if nodesKUBC is not None:

        def func_ux(x, y, z):
            return Ekl.dot([x, y, z])[0]

        def func_uy(x, y, z):
            return Ekl.dot([x, y, z])[1]

        def func_uz(x, y, z):
            return Ekl.dot([x, y, z])[2]

        simu.add_dirichlet(nodesKUBC, [func_ux, func_uy, func_uz], directions)

    if pairedNodes is not None:
        for n0, n1 in pairedNodes:
            nodes = np.array([n0, n1])

            delta = coord[n0] - coord[n1]
            values = Ekl @ delta

            for d, direction in enumerate(directions):
                dofs = simu.Bc_dofs_nodes(nodes, [direction])

                condition = LagrangeCondition(
                    "elastic", nodes, dofs, [direction], [values[d]], [1, -1]
                )
                simu._Bc_Add_Lagrange(condition)

        # remove rigid body motion
        nodes = mesh.nodes
        vect = np.ones(mesh.Nn) * 1 / mesh.Nn
        for direction in directions:
            # sum d_i / Nn = 0
            dofs = simu.Bc_dofs_nodes(nodes, [direction])
            condition = LagrangeCondition(
                "elastic", nodes, dofs, [direction], [0], [vect]
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


def Get_nodes(mesh: Mesh, tol=1e-6, plotSurfaces=False):
    """Returns nodesXm, nodesXp, nodesYm, nodesYp, nodesZm, nodesZp"""

    # PyVista.Plot_Tags(mesh).show()
    coord = mesh.coord
    nodesXm = mesh.Nodes_Conditions(lambda x, y, z: x <= coord[:, 0].min() + tol)
    nodesXp = mesh.Nodes_Conditions(lambda x, y, z: x >= coord[:, 0].max() - tol)
    nodesYm = mesh.Nodes_Conditions(lambda x, y, z: y <= coord[:, 1].min() + tol)
    nodesYp = mesh.Nodes_Conditions(lambda x, y, z: y >= coord[:, 1].max() - tol)
    nodesZm = mesh.Nodes_Conditions(lambda x, y, z: z <= coord[:, 2].min() + tol)
    nodesZp = mesh.Nodes_Conditions(lambda x, y, z: z >= coord[:, 2].max() - tol)

    if plotSurfaces:
        plotter = PyVista.Plot(mesh, alpha=0.1)
        colors = plt.get_cmap("tab10").colors

        dict_nodes = {
            f"{nodesXm=}".split("=")[0]: nodesXm,
            f"{nodesXp=}".split("=")[0]: nodesXp,
            f"{nodesYm=}".split("=")[0]: nodesYm,
            f"{nodesYp=}".split("=")[0]: nodesYp,
            f"{nodesZm=}".split("=")[0]: nodesZm,
            f"{nodesZp=}".split("=")[0]: nodesZp,
        }

        for i, (name, nodes) in enumerate(dict_nodes.items()):
            PyVista.Plot_Elements(
                mesh, nodes, 2, color=colors[i], label=name, plotter=plotter
            )
        plotter.add_legend()
        plotter.show()

    return nodesXm, nodesXp, nodesYm, nodesYp, nodesZm, nodesZp


def Get_pairedNodes(
    mesh: Mesh,
    nodesXm: np.ndarray,
    nodesXp: np.ndarray,
    nodesYm: np.ndarray,
    nodesYp: np.ndarray,
    nodesZm: np.ndarray,
    nodesZp: np.ndarray,
    plotPBC=False,
):

    # remove Xp nodes on Y nodes
    nodesYp = np.array(list(set(nodesYp) - set(nodesXp)))
    nodesYm = np.array(list(set(nodesYm) - set(nodesXp)))

    # remove Xp and Yp nodes on Z nodes
    nodesZp = np.array(list(set(nodesZp) - set(nodesXp) - set(nodesYp)))
    nodesZm = np.array(list(set(nodesZm) - set(nodesXp) - set(nodesYp)))

    list_tuple_nodes = [
        (nodesXm, nodesXp),
        (nodesYm, nodesYp),
        (nodesZm, nodesZp),
    ]

    # get paired nodes
    for i, (nodes1, nodes2) in enumerate(list_tuple_nodes):

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
        newPairedNodes = np.array([nodes1, nodes2], dtype=int).T
        if i == 0:
            pairedNodes = newPairedNodes
        else:
            pairedNodes = np.concatenate((pairedNodes, newPairedNodes))

    return pairedNodes


if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # use Periodic boundary conditions ?
    usePBC = True
    plotPBC = False
    plotSurfaces = False

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    meshSize = 1 / 5

    p0 = (-1 / 2, -1 / 2)
    p1 = (1 / 2, -1 / 2)
    p2 = (1 / 2, 1 / 2)
    p3 = (-1 / 2, 1 / 2)
    pts = [p0, p1, p2, p3]
    contour = Points(pts, meshSize)

    # inclusion
    f = 0.4
    r = 1 * np.sqrt(f / np.pi)
    inclusion = Circle((0, 0), 2 * r, meshSize, isFilled=True)
    # contour.Plot_Geoms([contour, inclusion])

    elemType = ElemType.PRISM15
    mesh = contour.Mesh_Extrude([inclusion], [0, 0, 1], [1 / meshSize], elemType)
    mesh.Translate(*-mesh.center)  # center mesh on 0,0,0

    plotter = PyVista.Plot_Mesh(mesh)
    plotter.show_grid()
    plotter.add_title("RVE")
    plotter.show()

    # ----------------------------------------------
    # Get paired nodes
    # ----------------------------------------------

    tuple_nodes = Get_nodes(mesh, plotSurfaces=plotSurfaces)
    if usePBC:
        nodesKUBC = None
        pairedNodes = Get_pairedNodes(mesh, *tuple_nodes, plotPBC=plotPBC)
    else:
        nodesKUBC = set(np.concatenate(tuple_nodes))
        nodesKUBC = list(nodesKUBC)
        pairedNodes = None

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    elements_inclusion = (
        np.array([]) if not inclusion.isFilled else mesh.Elements_Tags(["V1"])
    )
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
    E1 = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    E2 = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    E3 = np.array(
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

    u11 = Compute_ukl(simu, E1, nodesKUBC, pairedNodes, True)
    u22 = Compute_ukl(simu, E2, nodesKUBC, pairedNodes)
    u33 = Compute_ukl(simu, E3, nodesKUBC, pairedNodes)
    u12 = Compute_ukl(simu, E12, nodesKUBC, pairedNodes, True)
    u13 = Compute_ukl(simu, E13, nodesKUBC, pairedNodes)
    u23 = Compute_ukl(simu, E23, nodesKUBC, pairedNodes)

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

    if not inclusion.isFilled:
        C_hom *= 1 - f

    formatted_array = ""
    for i in range(6):
        formatted_array += "\n"
        for j in range(6):
            formatted_array += f"{C_hom[i,j]:10.3e} "

    print("C_hom =", formatted_array)

    plt.show()
