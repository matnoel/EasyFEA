# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Homog5
======

Conduct 3d homogenization on a periodic mesh generated with `microgen <https://microgen.readthedocs.io/en/v1.3.2/examples/mesh.html#periodic-mesh>`_.
"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np

from EasyFEA import Terminal, Folder, Models, Simulations, IO, PyVista
from EasyFEA.FEM import FeArray, MatrixType

from Homog4 import Compute_ukl, Get_nodes, Get_pairedNodes

if __name__ == "__main__":
    Terminal.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # use Periodic boundary conditions ?
    usePBC = True
    plotPBC = False
    plotSurfaces = False

    folderResults = Folder.Results_Dir()
    meshes_dir = Folder.Join(Folder.Dir(n=2), "_meshes")

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    gmshFile = Folder.Join(meshes_dir, "octet_truss.msh")
    mesh = IO.Gmsh.Load_mesh(gmshFile)
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
    material = Models.Elastic.Isotropic(3, E=1, v=0.3)

    simu = Simulations.Elastic(mesh, material)

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

    matrixType = MatrixType.mass
    u_e = np.stack(
        [mesh.Locates_sol_e(u) for u in (u11, u22, u33, u23, u13, u12)],
        axis=-1,
    )  # (Ne, nPe·dim, 6)

    # ----------------------------------------------
    # Effective elasticity tensor (C_hom)
    # ----------------------------------------------
    wJ_e_pg = mesh.groupElem.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = mesh.groupElem.Get_B_e_pg(matrixType)

    Ne, nPg, nS, _ = B_e_pg.shape
    u_e_pg = FeArray.from_e(u_e, nPg)
    C_Mat = FeArray.broadcast(material.C, Ne, nPg, tensor_shape=(nS, nS))

    xMin, yMin, zMin = mesh.coord.min(axis=0)
    xMax, yMax, zMax = mesh.coord.max(axis=0)
    volume = (xMax - xMin) * (yMax - yMin) * (zMax - zMin)

    C_hom = (wJ_e_pg * C_Mat @ B_e_pg @ u_e_pg).sum((0, 1)) / volume

    formatted_array = ""
    for i in range(6):
        formatted_array += "\n"
        for j in range(6):
            formatted_array += f"{C_hom[i,j]:10.3e} "

    print("C_hom =", formatted_array)

    plt.show()
