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

from EasyFEA import Display, Folder, Models, Simulations, MeshIO, PyVista
from EasyFEA.FEM import FeArray

from Homog4 import Compute_ukl, Get_nodes, Get_pairedNodes

# if __name__ == "__main__":
#     Display.Clear()

#     # ----------------------------------------------
#     # Configuration
#     # ----------------------------------------------

#     # use Periodic boundary conditions ?
#     usePBC = True
#     plotPBC = False
#     plotSurfaces = False

#     folderResults = Folder.Results_Dir()
#     meshes_dir = Folder.Join(Folder.Dir(n=2), "_meshes")

#     # ----------------------------------------------
#     # Mesh
#     # ----------------------------------------------

#     gmshFile = Folder.Join(meshes_dir, "octet_truss.msh")
#     mesh = MeshIO.Gmsh_to_EasyFEA(gmshFile)
#     mesh.Translate(*-mesh.center)  # center mesh on 0,0,0

#     plotter = PyVista.Plot_Mesh(mesh)
#     plotter.show_grid()
#     plotter.add_title("RVE")
#     plotter.show()

#     # ----------------------------------------------
#     # Get paired nodes
#     # ----------------------------------------------

#     tuple_nodes = Get_nodes(mesh, plotSurfaces=plotSurfaces)
#     if usePBC:
#         nodesKUBC = None
#         pairedNodes = Get_pairedNodes(mesh, *tuple_nodes, plotPBC=plotPBC)
#     else:
#         nodesKUBC = set(np.concatenate(tuple_nodes))
#         nodesKUBC = list(nodesKUBC)
#         pairedNodes = None

#     # ----------------------------------------------
#     # Material and Simulation
#     # ----------------------------------------------
#     material = Models.Elastic.Isotropic(3, E=1, v=0.3)

#     simu = Simulations.Elastic(mesh, material)

#     # ----------------------------------------------
#     # Homogenization
#     # ----------------------------------------------
#     r2 = np.sqrt(2)
#     E1 = np.array(
#         [
#             [1, 0, 0],
#             [0, 0, 0],
#             [0, 0, 0],
#         ]
#     )
#     E2 = np.array(
#         [
#             [0, 0, 0],
#             [0, 1, 0],
#             [0, 0, 0],
#         ]
#     )
#     E3 = np.array(
#         [
#             [0, 0, 0],
#             [0, 0, 0],
#             [0, 0, 1],
#         ]
#     )
#     E12 = np.array(
#         [
#             [0, 1 / r2, 0],
#             [1 / r2, 0, 0],
#             [0, 0, 0],
#         ]
#     )
#     E13 = np.array(
#         [
#             [0, 0, 1 / r2],
#             [0, 0, 0],
#             [1 / r2, 0, 0],
#         ]
#     )
#     E23 = np.array(
#         [
#             [0, 0, 0],
#             [0, 0, 1 / r2],
#             [0, 1 / r2, 0],
#         ]
#     )

#     u11 = Compute_ukl(simu, E1, nodesKUBC, pairedNodes, True)
#     u22 = Compute_ukl(simu, E2, nodesKUBC, pairedNodes)
#     u33 = Compute_ukl(simu, E3, nodesKUBC, pairedNodes)
#     u12 = Compute_ukl(simu, E12, nodesKUBC, pairedNodes, True)
#     u13 = Compute_ukl(simu, E13, nodesKUBC, pairedNodes)
#     u23 = Compute_ukl(simu, E23, nodesKUBC, pairedNodes)

#     u11_e = mesh.Locates_sol_e(u11, asFeArray=True)
#     u22_e = mesh.Locates_sol_e(u22, asFeArray=True)
#     u33_e = mesh.Locates_sol_e(u33, asFeArray=True)
#     u12_e = mesh.Locates_sol_e(u12, asFeArray=True)
#     u13_e = mesh.Locates_sol_e(u13, asFeArray=True)
#     u23_e = mesh.Locates_sol_e(u23, asFeArray=True)

#     # ----------------------------------------------
#     # Effective elasticity tensor (C_hom)
#     # ----------------------------------------------
#     U_e = FeArray.zeros(*u11_e.shape, 6)

#     U_e[..., 0] = u11_e
#     U_e[..., 1] = u22_e
#     U_e[..., 2] = u33_e
#     U_e[..., 3] = u23_e
#     U_e[..., 4] = u13_e
#     U_e[..., 5] = u12_e

#     matrixType = "mass"
#     wJ_e_pg = mesh.Get_weightedJacobian_e_pg(matrixType)
#     B_e_pg = mesh.Get_B_e_pg(matrixType)

#     C_Mat = Models.Reshape_variable(material.C, *B_e_pg.shape[:2])

#     C_hom = (wJ_e_pg * C_Mat @ B_e_pg @ U_e).sum((0, 1)) / mesh.volume

#     formatted_array = ""
#     for i in range(6):
#         formatted_array += "\n"
#         for j in range(6):
#             formatted_array += f"{C_hom[i,j]:10.3e} "

#     print("C_hom =", formatted_array)

#     plt.show()
