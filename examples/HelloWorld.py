# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
HelloWorld
==========

A cantilever beam undergoing bending deformation.
"""

from EasyFEA import Display, ElemType, Materials, Simulations
from EasyFEA.Geoms import Domain

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L = 120  # mm
h = 13

domain = Domain((0, 0), (L, h), h / 3)
mesh = domain.Mesh_2D([], ElemType.QUAD9, isOrganised=True)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
E = 210000  # MPa
v = 0.3
F = -800  # N

mat = Materials.ElasIsot(2, E, v, planeStress=True, thickness=h)

simu = Simulations.ElasticSimu(mesh, mat)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
simu.add_surfLoad(nodesXL, [F / h / h], ["y"])

simu.Solve()

# ----------------------------------------------
# Results
# ----------------------------------------------
Display.Plot_Mesh(simu, deformFactor=10)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, "uy", plotMesh=True)
Display.Plot_Result(simu, "Svm", plotMesh=True, ncolors=11)

Display.plt.show()
