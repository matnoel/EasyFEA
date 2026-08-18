# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
PlasticPlate
============

A perforated plate pulled past yield. The hole concentrates the stress, so plasticity starts
there and spreads outwards as the load increases.

Nothing here is specific to plasticity: the simulation asks the material for a stress and a
tangent, and the material decides what that means.
"""

# sphinx_gallery_thumbnail_number = -1

import numpy as np

from EasyFEA import Folder, ElemType, Models, Simulations, PyVista
from EasyFEA.Geoms import Point, Points
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
folder = Folder.Results_Dir()

L, h, r = 120.0, 60.0, 12.0  # mm
thickness = 5.0

E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa
eps_y = sigma_y / E

uMax = (eps_y * L / 2) * 3  # mm
nStep = 20

# ----------------------------------------------
# Mesh
# ----------------------------------------------
contour = Points(
    [
        Point(0, 0, r=-r),
        (L / 2, 0),
        (L / 2, h / 2),
        (0, h / 2),
    ],
    meshSize=h / 15,
)
mesh = contour.Mesh_2D([], ElemType.TRI6)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L / 2)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
material = Models.InElastic.Behavior(
    2,
    Isotropic(3, E=E, v=v),
    hardening=Models.InElastic.IsotropicHardening.Voce(120.0, 40.0),
    yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
    thickness=thickness,
    planeStress=True,
)

simu = Simulations.InElastic(mesh, material)

for u in np.linspace(uMax / nStep, uMax, nStep):
    simu.Bc_Init()
    simu.add_dirichlet(nodesX0, [0], ["x"])
    simu.add_dirichlet(nodesY0, [0], ["y"])
    simu.add_dirichlet(nodesXL, [u], ["x"])
    simu.Solve()
    simu.Save_Iter()

p_field = simu.Result("p", nodeValues=False)
print(f"yielded elements: {np.count_nonzero(p_field > 0)} / {mesh.Ne}")
print(f"max accumulated plastic strain: {p_field.max():.4f}")

# ----------------------------------------------
# Results
# ----------------------------------------------
print(simu)
PyVista.Plot_BoundaryConditions(simu).show()
PyVista.Plot(simu, "Svm", plotMesh=True, nColors=11).show()
plotter = PyVista.Plot(simu, "p", plotMesh=True, nColors=11)
plotter.add_title("accumulated plastic strain")
plotter.show()

PyVista.Movie_simu(simu, "p", folder, "p.gif", deformFactor=10)
