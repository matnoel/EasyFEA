# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
PlasticPlate
============

The same behaviour on a mesh.

A perforated plate pulled past yield. The hole concentrates the stress, so plasticity starts
there and spreads outwards as the load increases.

Nothing here is specific to plasticity: the simulation asks the material for a stress and a
tangent, and the material decides what that means.
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np

from EasyFEA import Matplotlib, ElemType, Models, Simulations
from EasyFEA.Geoms import Point, Points
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Mesh
# ----------------------------------------------
L, h, r = 120.0, 60.0, 12.0  # mm
thickness = 5.0

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

# ----------------------------------------------
# Material
# ----------------------------------------------
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa

material = Models.Behaviour(
    2,
    Isotropic(3, E=E, v=v),
    hardening=Models.IsotropicHardening.Voce(120.0, 40.0),
    yieldSurface=Models.Yield.VonMises(sigma_y),
    thickness=thickness,
    planeStress=True,
)

simu = Simulations.Behaviour(mesh, material)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L / 2)

# ----------------------------------------------
# Load incrementally — plasticity is path dependent
# ----------------------------------------------
uMax, Nu = 0.35, 40  # mm
for u in np.linspace(uMax / Nu, uMax, Nu):
    simu.Bc_Init()
    simu.add_dirichlet(nodesX0, [0], ["x"])
    simu.add_dirichlet(nodesY0, [0], ["y"])
    simu.add_dirichlet(nodesXL, [u], ["x"])
    simu.Solve()
    simu.Save_Iter()

p_field = simu.Result("p", nodeValues=False)
print(f"yielded elements: {np.count_nonzero(p_field > 0)} / {mesh.Ne}")
print(f"max accumulated plastic strain: {p_field.max():.4f}")

# no closed form here, but the two things the figures claim are still checkable: plasticity
# starts at the hole, and it has spread without reaching the far edge
center = mesh.coord[mesh.connect].mean(axis=1)
radius = np.hypot(center[:, 0], center[:, 1])
yielded = p_field > 0
print(
    f"plastic zone spans r = {radius[yielded].min():.1f} to {radius[yielded].max():.1f} mm, "
    f"hole at {r:.0f}, plate out to {radius.max():.1f}"
)
assert radius[np.argmax(p_field)] < 1.1 * r, "plasticity did not start at the hole"
assert radius[yielded].max() < radius.max(), "the whole plate yielded"

# ----------------------------------------------
# Results
# ----------------------------------------------
print(simu)
Matplotlib.Plot_Mesh(mesh)
Matplotlib.Plot(simu, "Svm", plotMesh=True, ncolors=11, title="von Mises stress [MPa]")
Matplotlib.Plot(
    simu, "p", plotMesh=True, ncolors=11, title="accumulated plastic strain"
)

Matplotlib.plt.show()
