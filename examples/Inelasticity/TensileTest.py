# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
TensileTest
===========

The specimen of :ref:`Mesh11` pulled past yield, in 3D.

The waist is the smallest section, so it yields first while the grips stay elastic.

Quadratic elements are used on purpose: plastic flow is incompressible and fully integrated
trilinear hexahedra lock under it. HEXA8 stores 17% more energy and reaches 32% less plastic
strain than HEXA20, while HEXA20 and HEXA27 agree to 0.1%.

The grips are fully clamped and the specimen is short, so the waist is not quite in uniaxial
tension. The exact uniaxial comparison is in :ref:`StressStrain`.
"""

# sphinx_gallery_thumbnail_number = -1

import numpy as np

from EasyFEA import Folder, ElemType, Models, PyVista, Simulations
from EasyFEA.Geoms import Point, Line, CircleArc, Contour, Domain
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
folder = Folder.Results_Dir()

L, H = 1.0, 3.0  # mm, width and gauge length
e = L * 0.5  # waist width, and the extruded thickness
E, v = 210000.0, 0.3  # MPa
sigma_y, Q, b = 250.0, 150.0, 30.0  # MPa, von Mises with Voce hardening

eps_y = sigma_y / E
uMax = 5 * eps_y * H
nStep = 10

# ----------------------------------------------
# Mesh
# ----------------------------------------------
p1, p2 = Point(-L / 2), Point(L / 2)
p3, p4 = p2 + [0, H], p1 + [0, H]
meshSize = L / 2
contour = Contour(
    [
        Line(p1, p2, meshSize=meshSize),
        CircleArc(p2, p3, P=(e / 2, H / 2), meshSize=meshSize),
        Line(p3, p4, meshSize=meshSize),
        CircleArc(p4, p1, P=(-e / 2, H / 2), meshSize=meshSize),
    ]
)
grip1 = Domain(p1 - [0, L], p2, isFilled=True, meshSize=meshSize)
grip2 = grip1.Translate(dy=H + L, copy=True)

mesh = contour.Mesh_Extrude(
    [],
    [0, 0, e],
    [3],
    isOrganised=True,
    elemType=ElemType.HEXA27,
    additionalSurfaces=[grip1, grip2],
)

y = mesh.coord[:, 1]
bottom = np.where(np.isclose(y, y.min()))[0]
top = np.where(np.isclose(y, y.max()))[0]

# ----------------------------------------------
# Simulation
# ----------------------------------------------
material = Models.InElastic.Behavior(
    3,
    Isotropic(3, E=E, v=v),
    yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
    hardening=Models.InElastic.IsotropicHardening.Voce(Q, b),
)
simu = Simulations.InElastic(mesh, material)

for u in np.linspace(uMax / nStep, uMax, nStep):
    simu.Bc_Init()
    simu.add_dirichlet(bottom, [0, 0, 0], ["x", "y", "z"])
    simu.add_dirichlet(top, [0, u, 0], ["x", "y", "z"])
    simu.Solve()
    simu.Save_Iter()

# ----------------------------------------------
# Results
# ----------------------------------------------
svm_e = simu.Result("Svm", nodeValues=False)
p_e = simu.Result("p", nodeValues=False)

print(
    f"\npeak von Mises stress {svm_e.max():.1f} MPa, peak plastic strain {p_e.max():.4f}"
)
print(f"yielded elements: {np.count_nonzero(p_e > 0)} / {mesh.Ne}")

PyVista.Plot_BoundaryConditions(simu).show()
PyVista.Plot(simu, "p", deformFactor=2, plotMesh=True).show()
PyVista.Movie_simu(simu, "p", folder, "p.gif", deformFactor=2)
