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
# sphinx_gallery_thumbnail_number = 2

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
uMax = 10 * eps_y * H
steps = 20


# ----------------------------------------------
# The specimen of Mesh11
# ----------------------------------------------
def Mesh():
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

    return contour.Mesh_Extrude(
        [],
        [0, 0, e],
        [3],
        isOrganised=True,
        elemType=ElemType.HEXA27,
        additionalSurfaces=[grip1, grip2],
    )


# ----------------------------------------------
# Clamp one grip, drag the other along the axis
# ----------------------------------------------
mesh = Mesh()
y = mesh.coord[:, 1]
bottom = np.where(np.isclose(y, y.min()))[0]
top = np.where(np.isclose(y, y.max()))[0]

material = Models.Behaviour(
    3,
    Isotropic(3, E=E, v=v),
    yieldSurface=Models.Yield.VonMises(sigma_y),
    hardening=Models.IsotropicHardening.Voce(Q, b),
)
simu = Simulations.Behaviour(mesh, material)

for u in np.linspace(uMax / steps, uMax, steps):
    simu.Bc_Init()
    simu.add_dirichlet(bottom, [0, 0, 0], ["x", "y", "z"])
    simu.add_dirichlet(top, [0, u, 0], ["x", "y", "z"])
    simu.Solve()
    simu.Save_Iter()

# ----------------------------------------------
# Where it yielded
# ----------------------------------------------
svm = simu.Result("Svm", nodeValues=False)
p = simu.Result("p", nodeValues=False)
centre = mesh.coord[mesh.connect].mean(axis=1)

print(f"peak von Mises stress {svm.max():.1f} MPa, peak plastic strain {p.max():.4f}")
print(f"yielded elements: {np.count_nonzero(p > 0)} / {mesh.Ne}")

yielded = p > 0
print(
    f"plastic zone spans y = {centre[yielded, 1].min():.2f} to "
    f"{centre[yielded, 1].max():.2f} of a {H:.0f} mm gauge, waist at {H / 2:.1f}"
)
# the waist is the smallest section, so that is where it must yield, and the grips must not
assert abs(centre[np.argmax(p), 1] - H / 2) < H / 4, "it did not yield at the waist"
assert p[centre[:, 1] < 0].max() == 0.0, "the lower grip yielded"

# how far from uniaxial the waist really is
sxx_e, syy_e, szz_e = simu.Result("Stress", nodeValues=False)[:, :3].T
i = int(np.argmax(svm))
print(f"\nmost stressed element, at y = {centre[i, 1]:.2f}:")
print(f"  Sxx {sxx_e[i]:7.1f}   Syy {syy_e[i]:7.1f}   Szz {szz_e[i]:7.1f} MPa")
transverse = max(abs(sxx_e[i]), abs(szz_e[i])) / abs(syy_e[i])
print(f"  transverse/axial = {transverse:.1%}, so not quite a uniaxial test")

# ----------------------------------------------
# Results
# ----------------------------------------------
PyVista.Plot_BoundaryConditions(simu).show()
PyVista.Plot(simu, "p", deformFactor=2, plotMesh=True).show()
PyVista.Movie_simu(simu, "p", folder, "p.gif", deformFactor=2)
