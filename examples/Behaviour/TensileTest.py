# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _TensileTest:

TensileTest
===========

The specimen of :ref:`Mesh11` pulled past yield, and the one thing that must hold exactly.

**The answer cannot depend on how the specimen is oriented.** ``Mesh11`` turns it twice into a
general orientation, and for an isotropic material the response must then be *identical* to the
unrotated one — not close, identical to round-off, because rotating the mesh and the boundary
conditions together is the same problem written in another frame.
"""

import numpy as np

from EasyFEA import Folder, Matplotlib, ElemType, Models, PyVista, Simulations
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
    # PyVista.Plot_Geoms([contour, grip1, grip2]).show()

    return contour.Mesh_Extrude(
        [],
        [0, 0, e],
        [3],
        isOrganised=True,
        elemType=ElemType.HEXA27,
        additionalSurfaces=[grip1, grip2],
    )


def Pull(rotate: bool, steps: int = 20):
    """Clamp one grip, drag the other along the specimen axis."""
    mesh = Mesh()

    # node numbering survives Rotate, so the grips are picked out while still axis-aligned
    y = mesh.coord[:, 1]
    bottom = np.where(np.isclose(y, y.min()))[0]
    top = np.where(np.isclose(y, y.max()))[0]

    if rotate:
        mesh.Rotate(-45, mesh.center)
        mesh.Rotate(45, mesh.center, (1, 0))

    # the axis, read off the mesh rather than from the rotation convention
    axis = mesh.coord[top].mean(axis=0) - mesh.coord[bottom].mean(axis=0)
    axis /= np.linalg.norm(axis)

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
        simu.add_dirichlet(top, list(u * axis), ["x", "y", "z"])
        simu.Solve()
        simu.Save_Iter()

    return simu, mesh


simuRef, mesh = Pull(rotate=False)
simuRot, meshRotated = Pull(rotate=True)

# ----------------------------------------------
# The same problem in another frame
# ----------------------------------------------
svm = simuRef.Result("Svm", nodeValues=False)
svmRot = simuRot.Result("Svm", nodeValues=False)
p = simuRef.Result("p", nodeValues=False)
pRot = simuRot.Result("p", nodeValues=False)
psi = simuRef.Results_dict_Energy()[r"$\Psi$"]
psiRot = simuRot.Results_dict_Energy()[r"$\Psi$"]

dSvm = np.max(np.abs(svmRot - svm)) / svm.max()
dP = np.max(np.abs(pRot - p)) / p.max()
dPsi = abs(psiRot / psi - 1)

print(f"peak von Mises stress {svm.max():.1f} MPa, peak plastic strain {p.max():.4f}")
print("\nrotated into a general orientation, relative to the reference run:")
print(f"  von Mises stress      {dSvm:.2e}")
print(f"  accumulated plastic p {dP:.2e}")
print(f"  stored energy         {dPsi:.2e}")

# round-off, amplified through the nonlinear increments and a different pivot order in the
# rotated system -- still orders below anything that could be a real dependence
assert dSvm < 1e-8, "the stress field depends on how the specimen is oriented"
assert dP < 1e-8, "the plastic strain depends on how the specimen is oriented"
assert dPsi < 1e-8, "the stored energy depends on how the specimen is oriented"

# how far from uniaxial the waist really is
sxx_e, syy_e, szz_e = simuRef.Result("Stress", nodeValues=False)[:, :3].T
i = int(np.argmax(svm))
centre = mesh.coord[mesh.connect].mean(axis=1)
print(f"\nmost stressed element, at y = {centre[i, 1]:.2f} of a {H:.0f} mm gauge:")
print(f"  Sxx {sxx_e[i]:7.1f}   Syy {syy_e[i]:7.1f}   Szz {szz_e[i]:7.1f} MPa")
transverse = max(abs(sxx_e[i]), abs(szz_e[i])) / abs(syy_e[i])
print(
    f"  transverse/axial = {transverse:.1%}: clamped grips, so not quite a uniaxial test"
)

# ----------------------------------------------
# Results
# ----------------------------------------------
ax = Matplotlib.Init_Axes()
ax.plot(centre[:, 1], svm, "o", ms=4, label="reference")
ax.plot(centre[:, 1], svmRot, "+", ms=7, label="rotated")
ax.axvline(H / 2, ls=":", c="k", lw=0.8)
ax.text(H / 2, svm.min(), " waist", fontsize=8)
ax.set_xlabel("$y$ along the specimen [mm]")
ax.set_ylabel(r"$\sigma_{vm}$ per element [MPa]")
ax.set_title("The two orientations give one curve")
ax.legend()
ax.grid(alpha=0.3)

PyVista.Plot_Mesh(meshRotated).show()
PyVista.Plot_BoundaryConditions(simuRef).show()
PyVista.Movie_simu(simuRef, "p", folder, "p.gif", deformFactor=2)

Matplotlib.plt.show()
