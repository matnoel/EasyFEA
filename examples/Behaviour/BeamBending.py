# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _BeamBending:

BeamBending
===========

The second closed-form check: a rectangular section bent past yield.

Yielding starts at the outer fibres and the plastic front moves inwards, leaving a shrinking
elastic core of half-depth :math:`c = \varepsilon_y/\kappa`. Integrating the stress over the
section gives the moment directly:

.. math::
    M = \sigma_y w \left(\frac{h^2}{4} - \frac{c^2}{3}\right)
    \qquad\Longrightarrow\qquad
    \frac{M}{M_e} = \frac32 - \frac12\left(\frac{\kappa_e}{\kappa}\right)^2

with :math:`M_e = \sigma_y w h^2/6` at first yield and :math:`M_p = \sigma_y w h^2/4` when the
core has vanished. Their ratio is the **shape factor**, exactly ``3/2`` for a rectangle — a pure
number, independent of material and size, which is why it is such a good check.

The end sections are given a linear axial displacement, which imposes the plane-sections
kinematics the closed form assumes. That is deliberate: it isolates the constitutive response
and the through-thickness integration, rather than also testing beam theory.

The sweep stops at :math:`\kappa/\kappa_e = 4`. With no hardening the bending tangent vanishes
as the section becomes fully plastic, so the global Newton grows ill-conditioned well before the
shape factor is reached — the curve approaches ``3/2`` without ever arriving there numerically.

The residual ~0.4% is **not** the return mapping: it is already there while the section is fully
elastic, where the FE answer is exact. Nodal stresses average over the elements meeting a node,
and the two free-surface nodes have only one, so the extreme fibre lags by exactly one element,
:math:`1/n`. On the moment that is :math:`O(1/n^2)` — measured 1.56 / 0.39 / 0.10 % at
:math:`n` = 8 / 16 / 32 elements through the depth.

Reference
---------
Chakrabarty, *Theory of Plasticity*, 3rd ed., Elsevier (2006), ch. 3 "Elastoplastic Bending and
Torsion".
"""

import numpy as np
from scipy.integrate import simpson

from EasyFEA import Folder, Matplotlib, ElemType, Models, PyVista, Simulations
from EasyFEA.Geoms import Domain, Line
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
folder = Folder.Results_Dir()

L, h, w = 80.0, 20.0, 5.0  # mm — length, depth, thickness
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa

M_e = sigma_y * w * h**2 / 6  # first yield
M_p = sigma_y * w * h**2 / 4  # fully plastic
kappa_e = 2 * sigma_y / (E * h)  # curvature at first yield

print(f"M_e = {M_e:.0f} N.mm, M_p = {M_p:.0f} N.mm, shape factor = {M_p / M_e:.3f}")


def Exact(ratio):
    """M/M_e as a function of kappa/kappa_e; linear until the outer fibre yields."""
    return np.where(ratio <= 1.0, ratio, 1.5 - 0.5 / np.maximum(ratio, 1e-12) ** 2)


# ----------------------------------------------
# Model
# ----------------------------------------------
domain = Domain((0, -h / 2), (L, h / 2), h / 16)
mesh = domain.Mesh_2D(
    [],
    ElemType.QUAD8,
    additionalLines=[Line((0, 0), (L, 0))],
    isOrganised=True,
)
# mesh = domain.Mesh_2D([], ElemType.QUAD8, isOrganised=True)

material = Models.Behaviour(
    2,
    Isotropic(3, E=E, v=v),
    yieldSurface=Models.Yield.VonMises(sigma_y),  # perfectly plastic
    thickness=w,
    planeStress=True,
)
simu = Simulations.Behaviour(mesh, material)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
nodesMid = mesh.Nodes_Conditions(lambda x, y, z: np.isclose(x, L / 2))
origin = mesh.Nodes_Conditions(lambda x, y, z: (x == 0) & (np.abs(y) < 1e-9))

yMid = mesh.coord[nodesMid, 1]
order = np.argsort(yMid)
yMid = yMid[order]

# ----------------------------------------------
# Bend it, curvature by curvature
# ----------------------------------------------
ratios = np.arange(0.4, 4.01, 0.4)
moments, profiles = [], {}

for ratio in ratios:
    kappa = ratio * kappa_e
    simu.Bc_Init()
    simu.add_dirichlet(nodesX0, [0], ["x"])
    simu.add_dirichlet(origin, [0], ["y"])
    # plane sections stay plane: u_x = -kappa * L * y at the far end
    simu.add_dirichlet(nodesXL, [lambda x, y, z: -kappa * L * y], ["x"])
    simu.Solve()
    simu.Save_Iter()

    sig_xx = simu.Result("Sxx")[nodesMid][order]
    # Simpson, since sig_xx * y is quadratic in y while the section is elastic
    moments.append(abs(simpson(sig_xx * yMid, x=yMid) * w))
    profiles[ratio] = sig_xx

moments = np.array(moments)
exact = Exact(ratios)

errors = 100 * np.abs(moments / M_e - exact) / exact

print("\nmoment-curvature against the closed form:")
for i in range(0, len(ratios), 4):
    print(
        f"  kappa/kappa_e = {ratios[i]:4.2f}:  M/M_e  FE = {moments[i] / M_e:6.4f}  "
        f"exact = {exact[i]:6.4f}   ({errors[i]:4.2f} %)"
    )
print(f"  worst over the sweep: {errors.max():.2f} %")
print(
    f"  approaching the shape factor: M/M_e -> {moments[-1] / M_e:.4f} of the limiting 1.5"
)

# a verification fails rather than prints; the shape factor is a pure number, so the second
# bound is independent of material and size
assert errors.max() < 1.0, f"{errors.max():.2f} % away from Chakrabarty"
assert 1.4 < moments[-1] / M_e < 1.5, (
    "the section does not approach the shape factor 3/2"
)

# ----------------------------------------------
# Results
# ----------------------------------------------
PyVista.Plot_BoundaryConditions(simu).show()

# the two curves lie on top of each other, so the error gets a panel of its own
fig, (ax, axErr) = Matplotlib.plt.subplots(
    2, 1, sharex=True, figsize=(6.4, 6.0), gridspec_kw={"height_ratios": [3, 1]}
)

fine = np.linspace(0.05, ratios[-1], 400)
ax.axvspan(0, 1, color="0.93", zorder=0)
ax.text(0.5, 0.07, "elastic", ha="center", fontsize=8, color="0.4")
ax.plot(fine, Exact(fine), "k-", lw=1.2, label="Chakrabarty")
ax.plot(ratios, moments / M_e, "o", ms=6, mfc="none", mew=1.3, label="EasyFEA")
ax.axhline(1.5, ls="--", c="r", lw=0.9)
ax.text(
    ratios[-1], 1.505, "shape factor $M_p/M_e = 3/2$  ", c="r", fontsize=8, ha="right"
)
ax.set_ylabel("$M/M_e$")
ax.set_ylim(0, 1.62)
ax.set_xlim(0, ratios[-1] + 0.1)
ax.set_title("Moment-curvature of a rectangular section")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

axErr.axvspan(0, 1, color="0.93", zorder=0)
axErr.plot(ratios, errors, "o-", ms=4, lw=1)
axErr.set_xlabel(r"$\kappa/\kappa_e$")
axErr.set_ylabel("error [%]")
axErr.set_ylim(0, max(1.05 * errors.max(), 0.5))
axErr.grid(alpha=0.3)
fig.align_ylabels()

# the plastic front eating into the section
ax = Matplotlib.Init_Axes()
yFine = np.linspace(-h / 2, h / 2, 400)
for i, ratio in enumerate((0.8, 1.6, 2.4, 4.0)):
    idx = int(np.argmin(np.abs(ratios - ratio)))
    ax.plot(
        profiles[ratios[idx]], yMid, label=rf"$\kappa/\kappa_e$ = {ratios[idx]:.1f}"
    )
    # exact: linear inside the elastic core of half-depth c, saturated at sigma_y outside.
    # the imposed u_x = -kappa L y puts the top fibre in compression, hence the sign
    core = h / (2 * ratios[idx])
    ax.plot(
        -sigma_y * np.clip(yFine / core, -1, 1),
        yFine,
        "k--",
        lw=0.8,
        label="exact" if i == 0 else None,
    )
ax.axvline(sigma_y, ls=":", c="k", lw=0.8)
ax.axvline(-sigma_y, ls=":", c="k", lw=0.8)
ax.set_xlabel(r"$\sigma_{xx}$ [MPa]")
ax.set_ylabel("$y$ [mm]")
ax.set_title("Stress through the section: the elastic core shrinks")
ax.legend()
ax.grid(alpha=0.3)

PyVista.Movie_simu(simu, "p", folder, "p.gif", deformFactor=2, plotMesh=True)

Matplotlib.plt.show()
