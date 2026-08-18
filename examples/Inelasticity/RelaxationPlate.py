# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
RelaxationPlate
===============

Viscoelastic relaxation on a mesh, with an exact check.

A perforated plate is stretched and then held. Nothing yields; the Maxwell branches bleed stress
off towards the equilibrium spring, so the load needed to hold the plate decays.

All branches carry a fraction of the same :math:`\Crm`, so the stress is the elastic field times
one scalar relaxation function,

.. math::
    \Sig(t) = R(t)\,\Crm : \Eps,
    \qquad
    R(t) = \Big(1 - \sum_i g_i\Big) + \sum_i g_i e^{-t/\tau_i}

and since :math:`\diver(R\,\Crm{:}\Eps) = 0`, the displacement field that solved equilibrium at
:math:`t=0` still solves it later. The whole field scales by one number, whatever the geometry.

Backward Euler replaces :math:`e^{-t/\tau}` by :math:`(1 + \dt/\tau)^{-n}`, so the check uses the
discrete form and holds to machine precision.
"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np

from EasyFEA import Matplotlib, ElemType, Models, Simulations
from EasyFEA.Geoms import Point, Points
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
L, h, r = 120.0, 60.0, 12.0  # mm
thickness = 5.0
E, v = 210000.0, 0.3  # MPa
stretch = 0.025  # mm, held constant — over the half-length, so 4.2e-4 average strain

branches = [
    Models.ViscoElastic.Maxwell(g=0.30, tau=1.0),
    Models.ViscoElastic.Maxwell(g=0.20, tau=10.0),
]
dt, nStep = 0.5, 40


def Relaxation(n: int) -> float:
    """R after n backward-Euler steps — exact for the scheme, unlike exp(-t/tau)."""
    g_eq = 1.0 - sum(br.g for br in branches)
    return g_eq + sum(br.g * (1 + dt / br.tau) ** -n for br in branches)


# ----------------------------------------------
# Model
# ----------------------------------------------
# the same quarter model as PlasticPlate: symmetry on both flats, pulled on the far edge
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

material = Models.Behaviour(
    2,
    Isotropic(3, E=E, v=v),
    branches=branches,  # no yield surface: this never flows, it only relaxes
    thickness=thickness,
    planeStress=True,
)
simu = Simulations.Behaviour(mesh, material)
simu.dt = dt

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L / 2)

# ----------------------------------------------
# Stretch, then hold
# ----------------------------------------------
time, peak, error = [], [], []
sig0 = None

for step in range(nStep):
    simu.Bc_Init()
    simu.add_dirichlet(nodesX0, [0], ["x"])
    simu.add_dirichlet(nodesY0, [0], ["y"])
    simu.add_dirichlet(nodesXL, [stretch], ["x"])
    simu.Solve()
    simu.Save_Iter()

    field = np.array([simu.Result(c, nodeValues=False) for c in ("Sxx", "Syy", "Sxy")])
    if sig0 is None:
        sig0 = field.copy()

    n = step + 1
    scale = Relaxation(n) / Relaxation(1)
    time.append(n * dt)
    peak.append(np.max(np.abs(field[0])))
    error.append(np.max(np.abs(field - scale * sig0)) / np.max(np.abs(sig0)))

print(f"held for {time[-1]:.0f} time units in {nStep} steps, {mesh.Ne} elements")
print(
    f"peak sigma_xx: {peak[0]:.1f} -> {peak[-1]:.1f} MPa "
    f"({100 * peak[-1] / peak[0]:.1f} % of the glassy value)"
)
print(f"max error against R(t): {max(error):.2e} — the field scales by one number")

# the discrete R(t) is exact for backward Euler, so the tolerance is machine precision
# rather than a discretisation error
assert max(error) < 1e-10, "the stress field does not relax by a single scalar"

# ----------------------------------------------
# Results
# ----------------------------------------------
ax = Matplotlib.Init_Axes()
n = np.arange(1, nStep + 1)
ax.plot(
    time, [Relaxation(k) / Relaxation(1) for k in n], "k-", lw=1, label="$R(t)$ exact"
)
ax.plot(time, np.array(peak) / peak[0], "o", ms=3, label="peak $\\sigma_{xx}$, FE")
g_eq = 1.0 - sum(br.g for br in branches)
ax.axhline(g_eq / Relaxation(1), ls=":", c="k", lw=0.8)
ax.text(0, g_eq / Relaxation(1), " equilibrium spring", fontsize=8, va="bottom")
ax.set_xlabel("time")
ax.set_ylabel("stress, normalised")
ax.set_title("Relaxation at held displacement")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# one colour scale for both: autoscaled they would look identical, since only the
# multiplying factor changes between them
clim = (0.0, float(np.max(simu.Result("Svm", iter=0))))
for it, t in ((0, time[0]), (-1, time[-1])):
    simu.Set_Iter(it)
    Matplotlib.Plot(
        simu,
        "Svm",
        ncolors=11,
        plotMesh=True,
        clim=clim,
        title=rf"von Mises at $t$ = {t:g} [MPa]",
    )

Matplotlib.plt.show()
