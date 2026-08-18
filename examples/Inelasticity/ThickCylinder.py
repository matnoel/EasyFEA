# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
ThickCylinder
=============

The classic elastic-plastic benchmark: a thick-walled cylinder under internal pressure, with
three independent analytical landmarks on one problem.

For an elastic-perfectly-plastic material in plane strain the von Mises condition reduces to
:math:`\sigma_\theta - \sigma_r = Y` with :math:`Y = 2\sigma_y/\sqrt3`, and equilibrium gives the
pressure putting the elastic-plastic boundary at radius ``c``:

.. math::
    p(c) = \frac{Y}{2}\left[\,2\ln\frac{c}{a} + 1 - \frac{c^2}{b^2}\right]

Below :math:`p_e = \frac{Y}{2}(1 - a^2/b^2)` the wall is elastic and Lamé holds. Between the two
the stresses follow from equilibrium and the yield condition alone, so no elastic constant enters
the plastic zone. At :math:`p_{lim} = Y\ln(b/a)` the wall is fully plastic and the cylinder
collapses.

The elastic-plastic boundary is a kink that no mesh resolves exactly, so the error converges
only first order even with quadratic elements.

References
----------
Hill, *The Mathematical Theory of Plasticity*, Oxford (1950), ch. V.

Bleyer, `Elasto-plastic analysis of a 2D von Mises material
<https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html>`_,
*Computational Mechanics Numerical Tours with FEniCSx* — the same cylinder, hardening where this
one is perfectly plastic.
"""

# sphinx_gallery_thumbnail_number = 3

import numpy as np
from scipy.optimize import brentq

from EasyFEA import Folder, ElemType, Models, Simulations, PyVista, Matplotlib
from EasyFEA.Geoms import CircleArc, Contour, Line, Circle
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
folder = Folder.Results_Dir()

a, b = 100.0, 200.0  # mm, inner and outer radius
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa

Y = 2 * sigma_y / np.sqrt(3)  # plane-strain yield in sigma_theta - sigma_r
p_e = Y / 2 * (1 - a**2 / b**2)  # bore starts to yield
p_lim = Y * np.log(b / a)  # whole wall plastic: collapse
pressure = 160.0  # MPa, between the two so the front sits inside the wall

c = brentq(lambda c: Y / 2 * (2 * np.log(c / a) + 1 - c**2 / b**2) - pressure, a, b)

print(f"yield starts at p_e   = {p_e:7.2f} MPa")
print(f"fully plastic at p_lim= {p_lim:7.2f} MPa")
print(f"applied      p      = {pressure:7.2f} MPa -> plastic front at c = {c:.2f} mm")


def Exact(r):
    """Radial and hoop stress, Hill 1950 ch. V."""
    plastic = r <= c
    sig_r = np.where(
        plastic,
        -pressure + Y * np.log(r / a),
        Y * c**2 / (2 * b**2) * (1 - b**2 / r**2),
    )
    sig_t = np.where(plastic, sig_r + Y, Y * c**2 / (2 * b**2) * (1 + b**2 / r**2))
    return sig_r, sig_t


def Elastic_u(p):
    """Bore displacement, Lamé in plane strain."""
    return (1 + v) * a * p / (E * (b**2 - a**2)) * ((1 - 2 * v) * a**2 + b**2)


# ----------------------------------------------
# Mesh
# ----------------------------------------------
origin = (0, 0)
p1 = (a, 0)
p2 = (b, 0)
p3 = (0, b)
p4 = (0, a)
meshSize = (b - a) / 16  # 16 elements through the wall
contour = Contour(
    [
        Line(p1, p2, meshSize),
        CircleArc(p2, p3, center=origin, meshSize=meshSize),
        Line(p3, p4, meshSize),
        CircleArc(p4, p1, center=origin, meshSize=meshSize),
    ]
)
mesh = contour.Mesh_2D([], ElemType.TRI6)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
bore = mesh.Nodes_Circle(Circle((0, 0), diam=2 * a))

# perfectly plastic, as Hill assumes
material = Models.InElastic.Behavior(
    2,
    Isotropic(3, E=E, v=v),
    yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
    planeStress=False,
)


# 1 & 3. One ramp, from first yield to collapse
# ----------------------------------------------
# sqrt spacing: the increments shorten as p_lim is approached, where they must. `pressure` is
# inserted rather than jumped to, since plasticity is path dependent and the wall is read there.
steps = np.sort(np.append(p_lim * 0.998 * np.linspace(0, 1, 26)[1:] ** 0.5, pressure))
iPressure = int(np.flatnonzero(steps == pressure)[0])

simu = Simulations.InElastic(mesh, material)
node_a = mesh.Nodes_Conditions(lambda x, y, z: (y == 0) & (x <= a * 1.001))

u_bore = []
for p in steps:
    simu.Bc_Init()
    simu.add_dirichlet(nodesY0, [0], ["y"])
    simu.add_dirichlet(nodesX0, [0], ["x"])
    simu.add_pressureLoad(bore, p)
    simu.Solve()
    simu.Save_Iter()
    u_bore.append(float(np.mean(simu.Result("ux")[node_a])))
u_bore = np.array(u_bore)

# ----------------------------------------------
# 2. Stresses through a partly plastic wall
# ----------------------------------------------
simu.Set_Iter(iPressure)

# sample along y = 0, where the radial direction is x, so sig_r = Sxx and sig_t = Syy
order = np.argsort(mesh.coord[nodesY0, 0])
r = mesh.coord[nodesY0, 0][order]
sig_r = simu.Result("Sxx")[nodesY0][order]
sig_t = simu.Result("Syy")[nodesY0][order]

exact_r, exact_t = Exact(r)
err = max(np.max(np.abs(sig_r - exact_r)), np.max(np.abs(sig_t - exact_t)))
print(
    f"\nat p = {pressure:.0f} MPa on {mesh.Ne} elements: "
    f"max |error| vs Hill = {err:.2f} MPa = {100 * err / Y:.2f} % of Y"
)
# the error above is a discretisation error, so no fixed bound on it says anything about the
# physics. What does: Hill puts the front at c, and a mesh of element size h cannot place it
# any closer than that.
p_r = simu.Result("p")[nodesY0][order]
front = r[p_r > 0].max()
print(
    f"plastic front at r = {front:.1f} mm, Hill says {c:.1f}, element size {meshSize:.1f}"
)
assert abs(front - c) < meshSize, "the plastic front is not where Hill puts it"

# ----------------------------------------------
# Results
# ----------------------------------------------
rr = np.linspace(a, b, 400)
exact_r, exact_t = Exact(rr)

PyVista.Plot_BoundaryConditions(simu).show()

ax = Matplotlib.Init_Axes()
ax.plot(rr / a, exact_t, "k-", lw=1, label=r"$\sigma_\theta$ exact")
ax.plot(rr / a, exact_r, "k--", lw=1, label=r"$\sigma_r$ exact")
ax.plot(r / a, sig_t, "o", ms=3, label=r"$\sigma_\theta$ FE")
ax.plot(r / a, sig_r, "s", ms=3, label=r"$\sigma_r$ FE")
ax.axvline(c / a, ls=":", c="k", lw=0.8)
ax.text(c / a, 0, "  plastic front", rotation=90, va="bottom")
ax.set_xlabel("$r/a$")
ax.set_ylabel("stress [MPa]")
ax.set_title(f"Thick cylinder at $p$ = {pressure:.0f} MPa")
ax.legend()
ax.grid(alpha=0.3)

ax = Matplotlib.Init_Axes()
ax.plot(u_bore, steps / p_lim, "o-", ms=3, lw=1, label="FE")
# the same pressures against the displacement Lamé predicts: the FE curve leaves it at p_e
ax.plot(Elastic_u(steps), steps / p_lim, "k--", lw=1, label="Lamé (elastic)")
ax.axhline(1.0, c="r", ls="--", lw=1)
ax.text(u_bore[0], 1.0, "$p_{lim} = Y \, \\ln(b/a)$", c="r", va="bottom")
ax.axhline(p_e / p_lim, c="k", ls=":", lw=0.8)
ax.text(u_bore[0], p_e / p_lim, "$p_e$", va="bottom")
ax.set_xlabel("radial displacement at the bore [mm]")
ax.set_ylabel("$p / p_{lim}$")
ax.set_title("Load-displacement to collapse")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(alpha=0.3)

PyVista.Plot(simu, "Svm", plotMesh=True, nColors=11).show()

PyVista.Movie_simu(simu, "p", folder, "p.gif", deformFactor=10)

Matplotlib.plt.show()
