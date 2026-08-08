# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _ThickCylinder:

ThickCylinder
=============

A verification, not a picture.

A thick-walled cylinder under internal pressure is *the* elastic-plastic benchmark: the plastic
zone spreads from the bore outwards, and everything about it is known in closed form. Every other
example here produces a plausible result whether or not the physics is right; this one puts three
independent analytical landmarks on a single problem, and fails if any of them moves.

For an elastic-perfectly-plastic material in plane strain the von Mises condition reduces to
:math:`\sigma_\theta - \sigma_r = Y` with :math:`Y = 2\sigma_y/\sqrt3`. Equilibrium then gives the
pressure at which the elastic-plastic boundary sits at radius ``c``:

.. math::
    p(c) = \frac{Y}{2}\left[\,2\ln\frac{c}{a} + 1 - \frac{c^2}{b^2}\right]

**1. Below** :math:`p_e = \frac{Y}{2}(1 - a^2/b^2)` the wall is elastic and Lamé's solution holds
exactly, so the bore displacement must be linear in the pressure with a known slope.

**2. Between** the two the wall is partly plastic and the stresses follow from equilibrium and the
yield condition alone — no elastic constants enter the plastic zone, which is why that check is
insensitive to Poisson's ratio. The mesh study is the real evidence: the error must *fall* with
refinement, and at the right rate. It converges first order — halving the element size halves the
error — even with quadratic elements, because the elastic-plastic boundary is a kink in the stress
field that no mesh resolves exactly. Second-order convergence would mean the front was being
missed.

**3. At** :math:`p_{lim} = Y\ln(b/a)` the whole wall is plastic and the cylinder collapses: the
displacement runs away while the pressure cannot rise. Pushing to :math:`0.995\,p_{lim}` under
pressure control is also the hardest test of the return mapping, since a perfectly plastic
structure has no stiffness left to help the global Newton.

References
----------
Hill, *The Mathematical Theory of Plasticity*, Oxford (1950), ch. V.

Bleyer, `Elasto-plastic analysis of a 2D von Mises material
<https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html>`_,
*Computational Mechanics Numerical Tours with FEniCSx* — the same cylinder, and the
load-displacement presentation used below. It hardens (:math:`H = E/100`) where this one is
perfectly plastic, so its limit load is approached rather than exact.
"""

import numpy as np
from scipy.optimize import brentq

from EasyFEA import Folder, ElemType, Models, Simulations, PyVista, Matplotlib
from EasyFEA.Geoms import CircleArc, Contour, Line, Point, Circle
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
print(f"applied      p        = {pressure:7.2f} MPa -> plastic front at c = {c:.2f} mm")


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
# Quarter model
# ----------------------------------------------
def Mesh(nThroughWall: int):
    """Quarter annulus, meshed to nThroughWall elements across the thickness."""
    p1, p2, p3, p4 = Point(a, 0), Point(b, 0), Point(0, b), Point(0, a)
    origin = Point(0, 0)
    h = (b - a) / nThroughWall
    contour = Contour(
        [
            Line(p1, p2, h),
            CircleArc(p2, p3, center=origin, meshSize=h),
            Line(p3, p4, h),
            CircleArc(p4, p1, center=origin, meshSize=h),
        ]
    )
    return contour.Mesh_2D([], ElemType.TRI6)


def Simulation(mesh):
    """Perfectly plastic, as Hill assumes."""
    material = Models.Behaviour(
        2,
        Isotropic(3, E=E, v=v),
        yieldSurface=Models.Yield.VonMises(sigma_y),
        planeStress=False,
    )
    return Simulations.Behaviour(mesh, material)


def Ramp(simu: Simulations.Behaviour, pressures: np.ndarray):
    """Applies the pressures in order; plasticity is path dependent so they cannot be skipped."""
    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    bore = mesh.Nodes_Circle(Circle((0, 0), diam=2 * a))

    for p in pressures:
        simu.Bc_Init()
        simu.add_dirichlet(nodesY0, [0], ["y"])
        simu.add_dirichlet(nodesX0, [0], ["x"])
        simu.add_pressureLoad(bore, p)
        simu.Solve()
        simu.Save_Iter()
        yield p


# ----------------------------------------------
# 2. Stresses through a partly plastic wall
# ----------------------------------------------
print("\nmesh convergence at p = 160 MPa (max error along y = 0, vs Hill):")
runs, errors = {}, []
for n in (8, 16, 32):
    mesh = Mesh(n)
    simu = Simulation(mesh)
    list(Ramp(simu, np.linspace(pressure / 8, pressure, 8)))

    # sample along y = 0, where the radial direction is x, so sig_r = Sxx and sig_t = Syy
    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    order = np.argsort(mesh.coord[nodesY0, 0])
    r = mesh.coord[nodesY0, 0][order]
    sig_r = simu.Result("Sxx")[nodesY0][order]
    sig_t = simu.Result("Syy")[nodesY0][order]

    exact_r, exact_t = Exact(r)
    err = max(np.max(np.abs(sig_r - exact_r)), np.max(np.abs(sig_t - exact_t)))
    runs[n] = (simu, mesh, r, sig_r, sig_t)
    errors.append(err)
    print(
        f"  {n:2d} elements through the wall ({mesh.Ne:5d} total): "
        f"max |error| = {err:5.2f} MPa = {100 * err / Y:4.2f} % of Y"
    )

# ----------------------------------------------
# 1 & 3. Elastic slope, then collapse
# ----------------------------------------------
# sqrt spacing: the increments shorten as the limit load is approached, where they must
steps = p_lim * 0.995 * np.linspace(0, 1, 26)[1:] ** 0.5

mesh = Mesh(16)
simu = Simulation(mesh)
node_a = mesh.Nodes_Conditions(lambda x, y, z: (y == 0) & (x <= a * 1.001))

applied, u_bore = [], []
for p in Ramp(simu, steps):
    applied.append(p)
    u_bore.append(float(np.mean(simu.Result("ux")[node_a])))
applied, u_bore = np.array(applied), np.array(u_bore)

ratio = u_bore / Elastic_u(applied)
elastic = applied < p_e
print(f"\nbelow p_e : max |u/u_Lame - 1| = {np.max(np.abs(ratio[elastic] - 1)):.2e}")
print(
    f"at {applied[-1] / p_lim:.3f} p_lim: u is {ratio[-1]:.2f} x the elastic extrapolation"
)

# ----------------------------------------------
# What must hold
# ----------------------------------------------
# a verification fails rather than prints; bounds loose enough to survive a different mesher,
# tight enough that losing the front, the elastic slope or the limit load trips them
assert errors[-1] < 0.02 * Y, f"{errors[-1]:.2f} MPa is too far from Hill's solution"
assert errors[0] / errors[-1] > 3, "convergence is slower than first order"
assert np.max(np.abs(ratio[elastic] - 1)) < 5e-3, "the elastic branch is not Lamé"
assert ratio[-1] > 1.8, "the cylinder does not collapse at Hill's limit load"

# ----------------------------------------------
# Results
# ----------------------------------------------
simu, _, r, sig_r, sig_t = runs[32]
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
ax.plot(u_bore, applied / p_lim, "o-", ms=3, lw=1, label="FE")
# the same pressures against the displacement Lamé predicts: the FE curve leaves it at p_e
ax.plot(Elastic_u(applied), applied / p_lim, "k--", lw=1, label="Lamé (elastic)")
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

PyVista.Movie_simu(simu, "p", folder, "p.gif", plotMesh=True, deformFactor=10)

Matplotlib.plt.show()
