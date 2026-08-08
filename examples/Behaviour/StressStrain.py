# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _StressStrain:

StressStrain
============

Uniaxial curves for the shipped hardening laws, against their closed forms.

A behaviour is assembled from independent pieces: an elastic law, a yield surface, and a
hardening law. Because hardening lives in the free energy rather than inside the surface, any
hardening composes with any surface — three hardening laws and two surfaces are five objects,
not six.

:class:`.MaterialPoint` drives one Gauss point with no mesh and no solver, but through the same
code path assembly uses, so this is the real behaviour rather than a second implementation of it.
It leaves the un-driven components **stress**-free, so imposing :math:`\varepsilon_{xx}` alone is
uniaxial tension and every curve here has an exact solution:

.. math::
    \sigma = E(\varepsilon - p) \quad\text{with}\quad \sigma = \sigma_y + R(p)

one scalar root per point. ``R`` is written out below from the definitions rather than taken from
the framework, so agreement is a check and not a tautology. Expect machine precision: this is the
constitutive law on its own, with no discretisation anywhere.
"""

import numpy as np
from scipy.optimize import brentq

from EasyFEA import Matplotlib, Models
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Material
# ----------------------------------------------
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa

elastic = Isotropic(3, E=E, v=v)
eps_y = sigma_y / E

path = np.linspace(0.0, 30 * eps_y, 200)

H = 2000.0  # linear
Q, b = 150.0, 30.0  # Voce
K, n, eps0 = 600.0, 0.2, 1e-4  # Swift, with the default pre-strain


def Exact(eps: np.ndarray, R) -> np.ndarray:
    """Uniaxial tension: sigma = E (eps - p), with p from sigma = sigma_y + R(p)."""
    out = np.where(E * eps <= sigma_y, E * eps, 0.0)
    for i, e in enumerate(eps):
        if E * e > sigma_y:
            # xtol tightened: at the default, brentq's own root error times E is the floor
            p = brentq(lambda p: E * (e - p) - sigma_y - R(p), 0.0, e, xtol=1e-15)
            out[i] = E * (e - p)
    return out


# ----------------------------------------------
# One curve per hardening law
# ----------------------------------------------
hardenings = {
    "perfect": (None, lambda p: 0.0),
    "linear, H = 2000": (Models.IsotropicHardening.Linear(H), lambda p: H * p),
    "Voce, Q = 150, b = 30": (
        Models.IsotropicHardening.Voce(Q, b),
        lambda p: Q * (1 - np.exp(-b * p)),
    ),
    "Swift, K = 600, n = 0.2": (
        Models.IsotropicHardening.Swift(K, n),
        lambda p: K * ((eps0 + p) ** n - eps0**n),
    ),
}

ax = Matplotlib.Init_Axes()
worst = 0.0
for i, (label, (hardening, R)) in enumerate(hardenings.items()):
    law = Models.Behaviour(
        3,
        elastic,
        hardening=hardening,
        yieldSurface=Models.Yield.VonMises(sigma_y),
    )
    res = Models.MaterialPoint(law).Run(strain={"xx": path})
    eps, sig = res["strain"][:, 0], res["stress"][:, 0]

    err = np.max(np.abs(sig - Exact(eps, R))) / sigma_y
    worst = max(worst, err)
    print(f"{label:28s} max |error| = {err:.2e} of sigma_y")

    ax.plot(eps * 100, sig, lw=1.4, label=label)
    ax.plot(
        eps * 100,
        Exact(eps, R),
        "k--",
        lw=0.8,
        label="closed form" if i == 0 else None,
    )

print(f"\nworst over every law: {worst:.2e} of sigma_y")
assert worst < 1e-10, "a hardening law does not reproduce its own closed form"

ax.axhline(sigma_y, ls=":", c="k", lw=0.8)
ax.text(path[-1] * 100, sigma_y, r"$\sigma_y$ ", ha="right", va="top")
ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("Isotropic hardening laws, uniaxial tension")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

# ----------------------------------------------
# The same hardening on a different surface
# ----------------------------------------------
# G + H must not be 1, or Hill reduces to von Mises along x and the two curves coincide
F, G, Hh, Lh, M, N = 0.7, 0.6, 0.9, 1.8, 1.2, 1.4
hill_y = sigma_y / np.sqrt(G + Hh)
surfaces = {
    "von Mises": Models.Yield.VonMises(sigma_y),
    r"Drucker-Prager $\eta$ = 0.2": Models.Yield.DruckerPrager(sigma_y, 0.2),
    "Hill (anisotropic)": Models.Yield.Hill(sigma_y, F=F, G=G, H=Hh, L=Lh, M=M, N=N),
}

ax = Matplotlib.Init_Axes()
for label, surface in surfaces.items():
    law = Models.Behaviour(
        3,
        elastic,
        hardening=Models.IsotropicHardening.Voce(Q, b),
        yieldSurface=surface,
    )
    res = Models.MaterialPoint(law).Run(strain={"xx": path})
    ax.plot(res["strain"][:, 0] * 100, res["stress"][:, 0], label=label)

    if surface is surfaces["Hill (anisotropic)"]:
        # uniaxially Hill reduces to sigma_xx sqrt(G + H), so first yield is bracketed
        onset = int(np.argmax(np.asarray(res["p"]) > 0))
        sig = res["stress"][:, 0]
        print(
            f"\nHill yields uniaxially at {hill_y:.2f} MPa, in [{sig[onset - 1]:.2f}, {sig[onset]:.2f}]"
        )
        assert sig[onset - 1] <= hill_y <= sig[onset], "Hill's uniaxial yield is wrong"

# annotated at the right edge, where these two levels sit below every curve
right = path[-1] * 100
ax.axhline(hill_y, ls=":", c="k", lw=0.8)
ax.text(right, hill_y, r"Hill: $\sigma_y/\sqrt{G+H}$ ", ha="right", va="top")
ax.axhline(sigma_y, ls=":", c="k", lw=0.8)
ax.text(right, sigma_y, r"von Mises: $\sigma_y$ ", ha="right", va="top")
ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("Voce hardening on three different surfaces")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

print("\nEvery curve above is the same engine: only the pieces handed to it differ.")

Matplotlib.plt.show()
