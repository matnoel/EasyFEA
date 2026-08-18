# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _StressStrain:

StressStrain
============

Uniaxial curves for the shipped hardening laws, against their closed forms.

:class:`.MaterialPoint` drives one Gauss point with no mesh and no solver, holding the un-driven
components stress-free. Imposing :math:`\varepsilon_{xx}` alone is therefore uniaxial tension,
and every curve has an exact solution:

.. math::
    \sigma = E(\varepsilon - p) \quad\text{with}\quad \sigma = \sigma_y + R(p)

one scalar root per point. ``R`` is written out below from the definitions rather than taken from
the framework, so the agreement is a check and not a tautology.
"""

from enum import Enum

import numpy as np
from scipy.optimize import brentq

from EasyFEA import Matplotlib, Models
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
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
class Hardenings(str, Enum):
    Perfect = "perfect"
    Linear = f"linear, H = {H:.0f}"
    Voce = f"Voce, Q = {Q:.0f}, b = {b:.0f}"
    Swift = f"Swift, K = {K:.0f}, n = {n}"

    def __str__(self):
        return self.name


hardenings = {
    Hardenings.Perfect: (None, lambda p: 0.0),
    Hardenings.Linear: (Models.InElastic.IsotropicHardening.Linear(H), lambda p: H * p),
    Hardenings.Voce: (
        Models.InElastic.IsotropicHardening.Voce(Q, b),
        lambda p: Q * (1 - np.exp(-b * p)),
    ),
    Hardenings.Swift: (
        Models.InElastic.IsotropicHardening.Swift(K, n),
        lambda p: K * ((eps0 + p) ** n - eps0**n),
    ),
}

ax = Matplotlib.Init_Axes()
worst = 0.0
for i, (label, (hardening, R)) in enumerate(hardenings.items()):
    law = Models.InElastic.Behavior(
        3,
        elastic,
        hardening=hardening,
        yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
    )
    res = Models.InElastic.MaterialPoint(law).Run(strain={"xx": path})
    eps, sig = res["strain"][:, 0], res["stress"][:, 0]

    err = np.max(np.abs(sig - Exact(eps, R))) / sigma_y
    worst = max(worst, err)
    print(f"{label:28s} max |error| = {err:.2e} of sigma_y")

    ax.plot(eps * 100, sig, lw=1.4, label=label.value)
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


class Surfaces(str, Enum):
    VonMises = "von Mises"
    DruckerPrager = r"Drucker-Prager $\eta$ = 0.2"
    Hill = "Hill (anisotropic)"

    def __str__(self):
        return self.name


surfaces = {
    Surfaces.VonMises: Models.InElastic.Yield.VonMises(sigma_y),
    Surfaces.DruckerPrager: Models.InElastic.Yield.DruckerPrager(sigma_y, 0.2),
    Surfaces.Hill: Models.InElastic.Yield.Hill(sigma_y, F=F, G=G, H=Hh, L=Lh, M=M, N=N),
}

ax = Matplotlib.Init_Axes()
for label, surface in surfaces.items():
    law = Models.InElastic.Behavior(
        3,
        elastic,
        hardening=Models.InElastic.IsotropicHardening.Voce(Q, b),
        yieldSurface=surface,
    )
    res = Models.InElastic.MaterialPoint(law).Run(strain={"xx": path})
    ax.plot(res["strain"][:, 0] * 100, res["stress"][:, 0], label=label.value)

    if label is Surfaces.Hill:
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
