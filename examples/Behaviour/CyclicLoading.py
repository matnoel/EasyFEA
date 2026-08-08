# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _CyclicLoading:

CyclicLoading
=============

Why cyclic plasticity needs a back-stress.

Isotropic hardening grows the yield surface, so reversing the load finds it *harder* to yield in
compression. Real metals do the opposite — the Bauschinger effect — because the surface
translates rather than grows. That translation is a back-stress, and a back-stress is the
derivative of stored energy with respect to a tensor internal variable.

The two mechanisms are indistinguishable in monotonic tension and completely different under
reversal, which is the whole point of the figure. Both halves of that are checked below: the
monotonic branches agree to machine precision, and on reversal the kinematic elastic range is
still exactly :math:`2\sigma_y` while the isotropic one has grown.
"""

from enum import Enum

import numpy as np

from EasyFEA import Matplotlib, Models
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Material
# ----------------------------------------------
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa
C_kin = 20000.0  # MPa, kinematic modulus
gamma = 200.0  # ArmstrongFrederick recall

elastic = Isotropic(3, E=E, v=v)
eps_y = sigma_y / E

# two and a half cycles
peak = 6 * eps_y
quarter = np.linspace(0.0, peak, 30)
path = np.concatenate(
    [quarter]
    + [
        np.linspace(peak, -peak, 60)[1:],
        np.linspace(-peak, peak, 60)[1:],
    ]
    * 2
)


# ----------------------------------------------
# Same monotonic response, opposite reversal
# ----------------------------------------------
# a Prager back-stress hardens like an isotropic modulus of C under uniaxial tension
class Laws(str, Enum):
    Isotropic = "isotropic"
    Prager = "kinematic (Prager)"
    ArmstrongFrederick = "kinematic (ArmstrongFrederick)"

    def __str__(self):
        return self.name


laws = {
    Laws.Isotropic: Models.Behaviour(
        3,
        elastic,
        hardening=Models.IsotropicHardening.Linear(C_kin),
        yieldSurface=Models.Yield.VonMises(sigma_y),
    ),
    Laws.Prager: Models.Behaviour(
        3,
        elastic,
        yieldSurface=Models.Yield.VonMises(sigma_y),
        kinematic=Models.KinematicHardening.Prager(C_kin),
    ),
    Laws.ArmstrongFrederick: Models.Behaviour(
        3,
        elastic,
        yieldSurface=Models.Yield.VonMises(sigma_y),
        kinematic=Models.KinematicHardening.ArmstrongFrederick(C_kin, gamma=gamma),
    ),
}


def Elastic_span(res) -> float:
    """Stress drop from the first peak until plastic flow resumes, i.e. the elastic range."""
    top = len(quarter) - 1
    p = np.asarray(res["p"])
    resumed = top + int(np.argmax(np.diff(p[top:]) > 1e-12)) + 1
    return float(res["stress"][top, 0] - res["stress"][resumed, 0])


ax = Matplotlib.Init_Axes()
runs = {}
for label, law in laws.items():
    res = Models.MaterialPoint(law).Run(strain={"xx": path})
    runs[label] = res
    ax.plot(res["strain"][:, 0] * 100, res["stress"][:, 0], label=label, lw=1.2)

# monotonic: a Prager back-stress of modulus C hardens exactly like isotropic Linear(C)
n = len(quarter)

stressIsotropic = runs[Laws.Isotropic]["stress"][:n, 0]
stressPrager = runs[Laws.Prager]["stress"][:n, 0]
mono = np.max(np.abs(stressIsotropic - stressPrager))
print(f"monotonic branch, isotropic vs Prager: max |difference| = {mono:.2e} MPa")
assert mono < 1e-9, "the two mechanisms are meant to be identical in monotonic tension"

# on reversal they part: kinematic keeps an elastic range of 2 sigma_y, isotropic has grown one
step = 2 * peak / 119 * E  # one elastic stress increment, the resolution of the bracket
print("\nelastic range on the first reversal:")
for label, res in runs.items():
    print(f"  {label:32s} {Elastic_span(res):8.1f} MPa")
print(f"  {'exact, for kinematic':32s} {2 * sigma_y:8.1f} MPa")

for label in ("kinematic (Prager)", "kinematic (ArmstrongFrederick)"):
    span = Elastic_span(runs[label])
    assert 2 * sigma_y <= span <= 2 * sigma_y + step, (
        f"{label} lost the Bauschinger effect"
    )
assert Elastic_span(runs["isotropic"]) > 2 * sigma_y + step, "isotropic did not harden"

ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("Cyclic loading: isotropic grows the surface, kinematic moves it")
ax.legend()
ax.grid(alpha=0.3)

# ----------------------------------------------
# The back-stress that does it
# ----------------------------------------------
saturation = 2 * C_kin / (3 * gamma)  # X = 2/3 C alpha, and alpha stalls at 1/gamma

ax = Matplotlib.Init_Axes()
for label in (Laws.Prager, Laws.ArmstrongFrederick):
    X_xx = 2 / 3 * C_kin * runs[label]["alpha0"][:, 0]
    ax.plot(runs[label]["strain"][:, 0] * 100, X_xx, label=label)
    if "Armstrong" in label:
        print(
            f"\nback-stress reaches {X_xx.max() / saturation:.0%} of its {saturation:.1f} MPa saturation"
        )

ax.axhline(saturation, ls=":", c="k", lw=0.8)
ax.text(
    path[-1] * 20,
    saturation,
    r"ArmstrongFrederick saturation $2C/3\gamma$ ",
    ha="right",
    va="top",
)
ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$X_{xx}$ [MPa]")
ax.set_title(
    "The back-stress: Prager grows without bound, ArmstrongFrederick saturates"
)
ax.legend()
ax.grid(alpha=0.3)

print("Monotonic tension cannot tell these apart; reversal can.")

Matplotlib.plt.show()
