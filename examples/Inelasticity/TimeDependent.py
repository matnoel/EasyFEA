# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
TimeDependent
=============

Two time-dependent mechanisms, three closed forms.

**Viscoelasticity** relaxes without ever yielding: Maxwell branches in parallel with a spring.
**Viscoplasticity** flows only past the surface, at a rate the overstress sets. One panel each:

1. *held strain, no yield surface* — the stress follows the Prony series
   :math:`\sigma = E\varepsilon_0[1 - \sum_i g_i(1 - e^{-t/\tau_i})]`;
2. *held strain, past yield* — the overstress bleeds off, so whatever the rate constant the
   stress ends on the rate-independent answer :math:`\sigma_y + H p`;
3. *held stress, past yield* — Norton with :math:`n = 1` and no hardening creeps at the
   constant rate :math:`A(\sigma - \sigma_y)/\sigma_0`.
"""

from enum import Enum

import numpy as np

from EasyFEA import Matplotlib, Models
from EasyFEA.Models.Elastic._laws import Isotropic

# ----------------------------------------------
# Configuration
# ----------------------------------------------
E, v = 210000.0, 0.3  # MPa
sigma_y = 250.0  # MPa
Hm = 2000.0  # MPa, isotropic hardening

elastic = Isotropic(3, E=E, v=v)
eps_y = sigma_y / E

# hold a fixed strain and let time pass; the first sample is one step in, not at t = 0
dt, nStep = 0.5, 60
eps_0 = 3.0 * eps_y
hold = np.full(nStep, eps_0)
time = (np.arange(nStep) + 1) * dt

# ----------------------------------------------
# Viscoelastic relaxation — no yield surface at all
# ----------------------------------------------


class Chains(str, Enum):
    OneBranch = r"one branch, $\tau$ = 5"
    TwoBranches = r"two branches, $\tau$ = 1 and 20"
    ThreeBranches = r"three branches, $\tau$ = 0.5, 5 and 50"

    def __str__(self):
        return self.name


# (g, tau) per branch; the fractions must sum below one, since 1 - sum(g) is the spring left
chains = {
    Chains.OneBranch: [(0.4, 5.0)],
    Chains.TwoBranches: [(0.25, 1.0), (0.25, 20.0)],
    Chains.ThreeBranches: [(0.2, 0.5), (0.2, 5.0), (0.2, 50.0)],
}

ax = Matplotlib.Init_Axes()
for label, branches in chains.items():
    law = Models.InElastic.Behavior(
        3,
        elastic,
        branches=[
            Models.InElastic.ViscoElastic.Maxwell(g=g, tau=t) for g, t in branches
        ],
    )
    sig = Models.InElastic.MaterialPoint(law).Run(strain={"xx": hold}, dt=dt)["stress"][
        :, 0
    ]
    ax.plot(time, sig, label=label.value)

    # what is left once every branch has relaxed: the lone spring
    equilibrium = E * eps_0 * (1 - sum(g for g, _ in branches))
    # backward Euler turns exp(-t/tau) into (1 + dt/tau)^-n, so this is exact, not approximate
    n = np.arange(nStep) + 1
    prony = equilibrium + E * eps_0 * sum(
        g * (1 + dt / tau) ** -n for g, tau in branches
    )
    ax.plot(time, prony, "k--", lw=0.8)

    err = np.max(np.abs(sig - prony))
    print(
        f"{label!s:14s} vs Prony {err:.1e} MPa; at t={time[-1]:.0f} "
        f"{sig[-1]:6.1f} of an eventual {equilibrium:6.1f}"
    )
    assert err < 1e-3, "the Maxwell chain does not follow its Prony series"

glassy = E * eps_0
ax.axhline(glassy, ls=":", c="k", lw=0.8)
ax.text(time[-1], glassy, r"glassy $E\epsilon$ ", ha="right", va="top")
ax.set_xlabel("time")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("Viscoelastic relaxation at held strain")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)

# ----------------------------------------------
# Viscoplastic relaxation — the overstress bleeds off onto the surface
# ----------------------------------------------
# whatever the rate, the overstress is gone in the end, so all three land on the
# rate-independent answer: sigma = sigma_y + Hm p with eps_0 = sigma/E + p
rateIndependent = (sigma_y + Hm * eps_0) / (1 + Hm / E)

ax = Matplotlib.Init_Axes()
for A in (1e-3, 1e-2, 1e-1):
    law = Models.InElastic.Behavior(
        3,
        elastic,
        hardening=Models.InElastic.IsotropicHardening.Linear(Hm),
        yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
        rate=Models.InElastic.ViscoPlastic.Norton(A, n=1.0, sigma_0=sigma_y),
    )
    sig = Models.InElastic.MaterialPoint(law).Run(strain={"xx": hold}, dt=dt)["stress"][
        :, 0
    ]
    ax.plot(time, sig, label=f"Norton, $A$ = {A:g}")
    print(
        f"Norton A = {A:<6g} ends at {sig[-1]:7.3f}, rate-independent {rateIndependent:7.3f}"
    )
    assert (
        abs(sig[-1] - rateIndependent) < 1e-6
    ), "relaxation missed the rate-independent limit"

ax.axhline(rateIndependent, ls=":", c="k", lw=0.8)
ax.text(time[-1], rateIndependent, "rate-independent limit ", ha="right", va="top")
ax.set_xlabel("time")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("Viscoplastic relaxation: same limit, reached at different rates")
ax.legend()
ax.grid(alpha=0.3)

# ----------------------------------------------
# Creep — the same law, holding the stress instead
# ----------------------------------------------
A = 1e-2
law = Models.InElastic.Behavior(
    3,
    elastic,
    yieldSurface=Models.InElastic.Yield.VonMises(sigma_y),
    rate=Models.InElastic.ViscoPlastic.Norton(A, n=1.0, sigma_0=sigma_y),
)

ax = Matplotlib.Init_Axes()
for held in (1.2 * sigma_y, 1.4 * sigma_y):
    # yz is pinned only because one component must be strain-driven; it carries no stress
    # here, so the state stays uniaxial. Pinning yy instead would constrain the lateral
    # contraction and make it biaxial, which changes the answer.
    res = Models.InElastic.MaterialPoint(law).Run(
        strain={"yz": np.zeros(nStep)},
        stress={"xx": np.full(nStep, held)},
        dt=dt,
    )
    eps = res["strain"][:, 0]
    ax.plot(time, eps * 100, label=rf"$\sigma_{{xx}}$ = {held:.0f} MPa")

    # no hardening, so the overstress never changes and the creep rate is constant
    exact = A * (held - sigma_y) / sigma_y
    rate = np.diff(eps) / dt
    print(f"creep at {held:5.0f} MPa: rate {rate.mean():.6e}, exact {exact:.6e}")
    assert (
        np.max(np.abs(rate - exact)) < 1e-12
    ), "creep is not stationary at the Norton rate"

ax.set_xlabel("time")
ax.set_ylabel("axial strain [%]")
ax.set_title("Creep at held stress: stationary, at the Norton rate")
ax.legend()
ax.grid(alpha=0.3)

print("\nRelaxation and creep are the same law seen at held strain and held stress.")

Matplotlib.plt.show()
