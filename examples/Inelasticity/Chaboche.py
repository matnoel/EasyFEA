# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""
.. _Chaboche:

Chaboche
========

Why one back-stress is not enough.

A single Armstrong-Frederick back-stress is one exponential, saturating at :math:`2C/3\gamma`
with a single rate. A measured hysteresis loop has both a sharp knee after yield and a long,
nearly linear tail, and no single exponential fits both.

Chaboche superposes several, :math:`X = \sum_i X_i`: a fast component for the knee, an
intermediate one for the curvature, and a linear one (:math:`\gamma = 0`) for the tail.

Reference
---------
J.-L. Chaboche, *Time-independent constitutive theories for cyclic plasticity*, Int. J.
Plasticity **2** (1986) 149--188.
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
elastic = Isotropic(3, E=E, v=v)
eps_y = sigma_y / E

KH = Models.KinematicHardening

# three components: fast knee, intermediate curvature, linear tail
components = [(60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0)]


class Laws(str, Enum):
    ArmstrongFrederick = "single Armstrong-Frederick"
    Chaboche = "Chaboche, 3 components"

    def __str__(self):
        return self.name


laws = {
    Laws.ArmstrongFrederick: KH.ArmstrongFrederick(*components[0]),
    Laws.Chaboche: KH.Chaboche(*components),
}


def Behaviour(kinematic):
    return Models.Behaviour(
        3,
        elastic,
        yieldSurface=Models.Yield.VonMises(sigma_y),
        kinematic=kinematic,
    )


# ----------------------------------------------
# The loop shape: knee and tail
# ----------------------------------------------
peak = 8 * eps_y
quarter = np.linspace(0.0, peak, 40)
path = np.concatenate(
    [
        quarter,
        np.linspace(peak, -peak, 80)[1:],
        np.linspace(-peak, peak, 80)[1:],
    ]
)

ax = Matplotlib.Init_Axes()
for label, kinematic in laws.items():
    res = Models.MaterialPoint(Behaviour(kinematic)).Run(strain={"xx": path})
    ax.plot(res["strain"][:, 0] * 100, res["stress"][:, 0], lw=1.2, label=label)

# a superposition of one term is that term: the machinery must add nothing of its own
C0, g0 = components[0]
one = Models.MaterialPoint(Behaviour(KH.Chaboche((C0, g0)))).Run(strain={"xx": path})
alone = Models.MaterialPoint(Behaviour(KH.ArmstrongFrederick(C0, g0))).Run(
    strain={"xx": path}
)
same = np.max(np.abs(one["stress"][:, 0] - alone["stress"][:, 0]))
print(f"Chaboche with one component vs ArmstrongFrederick: {same:.1e} MPa")
assert same == 0.0, "the superposition is not exact for a single component"

ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$\sigma_{xx}$ [MPa]")
ax.set_title("One exponential cannot follow both the knee and the tail")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ----------------------------------------------
# The components that make it up
# ----------------------------------------------
behaviour = Behaviour(KH.Chaboche(*components))
res = Models.MaterialPoint(behaviour).Run(strain={"xx": path})

ax = Matplotlib.Init_Axes()
total = np.zeros_like(res["strain"][:, 0])
for i, (C, gamma) in enumerate(components):
    X_xx = 2 / 3 * C * res[f"alpha{i}"][:, 0]
    total = total + X_xx
    ax.plot(
        res["strain"][:, 0] * 100,
        X_xx,
        lw=1,
        label=rf"$X_{i + 1}$: $C$={C:.0f}, $\gamma$={gamma:.0f}",
    )
    if gamma > 0:
        # the recall term bounds each component; the fast one all but reaches its bound
        bound = 2 * C / (3 * gamma)
        print(
            f"  X{i + 1} peaks at {np.abs(X_xx).max():7.2f} of its {bound:7.2f} bound"
        )
        assert np.abs(X_xx).max() <= bound * (1 + 1e-9), f"X{i + 1} passed 2C/3gamma"
    else:
        # with no recall alpha is just the plastic strain, so this component is linear in it
        linear = np.max(np.abs(X_xx - 2 / 3 * C * res["eps_p"][:, 0]))
        print(f"  X{i + 1} is linear in the plastic strain to {linear:.2e} MPa")
        assert linear < 1e-9, "the gamma = 0 component is not linear"

ax.plot(res["strain"][:, 0] * 100, total, "k-", lw=1.4, label=r"$X = \sum_i X_i$")

# the yield condition, read off the peak: the axial equivalent of X is 3/2 its xx component
k = int(np.argmax(np.abs(res["stress"][:, 0])))
print(
    f"  at the peak, |sigma| = {abs(res['stress'][k, 0]):.2f} "
    f"= sigma_y + 3/2 |X| = {sigma_y + 1.5 * abs(total[k]):.2f}"
)
assert abs(abs(res["stress"][k, 0]) - sigma_y - 1.5 * abs(total[k])) < 1e-6
ax.set_xlabel("axial strain [%]")
ax.set_ylabel(r"$X_{xx}$ [MPa]")
ax.set_title(r"The superposition: fast, intermediate and linear ($\gamma$ = 0)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

Matplotlib.plt.show()
