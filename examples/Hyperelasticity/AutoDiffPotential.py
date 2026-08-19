# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
.. _AutoDiffPotential:

AutoDiffPotential
=================

Write the strain energy. Get the stress and the tangent for free.

A hyperelastic law is a potential ``W(C)``: the second Piola-Kirchhoff stress is ``2 dW/dC``, the
tangent ``4 d2W/dC2``. Every shipped law is restated below as a one-point potential and compared
against its hand-written derivatives.

Needs ``jax`` (``pip install easyfea[jax]``); EasyFEA does not require it.

Reference: Holzapfel & Ogden, Phil. Trans. R. Soc. A 367 (2009).
"""

import numpy as np

from EasyFEA import Terminal, Models, ElemType, MatrixType
from EasyFEA.Models._autodiff import Enable_x64
from EasyFEA.Models.HyperElastic._state import HyperElasticState
from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Geoms import Domain, Point

try:
    import jax.numpy as jnp
except ModuleNotFoundError:
    raise Exception("jax must be installed!")

Enable_x64()


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# same names as ComputeHyperelasticLaws.py, the sympy source of the hand-written derivatives
def Invariants(C):
    I1 = jnp.trace(C)
    I2 = (I1**2 - jnp.trace(C @ C)) / 2
    I3 = jnp.linalg.det(C)
    return I1, I2, I3


def Reduced(I1, I2, I3):
    J1 = I1 * I3 ** (-1 / 3)
    J2 = I2 * I3 ** (-2 / 3)
    J3 = jnp.sqrt(I3)
    return J1, J2, J3


if __name__ == "__main__":
    Terminal.Clear()

    K, K1, K2 = 200.0, 30.0, 8.0  # MPa
    lmbda, mu = 121153.0, 80769.0

    # ----------------------------------------------
    # The five shipped laws, each as one function of physics
    # ----------------------------------------------
    def NeoHookean(C):
        J1, _, _ = Reduced(*Invariants(C))
        return K * (J1 - 3)

    def MooneyRivlin(C):
        J1, J2, J = Reduced(*Invariants(C))
        return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1) ** 2

    def CiarletGeymonat(C):
        J1, J2, J = Reduced(*Invariants(C))
        return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1 - jnp.log(J))

    def SaintVenantKirchhoff(C):
        I1, I2, I3 = Invariants(C)
        return (
            I1**2 * (lmbda / 8 + mu / 4)
            - I1 * (3 * lmbda / 4 + mu / 2)
            - I2 * mu / 2
            + 0.5 * K * (I3 - 1) ** 2
            + 9 * lmbda / 8
            + 3 * mu / 4
        )

    # Passive myocardium, Holzapfel & Ogden (2009), in Pa
    a, a_f, a_fs, a_s = 19.0, 6157.0, 72.0, 827.0
    b, b_f, b_fs, b_s = 8.023, 16.026, 11.436, 11.12
    HO = dict(
        C0=a / 2 / b, C1=b, C2=a_f / 2 / b_f, C3=b_f,
        C4=a_s / 2 / b_s, C5=b_s, C6=a_fs / 2 / b_fs, C7=b_fs,
        K=1e6, Mu1=0.0, Mu2=0.0, ks=100.0,
    )  # fmt: skip

    def HolzapfelOgden(C, T1, T2):
        """13 parameters, 6 invariants, a sigmoid switch."""
        J1, J2, J = Reduced(*Invariants(C))
        I4 = T1 @ C @ T1
        I6 = T2 @ C @ T2
        I8 = T1 @ C @ T2

        # the sigmoid: the fibre terms stiffen in tension and switch off in compression
        def chi(Ii):
            return 1 / (1 + jnp.exp(-HO["ks"] * (Ii - 1)))

        return (
            HO["C0"] * (jnp.exp(HO["C1"] * (J1 - 3)) - 1)
            + HO["C2"] * chi(I4) * (jnp.exp(HO["C3"] * (I4 - 1) ** 2) - 1)
            + HO["C4"] * chi(I6) * (jnp.exp(HO["C5"] * (I6 - 1) ** 2) - 1)
            + HO["C6"] * (jnp.exp(HO["C7"] * I8**2) - 1)
            + HO["K"] / 4 * (J**2 - 1 - 2 * jnp.log(J))
            + HO["Mu1"] * (J1 - 3)
            + HO["Mu2"] * (J2 - 3)
        )

    # ----------------------------------------------
    # A state to evaluate them on
    # ----------------------------------------------
    mesh = Domain(Point(0, 0, 0), Point(1, 1, 0), meshSize=0.5).Mesh_Extrude(
        [], [0, 0, 1], [2], ElemType.HEXA8
    )
    u = np.random.default_rng(0).normal(0.0, 0.02, mesh.Nn * 3)
    state = HyperElasticState(mesh.groupElem, u, MatrixType.rigi)

    # a fibre field, not a constant. FeArray, or a bare (Ne, nPg, 3) reads as one big tensor.
    Ne, nPg = mesh.Ne, mesh.groupElem.Get_gauss(MatrixType.rigi).nPg
    angle = np.linspace(-np.pi / 3, np.pi / 3, Ne)[:, None] * np.ones((1, nPg))
    zero = np.zeros_like(angle)
    T1 = FeArray.asfearray(np.stack([np.cos(angle), np.sin(angle), zero], axis=-1))
    T2 = FeArray.asfearray(np.stack([-np.sin(angle), np.cos(angle), zero], axis=-1))

    # ----------------------------------------------
    # Every law, autodiff vs hand-written
    # ----------------------------------------------
    LAWS = (
        (NeoHookean, Models.HyperElastic.NeoHookean(3, K=K), ()),
        (MooneyRivlin, Models.HyperElastic.MooneyRivlin(3, K1=K1, K2=K2, K=K), ()),
        (CiarletGeymonat, Models.HyperElastic.CiarletGeymonat(3, K1=K1, K2=K2, K=K), ()),
        (SaintVenantKirchhoff, Models.HyperElastic.SaintVenantKirchhoff(3, lmbda=lmbda, mu=mu, K=K), ()),  # fmt: skip
        (HolzapfelOgden, Models.HyperElastic.HolzapfelOgden(3, T1=T1, T2=T2, **HO), (T1, T2)),  # fmt: skip
    )

    print(f"{'law':24s}{'W':>12s}{'dWde':>12s}{'d2Wde':>12s}")
    for potential, shipped, aux in LAWS:
        W, dWde, d2Wde = Models.HyperElastic.HyperElasticPotential(potential)
        errors = (
            rel(W(state, *aux), shipped.Compute_W(state)),
            rel(dWde(state, *aux), shipped.Compute_dWde(state)),
            rel(d2Wde(state, *aux), shipped.Compute_d2Wde(state)),
        )
        print(f"{type(shipped).__name__:24s}" + "".join(f"{e:12.2e}" for e in errors))

    print(
        "\nHolzapfel-Ogden carries a fibre direction per Gauss point, as the cardiac mesh does."
    )
