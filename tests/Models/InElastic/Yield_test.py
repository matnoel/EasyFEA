# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Models.InElastic.Yield import DruckerPrager, Hill, VonMises

pytest.importorskip("jax")

SIGMA_Y = 250.0
SIG = np.random.default_rng(0).normal(0.0, 200.0, 6)
UNIAXIAL = np.array([1.0, 0, 0, 0, 0, 0])
HILL = Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4)


SURFACES = {
    "VonMises": (VonMises, ()),
    "Hill": (Hill, ()),
    "DruckerPrager": (DruckerPrager, (0.2,)),
}
"""Surface subclass and its arguments after sigma_y."""


def test_Surface_is_minus_sigma_y_minus_R_at_zero_stress(subtests):
    for name, (surface, args) in SURFACES.items():
        with subtests.test(surface=name):
            f = surface(SIGMA_Y, *args)
            assert np.isclose(f(np.zeros(6), 10.0), -SIGMA_Y - 10.0)


def test_Surface_rejects_a_non_positive_sigma_y(subtests):
    for name, (surface, args) in SURFACES.items():
        with subtests.test(surface=name):
            with pytest.raises(AssertionError, match="sigma_y must be > 0"):
                surface(0.0, *args)


def test_VonMises_yields_uniaxially_at_sigma_y():
    assert np.isclose(VonMises(SIGMA_Y)(1.2 * SIGMA_Y * UNIAXIAL, 0.0), 0.2 * SIGMA_Y)


def test_Hill_defaults_to_von_mises():
    assert np.isclose(Hill(SIGMA_Y)(SIG, 10.0), VonMises(SIGMA_Y)(SIG, 10.0))


def test_Hill_yields_uniaxially_at_sigma_y_over_sqrt_g_plus_h():
    sig = 1.2 * SIGMA_Y / np.sqrt(0.4 + 0.6) * UNIAXIAL

    assert np.isclose(HILL(sig, 0.0), 0.2 * SIGMA_Y)


def test_Hill_P_is_symmetric_and_deviatoric():
    P = HILL.P

    assert np.array_equal(P, P.T)
    assert np.allclose(P @ [1, 1, 1, 0, 0, 0], 0.0)
    assert np.array_equal(np.diag(P)[3:], [1.8, 1.2, 1.4])


def test_DruckerPrager_shifts_von_mises_by_eta_trace():
    eta = 0.2

    assert np.isclose(
        DruckerPrager(SIGMA_Y, eta)(SIG, 10.0),
        VonMises(SIGMA_Y)(SIG, 10.0) + eta * SIG[:3].sum(),
    )
