# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Models.InElastic.IsotropicHardening import Linear, Perfect, Swift, Voce

pytest.importorskip("jax")

P = np.array([0.0, 1e-3, 1e-2, 1.0])


def test_Perfect_never_hardens():
    assert np.array_equal(Perfect()(P), np.zeros_like(P))


def test_Linear_is_H_p():
    assert np.allclose(Linear(2000.0)(P), 2000.0 * P)


def test_Linear_rejects_a_negative_slope():
    with pytest.raises(AssertionError, match="H must be >= 0"):
        Linear(-1.0)


def test_Voce_saturates_at_Q():
    R = np.asarray(Voce(150.0, 30.0)(P))

    assert np.allclose(R, 150.0 * (1 - np.exp(-30.0 * P)))
    assert np.isclose(Voce(150.0, 30.0)(1e3), 150.0)


def test_Voce_rejects_bad_parameters(subtests):
    for Q, b in [(-1.0, 1.0), (1.0, 0.0)]:
        with subtests.test(Q=Q, b=b):
            with pytest.raises(AssertionError, match="Q must be >= 0 and b > 0"):
                Voce(Q, b)


def test_Swift_is_zero_at_the_origin_and_power_law_after():
    K, n, eps0 = 600.0, 0.2, 1e-4

    assert np.allclose(Swift(K, n)(P), K * ((eps0 + P) ** n - eps0**n))
    assert Swift(K, n)(0.0) == 0.0


def test_Swift_rejects_bad_parameters(subtests):
    for K, n, eps0 in [(0.0, 0.2, 1e-4), (1.0, 1.0, 1e-4), (1.0, 0.2, 0.0)]:
        with subtests.test(K=K, n=n, eps0=eps0):
            with pytest.raises(AssertionError, match="need K > 0"):
                Swift(K, n, eps0)
