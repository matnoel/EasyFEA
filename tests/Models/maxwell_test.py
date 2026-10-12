# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Maxwell, the first behavior on the contract, against its closed forms. Skipped whole when jax is absent."""

import numpy as np
import pytest

from EasyFEA.Models import _autodiff
from EasyFEA.Models.Elastic._laws import Isotropic
from EasyFEA.Models.InElastic import MaterialPoint, Maxwell

from .conftest import assert_consistent_tangent

pytest.importorskip("jax")
_autodiff.Enable_x64()

E, nu = 210000.0, 0.3
EPS = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
ELASTIC = Isotropic(3, E=E, v=nu)
C = ELASTIC.C
G, TAU = [0.3, 0.2], [1.0, 10.0]


def _hold(
    behavior: Maxwell,
    eps,
    nstep: int,
    dt: float,
) -> np.ndarray:
    point, z = MaterialPoint(behavior), None
    for _ in range(nstep):
        sig, _, z = point.Update(eps, z, dt)
    return sig


def test_state_is_sized_by_the_branches():
    behavior = Maxwell(ELASTIC, G, TAU)

    assert behavior.Virgin_internals().eps_v.shape == (2, 6)
    assert behavior.Virgin_internals_e_pg(5, 4)["eps_v"].shape == (5, 4, 2, 6)
    # a (2, 6) default must not be taken for an (Ne, nPg) field
    assert behavior.Virgin_internals_e_pg(2, 6)["eps_v"].shape == (2, 6, 2, 6)


def test_fractions_must_leave_an_equilibrium_spring():
    with pytest.raises(AssertionError, match="sum to less than 1"):
        Maxwell(ELASTIC, [0.6, 0.4], [1.0, 2.0])


def test_a_field_is_not_taken_for_branches():
    with pytest.raises(AssertionError, match="one g and one tau per branch"):
        Maxwell(ELASTIC, np.full((5, 4), 0.01), np.ones((5, 4)))


def test_glassy_response_is_the_full_stiffness():
    """dt = 0: the dashpots are rigid."""
    sig, C_alg, _ = MaterialPoint(Maxwell(ELASTIC, 0.3, 1.0)).Update(EPS, dt=0.0)

    assert np.allclose(sig, C @ EPS)
    assert np.allclose(C_alg, C)


def test_relaxation_matches_the_backward_euler_closed_form():
    """sigma_n = C:eps [(1 - sum g) + sum g (1 + dt/tau)^-n] for a held strain."""
    g, tau, dt, nstep = np.array(G), np.array(TAU), 0.25, 12
    sig = _hold(Maxwell(ELASTIC, g, tau), EPS, nstep, dt)

    factor = 1 - g.sum() + g @ (1 + dt / tau) ** -nstep
    assert np.allclose(sig, factor * C @ EPS, rtol=1e-12)


def test_fully_relaxed_response_is_the_equilibrium_stiffness():
    assert np.allclose(_hold(Maxwell(ELASTIC, 0.3, 1.0), EPS, 200, 1.0), 0.7 * C @ EPS)


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_tangent_matches_central_difference(planeStress: bool):
    point = MaterialPoint(
        Maxwell(Isotropic(2, E=E, v=nu, planeStress=planeStress), G, TAU)
    )
    eps = EPS[[0, 1, 5]]
    z = point.Update(eps, dt=0.5).internals  # a history, so eps_v is not zero

    assert_consistent_tangent(point, 2 * eps, z, 0.5)


def test_tangent_matches_central_difference():
    assert_consistent_tangent(MaterialPoint(Maxwell(ELASTIC, G, TAU)), EPS, dt=5.0)
