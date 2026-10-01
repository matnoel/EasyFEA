# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Maxwell, the first behavior on the contract, against its closed forms. Skipped whole when jax is absent."""

import numpy as np
import pytest

from EasyFEA.Models import _autodiff
from EasyFEA.Models.Elastic._laws import Isotropic
from EasyFEA.Models.InElastic import Maxwell

from .conftest import one_point_field, point_value

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
    z = None
    for _ in range(nstep):
        sig, _, z = behavior.Integrate(one_point_field(eps), z, dt)
    return point_value(sig)


def _central_difference(
    behavior: Maxwell,
    eps,
    z,
    dt: float,
    h: float = 1e-9,
):
    C_fd = np.zeros((eps.size, eps.size))
    for j, d in enumerate(np.eye(eps.size) * h):
        sigP = point_value(behavior.Integrate(one_point_field(eps + d), z, dt)[0])
        sigM = point_value(behavior.Integrate(one_point_field(eps - d), z, dt)[0])
        C_fd[:, j] = (sigP - sigM) / (2 * h)
    return C_fd


def test_state_is_sized_by_the_branches():
    behavior = Maxwell(ELASTIC, G, TAU)

    assert behavior.Virgin_internals().eps_v.shape == (2, 6)
    assert behavior.Virgin_internals_e_pg(5, 4)["eps_v"].shape == (5, 4, 2, 6)
    # a (2, 6) default must not be taken for an (Ne, nPg) field
    assert behavior.Virgin_internals_e_pg(2, 6)["eps_v"].shape == (2, 6, 2, 6)


def test_fractions_must_leave_an_equilibrium_spring():
    with pytest.raises(AssertionError, match="sum to less than 1"):
        Maxwell(ELASTIC, [0.6, 0.4], [1.0, 2.0])


def test_glassy_response_is_the_full_stiffness():
    """dt = 0: the dashpots are rigid."""
    sig, C_alg, _ = Maxwell(ELASTIC, 0.3, 1.0).Integrate(one_point_field(EPS), dt=0.0)

    assert np.allclose(point_value(sig), C @ EPS)
    assert np.allclose(point_value(C_alg), C)


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
    behavior = Maxwell(Isotropic(2, E=E, v=nu, planeStress=planeStress), G, TAU)
    eps = EPS[[0, 1, 5]]
    _, _, z = behavior.Integrate(
        one_point_field(eps), dt=0.5
    )  # a history, so eps_v is not zero

    C_alg = point_value(behavior.Integrate(one_point_field(2 * eps), z, 0.5)[1])

    assert np.allclose(C_alg, _central_difference(behavior, 2 * eps, z, 0.5), rtol=1e-6)


def test_tangent_matches_central_difference():
    behavior = Maxwell(ELASTIC, G, TAU)
    C_alg = point_value(behavior.Integrate(one_point_field(EPS), dt=5.0)[1])

    assert np.allclose(C_alg, _central_difference(behavior, EPS, None, 5.0), rtol=1e-6)
