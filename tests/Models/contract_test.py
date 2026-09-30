# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The behavior contract: one Update at one 3D point, lifted by EasyFEA. Skipped whole when jax is absent."""

import pickle
import subprocess
import sys
from typing import NamedTuple

import numpy as np
import pytest

from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Models import _autodiff
from EasyFEA.Models.Elastic._laws import Isotropic
from EasyFEA.Models.InElastic.Contract import (
    ZERO_SCALAR,
    ZERO_TENSOR,
    _Behavior,
    Deviator,
    MaterialPoint,
    Newton,
    Trace,
    Von_Mises_stress,
)

jax = pytest.importorskip("jax")
jnp = jax.numpy

_autodiff.Enable_x64()

E, nu = 210000.0, 0.3
EPS = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
C = Isotropic(3, E=E, v=nu).C


def _fe(vec) -> FeArray.FeArrayALike:
    """A (1, 1, ...) field holding one value."""
    return FeArray.asfearray(np.asarray(vec, dtype=float)[np.newaxis, np.newaxis])


def _at(field) -> np.ndarray:
    return np.asarray(field)[0, 0]


class Linear(_Behavior):
    """No internal variable: sigma = k C eps."""

    def __init__(self, k: float = 1.0, **plane):
        super().__init__(**plane)
        self.k = k

    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        return self.k * C @ eps, z


class Damage(_Behavior):
    """A tensor and a scalar internal variable, to check the packing."""

    class State(NamedTuple):
        eps_old: np.ndarray = ZERO_TENSOR
        d: np.ndarray = ZERO_SCALAR

    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        d = z.d + 0.1
        return (1 - d) * (C @ eps), Damage.State(eps_old=eps, d=d)


class NoRoot(_Behavior):
    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        x = Newton(lambda x: x**2 + 1.0, jnp.array([1.0]))
        return C @ eps + x[0], z


def test_importing_easyfea_does_not_pull_jax():
    """In a subprocess, because this module imports jax itself."""
    code = "import EasyFEA, sys; sys.exit('jax' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ----------------------------------------------
# Helpers
# ----------------------------------------------


def test_helpers_on_uniaxial_stress():
    sig = jnp.array([300.0, 0, 0, 0, 0, 0])

    assert float(Trace(sig)) == 300.0
    assert np.allclose(Deviator(sig), [200.0, -100, -100, 0, 0, 0])
    assert np.isclose(float(Von_Mises_stress(sig)), 300.0)


def test_von_mises_stress_is_differentiable_at_zero():
    assert np.all(np.isfinite(jax.grad(Von_Mises_stress)(jnp.zeros(6))))


class Pair(NamedTuple):
    a: jax.Array
    b: jax.Array


def test_newton_solves_a_state_and_its_derivative_goes_through_the_root():
    def Solve(c):
        return Newton(lambda x: Pair(a=x.a**2 - c, b=x.b - 2 * x.a), Pair(1.0, 0.0))

    root = Solve(4.0)
    assert np.isclose(float(root.a), 2.0) and np.isclose(float(root.b), 4.0)
    # b = 2 sqrt(c), so db/dc = 1 / sqrt(c)
    assert np.isclose(float(jax.grad(lambda c: Solve(c).b)(4.0)), 0.5)


def test_newton_leaves_an_inactive_point_at_its_guess():
    x = Newton(lambda x: x**2 - 4.0, jnp.array([1.0]), active=False)
    assert np.allclose(x, 1.0)


def test_newton_flags_a_missing_root_as_nan():
    assert np.isnan(Newton(lambda x: x**2 + 1.0, jnp.array([1.0]))).all()


# ----------------------------------------------
# The engine
# ----------------------------------------------


def test_no_internal_variable_gives_back_the_elastic_response():
    sig, C_alg, z = Linear().Integrate(_fe(EPS))

    assert np.allclose(_at(sig), C @ EPS)
    assert np.allclose(_at(C_alg), C)
    assert z == {}


def test_state_is_held_by_name():
    z = Damage().Integrate(_fe(EPS))[2]

    assert list(z) == ["eps_old", "d"]
    assert np.allclose(_at(z["eps_old"]), EPS) and np.isclose(_at(z["d"]), 0.1)


def test_every_gauss_point_is_integrated():
    eps = np.random.default_rng(0).normal(0, 1e-3, (4, 3, 6))
    sig, C_alg, z = Damage().Integrate(FeArray.asfearray(eps))

    assert isinstance(sig, FeArray) and sig.shape == (4, 3, 6)
    assert C_alg.shape == (4, 3, 6, 6)
    assert z["eps_old"].shape == (4, 3, 6) and z["d"].shape == (4, 3)
    assert np.allclose(sig[1, 2], 0.9 * C @ eps[1, 2])


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_matches_the_elastic_law(planeStress: bool):
    eps = EPS[[0, 1, 5]]
    sig, C_alg, _ = Linear(dim=2, planeStress=planeStress).Integrate(_fe(eps))

    ref = Isotropic(2, E=E, v=nu, planeStress=planeStress).C
    assert np.allclose(_at(sig), ref @ eps)
    assert np.allclose(_at(C_alg), ref)


def test_plane_stress_keeps_eps_zz_in_the_state():
    """Elastic plane stress: eps_zz = -nu / (1 - nu) (eps_xx + eps_yy)."""
    behavior = Damage(dim=2, planeStress=True)
    z = behavior.Integrate(_fe(EPS[[0, 1, 5]]))[2]

    assert np.isclose(_at(z["eps_zz"]), -nu / (1 - nu) * (EPS[0] + EPS[1]))


def test_plane_stress_solve_settings_are_checked():
    behavior = Linear()
    with pytest.raises(AssertionError):
        behavior._tol = -1.0
    with pytest.raises(AssertionError):
        behavior._maxIter = -1


def test_plane_stress_is_a_2d_assumption():
    with pytest.raises(AssertionError):
        Linear(dim=3, planeStress=True)


def test_a_failed_local_solve_is_reported():
    with pytest.raises(AssertionError, match="did not converge"):
        NoRoot().Integrate(_fe(EPS))


def test_changing_a_parameter_rebuilds_the_kernel():
    behavior = Linear()
    behavior.Integrate(_fe(EPS))
    behavior.k = 2.0
    behavior.Need_Update()

    assert np.allclose(_at(behavior.Integrate(_fe(EPS))[0]), 2 * C @ EPS)


def test_behavior_survives_a_pickle_round_trip():
    """``Load_Simu`` pickles the material with the simulation, after its kernel is built."""
    behavior = Damage(dim=2, planeStress=True, thickness=5.0)
    eps = EPS[[0, 1, 5]]
    before = behavior.Integrate(_fe(eps))

    reloaded = pickle.loads(pickle.dumps(behavior))

    assert reloaded.thickness == 5.0 and reloaded.planeStress
    sig, C_alg, z = reloaded.Integrate(_fe(eps))
    assert np.allclose(sig, before[0]) and np.allclose(C_alg, before[1])
    assert all(np.allclose(z[name], before[2][name]) for name in z)


# ----------------------------------------------
# MaterialPoint
# ----------------------------------------------


def test_material_point_solves_uniaxial_stress():
    """eps_xx driven, everything else free: sig_xx = E eps_xx and the lateral strain is -nu eps_xx."""
    res = MaterialPoint(Linear()).Run(strain={"xx": np.array([1e-3, 2e-3])})

    assert np.allclose(res["stress"][:, 0], E * np.array([1e-3, 2e-3]))
    assert np.allclose(res["stress"][:, 1:], 0.0, atol=1e-6)
    assert np.allclose(res["strain"][:, 1], -nu * np.array([1e-3, 2e-3]))


def test_material_point_returns_each_internal_variable():
    res = MaterialPoint(Damage()).Run(strain={"xx": np.zeros(3), "yy": np.zeros(3)})

    assert np.allclose(res["d"], [0.1, 0.2, 0.3])
    assert res["eps_old"].shape == (3, 6)
