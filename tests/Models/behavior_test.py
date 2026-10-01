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
from EasyFEA.Models.Elastic._laws import Anisotropic, Isotropic, _Elastic
from EasyFEA.Models.InElastic import (
    ONE,
    ZERO_SCALAR,
    ZERO_TENSOR,
    _Behavior,
    Deviator,
    MaterialPoint,
    Newton,
    Plasticity,
    Trace,
    Von_Mises_stress,
)
from EasyFEA.Models.InElastic.IsotropicHardening import Linear as LinearIsotropic
from EasyFEA.Models.InElastic.Yield import VonMises
from EasyFEA.Utilities import _params

jax = pytest.importorskip("jax")
jnp = jax.numpy

_autodiff.Enable_x64()

E, nu = 210000.0, 0.3
ALPHA = 1e-5
EPS = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
C = Isotropic(3, E=E, v=nu).C


def _elastic(dim: int = 3, **kwargs) -> Isotropic:
    return Isotropic(dim, E=E, v=nu, **kwargs)


def _fe(vec) -> FeArray.FeArrayALike:
    """A (1, 1, ...) field holding one value."""
    return FeArray.asfearray(np.asarray(vec, dtype=float)[np.newaxis, np.newaxis])


def _at(field) -> np.ndarray:
    return np.asarray(field)[0, 0]


class Linear(_Behavior):
    """No internal variable: sigma = k C eps."""

    def __init__(self, elastic: _Elastic, k: float = 1.0):
        super().__init__(elastic)
        self.k = k

    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        return self.Stress(eps, z), z

    def Stress(self, eps, z, **external):
        return self.k * self.C @ eps


class Damage(_Behavior):
    """A tensor and a scalar internal variable, to check the packing."""

    class Internals(NamedTuple):
        eps_old: np.ndarray = ZERO_TENSOR
        d: np.ndarray = ZERO_SCALAR

    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        new = Damage.Internals(eps_old=eps, d=z.d + 0.1)
        return self.Stress(eps, new), new

    def Stress(self, eps, z, **external):
        return (1 - z.d) * (self.C @ eps)


class NoRoot(_Behavior):
    def Update(
        self,
        eps,
        z,
        dt,
        **external,
    ):
        x = Newton(lambda x: x**2 + 1.0, jnp.array([1.0]))
        return self.C @ eps + x[0], z

    def Stress(self, eps, z, **external):
        return self.C @ eps


class ThermoElastic(_Behavior):
    """Reads the temperature change T."""

    class Externals(NamedTuple):
        T: float

    def Update(self, eps, z, dt, T):
        return self.Stress(eps, z, T=T), z

    def Stress(self, eps, z, T):
        return self.C @ (eps - ALPHA * T * ONE)


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
    sig, C_alg, z = Linear(_elastic()).Integrate(_fe(EPS))

    assert np.allclose(_at(sig), C @ EPS)
    assert np.allclose(_at(C_alg), C)
    assert z == {}


def test_state_is_held_by_name():
    z = Damage(_elastic()).Integrate(_fe(EPS))[2]

    assert list(z) == ["eps_old", "d"]
    assert np.allclose(_at(z["eps_old"]), EPS) and np.isclose(_at(z["d"]), 0.1)


def test_every_gauss_point_is_integrated():
    eps = np.random.default_rng(0).normal(0, 1e-3, (4, 3, 6))
    sig, C_alg, z = Damage(_elastic()).Integrate(FeArray.asfearray(eps))

    assert isinstance(sig, FeArray) and sig.shape == (4, 3, 6)
    assert C_alg.shape == (4, 3, 6, 6)
    assert z["eps_old"].shape == (4, 3, 6) and z["d"].shape == (4, 3)
    assert np.allclose(sig[1, 2], 0.9 * C @ eps[1, 2])


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_matches_the_elastic_law(planeStress: bool):
    eps = EPS[[0, 1, 5]]
    sig, C_alg, _ = Linear(_elastic(2, planeStress=planeStress)).Integrate(_fe(eps))

    ref = Isotropic(2, E=E, v=nu, planeStress=planeStress).C
    assert np.allclose(_at(sig), ref @ eps)
    assert np.allclose(_at(C_alg), ref)


def test_plane_stress_keeps_eps_zz_in_the_state():
    """Elastic plane stress: eps_zz = -nu / (1 - nu) (eps_xx + eps_yy)."""
    behavior = Damage(_elastic(2, planeStress=True))
    z = behavior.Integrate(_fe(EPS[[0, 1, 5]]))[2]

    assert np.isclose(_at(z["eps_zz"]), -nu / (1 - nu) * (EPS[0] + EPS[1]))


def test_plane_stress_solve_settings_are_checked():
    behavior = Linear(_elastic())
    with pytest.raises(AssertionError):
        behavior._tol = -1.0
    with pytest.raises(AssertionError):
        behavior._maxIter = -1


def test_a_3d_model_is_never_plane_stress():
    assert not Linear(_elastic(3, planeStress=True)).planeStress


def test_a_2d_anisotropic_model_has_no_3d_stiffness():
    behavior = Linear(Anisotropic(2, _elastic(2).C, useVoigtNotation=False))
    with pytest.raises(AssertionError, match="own dimension"):
        behavior.C


def test_a_failed_local_solve_is_reported():
    with pytest.raises(AssertionError, match="did not converge"):
        NoRoot(_elastic()).Integrate(_fe(EPS))


def test_changing_a_parameter_rebuilds_the_kernel():
    behavior = Linear(_elastic())
    behavior.Integrate(_fe(EPS))
    behavior.k = 2.0
    behavior.Need_Update()

    assert np.allclose(_at(behavior.Integrate(_fe(EPS))[0]), 2 * C @ EPS)


def test_modifying_the_elastic_model_rebuilds_the_kernel():
    elastic = _elastic()
    behavior = Linear(elastic)
    behavior.Integrate(_fe(EPS))
    elastic.E = 2 * E

    assert np.allclose(_at(behavior.Integrate(_fe(EPS))[0]), 2 * C @ EPS)


def test_missing_or_unknown_external_variables_are_refused():
    behavior = ThermoElastic(_elastic())
    with pytest.raises(AssertionError, match="missing"):
        behavior.Integrate(_fe(EPS))
    with pytest.raises(AssertionError, match="reads no"):
        behavior.Integrate(_fe(EPS), T=1.0, P=1.0)


@pytest.mark.parametrize("per", ["element", "point"])
def test_a_heterogeneous_elastic_model_is_integrated_point_by_point(per: str):
    """C scales with E, so sigma = E / E_ref C_ref eps at each point."""
    scale = np.array([[1.0, 2.0, 0.5], [3.0, 1.5, 0.8]])
    if per == "element":
        scale = scale[:, :1].repeat(3, axis=1)
    E_e_pg = E * (scale[:, 0] if per == "element" else scale)
    eps = FeArray.asfearray(np.broadcast_to(EPS, (2, 3, 6)).copy())

    sig, C_alg, _ = Linear(Isotropic(3, E=E_e_pg, v=nu)).Integrate(eps)

    assert np.allclose(sig, scale[..., None] * (C @ EPS))
    assert np.allclose(C_alg, scale[..., None, None] * C)


def test_behavior_survives_a_pickle_round_trip():
    """``Load_Simu`` pickles the material with the simulation, after its kernel is built."""
    behavior = Damage(_elastic(2, planeStress=True, thickness=5.0))
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
    res = MaterialPoint(Linear(_elastic())).Run(strain={"xx": np.array([1e-3, 2e-3])})

    assert np.allclose(res["stress"][:, 0], E * np.array([1e-3, 2e-3]))
    assert np.allclose(res["stress"][:, 1:], 0.0, atol=1e-6)
    assert np.allclose(res["strain"][:, 1], -nu * np.array([1e-3, 2e-3]))


def test_material_point_reads_the_external_variables_at_each_step():
    """Fully constrained heating: sigma = -3 K alpha T on the diagonal."""
    T = np.array([0.0, 10.0, 20.0])
    zero = np.zeros(3)
    res = MaterialPoint(ThermoElastic(_elastic())).Run(
        strain={"xx": zero, "yy": zero, "zz": zero}, external={"T": T}
    )

    K = E / (3 * (1 - 2 * nu))
    assert np.allclose(res["stress"][:, :3], (-3 * K * ALPHA * T)[:, None])


def test_material_point_returns_each_internal_variable():
    res = MaterialPoint(Damage(_elastic())).Run(
        strain={"xx": np.zeros(3), "yy": np.zeros(3)}
    )

    assert np.allclose(res["d"], [0.1, 0.2, 0.3])
    assert res["eps_old"].shape == (3, 6)


# ----------------------------------------------
# The how-to example
# ----------------------------------------------


class LinearHardening(_Behavior):
    """The how-to's example: keep it identical to docs/howto/create_models.md."""

    class Internals(NamedTuple):
        eps_p: jax.Array = ZERO_TENSOR  # plastic strain
        p: jax.Array = ZERO_SCALAR  # accumulated plastic strain

    sigma_y: float = _params.PositiveScalarParameter()
    H: float = _params.PositiveScalarParameter()

    def __init__(self, elastic, sigma_y, H):
        super().__init__(elastic)
        self.sigma_y = sigma_y
        self.H = H

    def Stress(self, eps, z):
        return self.C @ (eps - z.eps_p)  # C: the 3D stiffness, even in 2D

    def Update(self, eps, z, dt):
        def f(sig, p):
            return Von_Mises_stress(sig) - self.sigma_y - self.H * p

        def Residual(new):
            sig = self.Stress(eps, new)
            N = jax.grad(f)(sig, new.p)
            return LinearHardening.Internals(
                eps_p=new.eps_p - z.eps_p - (new.p - z.p) * N,
                p=f(sig, new.p) / self.sigma_y,
            )

        flows = f(self.Stress(eps, z), z.p) > 0  # elsewhere, the internal state stays z
        new = Newton(Residual, z, flows)
        return self.Stress(eps, new), new


def test_the_howto_behavior_matches_plasticity():
    try:
        mine = LinearHardening(_elastic(), 250.0, 1e3)
        ref = Plasticity(_elastic(), VonMises(250.0), LinearIsotropic(1e3))
        path = {"xx": np.linspace(0, 5e-3, 20)}  # uniaxial stress, through yield
        a, b = MaterialPoint(mine).Run(path), MaterialPoint(ref).Run(path)
        assert np.allclose(a["stress"], b["stress"], rtol=1e-10, atol=1e-8)
        assert np.allclose(a["p"], b["p"], rtol=1e-10, atol=1e-14)
    except Exception as error:
        raise AssertionError(
            "LinearHardening no longer matches Plasticity: fix it, then copy the "
            "updated version into docs/howto/create_models.md"
        ) from error
