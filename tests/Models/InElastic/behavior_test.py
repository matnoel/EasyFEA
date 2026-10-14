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

from EasyFEA.FEM import FeArray
from EasyFEA.Models.Elastic import Anisotropic, Isotropic, _Elastic
from EasyFEA.Models.InElastic import (
    ONE,
    PointUpdate,
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

from ..conftest import affine_kinematics

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

E, nu = 210000.0, 0.3
ALPHA = 1e-5
EPS = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
C = Isotropic(3, E=E, v=nu).C


def _elastic(dim: int = 3, **kwargs) -> Isotropic:
    return Isotropic(dim, E=E, v=nu, **kwargs)


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


def test_Behavior_importing_easyfea_does_not_pull_jax():
    """In a subprocess, because this module imports jax itself."""
    code = "import EasyFEA, sys; sys.exit('jax' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ----------------------------------------------
# Helpers
# ----------------------------------------------


UNIAXIAL = jnp.array([300.0, 0, 0, 0, 0, 0])


def test_Trace_on_uniaxial_stress():
    assert float(Trace(UNIAXIAL)) == 300.0


def test_Deviator_on_uniaxial_stress():
    assert np.allclose(Deviator(UNIAXIAL), [200.0, -100, -100, 0, 0, 0])


def test_Von_Mises_stress_on_uniaxial_stress():
    assert np.isclose(float(Von_Mises_stress(UNIAXIAL)), 300.0)


def test_Von_Mises_stress_is_differentiable_at_zero():
    assert np.all(np.isfinite(jax.grad(Von_Mises_stress)(jnp.zeros(6))))


class Pair(NamedTuple):
    a: jax.Array
    b: jax.Array


def test_Newton_solves_a_state_and_its_derivative_goes_through_the_root():
    def Solve(c):
        return Newton(lambda x: Pair(a=x.a**2 - c, b=x.b - 2 * x.a), Pair(1.0, 0.0))

    root = Solve(4.0)
    assert np.isclose(float(root.a), 2.0) and np.isclose(float(root.b), 4.0)
    # b = 2 sqrt(c), so db/dc = 1 / sqrt(c)
    assert np.isclose(float(jax.grad(lambda c: Solve(c).b)(4.0)), 0.5)


def test_Newton_leaves_an_inactive_point_at_its_guess():
    x = Newton(lambda x: x**2 - 4.0, jnp.array([1.0]), active=False)
    assert np.allclose(x, 1.0)


def test_Newton_flags_a_missing_root_as_nan():
    assert np.isnan(Newton(lambda x: x**2 + 1.0, jnp.array([1.0]))).all()


# ----------------------------------------------
# The engine
# ----------------------------------------------


def test_Behavior_Update_without_internals_is_elastic():
    sig, C_alg, z = MaterialPoint(Linear(_elastic())).Update(EPS)

    assert np.allclose(sig, C @ EPS)
    assert np.allclose(C_alg, C)
    assert z == {}


def test_Behavior_Update_holds_the_state_by_name():
    z = MaterialPoint(Damage(_elastic())).Update(EPS).internals

    assert list(z) == ["eps_old", "d"]
    assert np.allclose(z["eps_old"], EPS) and np.isclose(z["d"], 0.1)


def test_Behavior_Integrate_every_gauss_point():
    affine = affine_kinematics(3)
    sig, C_alg, z = Damage(_elastic()).Integrate(affine.kinematics)

    assert isinstance(sig, FeArray) and sig.shape == (1, 8, 6)
    assert C_alg.shape == (1, 8, 6, 6)
    assert z["eps_old"].shape == (1, 8, 6) and z["d"].shape == (1, 8)
    assert np.allclose(sig, 0.9 * C @ affine.epsilon)


def test_Behavior_planeStress_keeps_eps_zz_in_the_state():
    """Elastic plane stress: eps_zz = -nu / (1 - nu) (eps_xx + eps_yy)."""
    behavior = Damage(_elastic(2, planeStress=True))
    z = MaterialPoint(behavior).Update(EPS[[0, 1, 5]]).internals

    assert np.isclose(z["eps_zz"], -nu / (1 - nu) * (EPS[0] + EPS[1]))


def test_Behavior_planeStress_solve_settings_are_checked():
    behavior = Linear(_elastic())
    with pytest.raises(AssertionError):
        behavior._tol = -1.0
    with pytest.raises(AssertionError):
        behavior._maxIter = -1


def test_Behavior_planeStress_never_in_3d():
    assert not Linear(_elastic(3, planeStress=True)).planeStress


def test_Behavior_C_needs_a_3d_stiffness():
    behavior = Linear(Anisotropic(2, _elastic(2).C, useVoigtNotation=False))
    with pytest.raises(ValueError, match="no 3D stiffness"):
        behavior.C


def test_Behavior_Integrate_reports_a_failed_local_solve():
    with pytest.raises(AssertionError, match="did not converge"):
        MaterialPoint(NoRoot(_elastic())).Update(EPS)


def test_Behavior_Update_after_Need_Update_rebuilds_the_kernel():
    behavior = Linear(_elastic())
    point = MaterialPoint(behavior)
    point.Update(EPS)
    behavior.k = 2.0
    behavior.Need_Update()

    assert np.allclose(point.Update(EPS).stress, 2 * C @ EPS)


def test_Behavior_elastic_modified_rebuilds_the_kernel():
    elastic = _elastic()
    point = MaterialPoint(Linear(elastic))
    point.Update(EPS)
    elastic.E = 2 * E

    assert np.allclose(point.Update(EPS).stress, 2 * C @ EPS)


def test_Behavior_externalNames_missing_or_unknown_are_refused():
    point = MaterialPoint(ThermoElastic(_elastic()))
    with pytest.raises(AssertionError, match="missing"):
        point.Update(EPS)
    with pytest.raises(AssertionError, match="reads no"):
        point.Update(EPS, T=1.0, P=1.0)


def test_Behavior_Integrate_a_heterogeneous_elastic_model_point_by_point(subtests):
    """C scales with E, so sigma = E / E_ref C_ref eps at each point."""
    affine = affine_kinematics(3, Ne=2)
    perPoint = np.random.default_rng(0).uniform(0.5, 3.0, (2, 8))
    perElement = perPoint[:, :1].repeat(8, axis=1)
    for per, scale, E_e_pg in [
        ("element", perElement, E * perElement[:, 0]),
        ("point", perPoint, E * perPoint),
    ]:
        with subtests.test(per=per):
            elastic = Isotropic(3, E=E_e_pg, v=nu)
            sig, C_alg, _ = Linear(elastic).Integrate(affine.kinematics)

            assert np.allclose(sig, scale[..., None] * (C @ affine.epsilon))
            assert np.allclose(C_alg, scale[..., None, None] * C)


def test_Behavior_survives_a_pickle_round_trip():
    """``Load_Simu`` pickles the material with the simulation, after its kernel is built."""
    behavior = Damage(_elastic(2, planeStress=True, thickness=5.0))
    eps = EPS[[0, 1, 5]]
    before = MaterialPoint(behavior).Update(eps)

    reloaded = pickle.loads(pickle.dumps(behavior))

    assert reloaded.thickness == 5.0 and reloaded.planeStress
    sig, C_alg, z = MaterialPoint(reloaded).Update(eps)
    assert np.allclose(sig, before[0]) and np.allclose(C_alg, before[1])
    assert all(np.allclose(z[name], before[2][name]) for name in z)


def test_Behavior_elastic_is_the_given_model():
    elastic = _elastic()

    assert Linear(elastic).elastic is elastic


def test_Behavior_externalNames_follow_the_declaration():
    assert ThermoElastic(_elastic()).externalNames == ("T",)
    assert Linear(_elastic()).externalNames == ()


def test_Behavior_Virgin_internals_are_the_declared_defaults():
    z = Damage(_elastic()).Virgin_internals()

    assert np.array_equal(z.eps_old, np.zeros(6)) and z.d == 0.0


def test_Behavior_Virgin_internals_e_pg_adds_eps_zz_in_plane_stress(subtests):
    for planeStress in (False, True):
        with subtests.test(planeStress=planeStress):
            behavior = Damage(_elastic(2, planeStress=planeStress))
            z = behavior.Virgin_internals_e_pg(5, 4)

            assert z["eps_old"].shape == (5, 4, 6) and z["d"].shape == (5, 4)
            assert ("eps_zz" in z) is planeStress


def test_Behavior_Stress_is_what_Update_returns_at_the_new_state():
    behavior = Damage(_elastic())
    sig, new = behavior.Update(EPS, behavior.Virgin_internals(), 0.0)

    assert np.allclose(behavior.Stress(EPS, new), sig)


def test_Behavior__Integrate_at_a_prescribed_strain_field():
    eps_e_pg = np.broadcast_to(EPS, (2, 3, 6))
    sig, C_alg, z = Linear(_elastic(), k=2.0)._Integrate(eps_e_pg)

    assert sig.shape == (2, 3, 6) and np.allclose(sig, 2 * C @ EPS)
    assert np.allclose(C_alg, 2 * C) and z == {}


def test_Behavior_Compute_Sigma_reads_the_given_state_with_no_step():
    affine = affine_kinematics(3)
    behavior = Damage(_elastic())
    _, _, z = behavior.Integrate(affine.kinematics)

    assert np.allclose(
        behavior.Compute_Sigma(affine.kinematics, z), 0.9 * C @ affine.epsilon
    )


# ----------------------------------------------
# MaterialPoint
# ----------------------------------------------


def test_PointUpdate_names_stress_tangent_and_internals():
    update = MaterialPoint(Linear(_elastic())).Update(EPS)

    assert isinstance(update, PointUpdate)
    assert PointUpdate._fields == ("stress", "tangent", "internals")
    assert np.allclose(update.stress, C @ EPS) and update.internals == {}


def test_MaterialPoint_refuses_what_is_not_a_behavior():
    with pytest.raises(AssertionError, match="behavior must be a _Behavior"):
        MaterialPoint(_elastic())  # type: ignore[arg-type]


def test_MaterialPoint_Update_gives_plain_arrays_at_one_point():
    stress, tangent, internals = MaterialPoint(Damage(_elastic())).Update(EPS)

    assert np.allclose(stress, 0.9 * C @ EPS) and np.allclose(tangent, 0.9 * C)
    assert not isinstance(stress, FeArray) and stress.shape == (6,)
    assert np.allclose(internals["eps_old"], EPS) and np.isclose(internals["d"], 0.1)


def test_MaterialPoint_Update_continues_from_the_given_internals():
    point = MaterialPoint(Damage(_elastic()))
    internals = point.Update(EPS).internals

    assert np.isclose(point.Update(EPS, internals).internals["d"], 0.2)


def test_MaterialPoint_Update_runs_a_2d_behavior(subtests):
    eps = EPS[[0, 1, 5]]
    for planeStress in (False, True):
        with subtests.test(planeStress=planeStress):
            point = MaterialPoint(Linear(_elastic(2, planeStress=planeStress)))
            update = point.Update(eps)

            ref = Isotropic(2, E=E, v=nu, planeStress=planeStress).C
            assert np.allclose(update.stress, ref @ eps)
            assert np.allclose(update.tangent, ref)


def test_MaterialPoint_Update_refuses_a_strain_of_the_wrong_size():
    with pytest.raises(AssertionError, match="strain"):
        MaterialPoint(Linear(_elastic(2))).Update(EPS)


def test_MaterialPoint_Run_stays_3d():
    with pytest.raises(AssertionError, match="3D"):
        MaterialPoint(Linear(_elastic(2))).Run(strain={"xx": np.zeros(1)})


def test_MaterialPoint_Run_solves_uniaxial_stress():
    """eps_xx driven, everything else free: sig_xx = E eps_xx and the lateral strain is -nu eps_xx."""
    res = MaterialPoint(Linear(_elastic())).Run(strain={"xx": np.array([1e-3, 2e-3])})

    assert np.allclose(res["stress"][:, 0], E * np.array([1e-3, 2e-3]))
    assert np.allclose(res["stress"][:, 1:], 0.0, atol=1e-6)
    assert np.allclose(res["strain"][:, 1], -nu * np.array([1e-3, 2e-3]))


def test_MaterialPoint_Run_reads_the_external_variables_at_each_step():
    """Fully constrained heating: sigma = -3 K alpha T on the diagonal."""
    T = np.array([0.0, 10.0, 20.0])
    zero = np.zeros(3)
    res = MaterialPoint(ThermoElastic(_elastic())).Run(
        strain={"xx": zero, "yy": zero, "zz": zero}, external={"T": T}
    )

    K = E / (3 * (1 - 2 * nu))
    assert np.allclose(res["stress"][:, :3], (-3 * K * ALPHA * T)[:, None])


def test_MaterialPoint_Run_returns_each_internal_variable():
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


def test_Behavior_howto_example_matches_plasticity():
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
