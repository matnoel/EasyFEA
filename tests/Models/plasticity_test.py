# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Plasticity, Norton and Chaboche on the behavior contract, against closed forms. Skipped whole when jax is absent."""

import numpy as np
import pytest
from scipy.optimize import brentq

from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Models import _autodiff
from EasyFEA.Models.Elastic._laws import Isotropic, Orthotropic
from EasyFEA.Models.InElastic import Chaboche, MaterialPoint, Norton, Plasticity
from EasyFEA.Models.InElastic.IsotropicHardening import Linear, Perfect, Swift, Voce
from EasyFEA.Models.InElastic.Yield import DruckerPrager, Hill, VonMises

pytest.importorskip("jax")
_autodiff.Enable_x64()

E, nu = 210000.0, 0.3
SIGMA_Y, H = 250.0, 2000.0
EPS_Y = SIGMA_Y / E
ELASTIC = Isotropic(3, E=E, v=nu)


def _fe(vec) -> FeArray.FeArrayALike:
    return FeArray.asfearray(np.asarray(vec, dtype=float)[np.newaxis, np.newaxis])


def _at(field) -> np.ndarray:
    return np.asarray(field)[0, 0]


def _central_difference(behavior, eps, z=None, dt: float = 0.0, h: float = 1e-9):
    C_fd = np.zeros((eps.size, eps.size))
    for j, d in enumerate(np.eye(eps.size) * h):
        sigP = _at(behavior.Integrate(_fe(eps + d), z, dt)[0])
        sigM = _at(behavior.Integrate(_fe(eps - d), z, dt)[0])
        C_fd[:, j] = (sigP - sigM) / (2 * h)
    return C_fd


def _assert_tangent(behavior, eps, z=None, dt: float = 0.0):
    """C_alg == dsigma/deps by central differences, at a point that really flows."""
    _, C_alg, zNew = behavior.Integrate(_fe(eps), z, dt)
    assert _at(zNew["p"]).max() > 0.0
    assert np.allclose(
        _at(C_alg), _central_difference(behavior, eps, z, dt), rtol=1e-5, atol=1e-2
    )


EPS = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
HILL = Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4)
SURFACES = {
    "VonMises": VonMises(SIGMA_Y),
    "DruckerPrager": DruckerPrager(SIGMA_Y, 0.2),
    "Hill": HILL,
}


def _uniaxial(behavior, path, **kwargs) -> dict:
    return MaterialPoint(behavior).Run(strain={"xx": np.asarray(path)}, **kwargs)


def test_uniaxial_bar_matches_the_closed_form():
    """sigma = E (sigma_y + H eps) / (E + H) once yielding."""
    behavior = Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(H))
    sig = _uniaxial(behavior, [0.5 * EPS_Y, 5 * EPS_Y])["stress"][:, 0]

    assert np.isclose(sig[0], 0.5 * SIGMA_Y, rtol=1e-9)
    assert np.isclose(sig[1], E * (SIGMA_Y + H * 5 * EPS_Y) / (E + H), rtol=1e-9)


@pytest.mark.parametrize("surface", list(SURFACES))
def test_plastic_tangent_matches_central_difference(surface: str):
    _assert_tangent(Plasticity(ELASTIC, SURFACES[surface], Linear(H)), EPS)


def test_hill_defaults_to_von_mises():
    sig = np.random.default_rng(0).normal(0.0, 200.0, 6)

    assert np.isclose(Hill(SIGMA_Y)(sig, 10.0), VonMises(SIGMA_Y)(sig, 10.0))


def test_hill_yields_uniaxially_at_sigma_y_over_sqrt_g_plus_h():
    sig = SIGMA_Y / np.sqrt(0.4 + 0.6) * np.array([1.0, 0, 0, 0, 0, 0]) * 1.2

    assert np.isclose(HILL(sig, 0.0), 0.2 * SIGMA_Y)


Q, B = 150.0, 30.0
K, N_SWIFT, EPS0 = 600.0, 0.2, 1e-4
HARDENINGS = {
    "Linear": (Linear(H), lambda p: H * p),
    "Voce": (Voce(Q, B), lambda p: Q * (1 - np.exp(-B * p))),
    "Swift": (Swift(K, N_SWIFT), lambda p: K * ((EPS0 + p) ** N_SWIFT - EPS0**N_SWIFT)),
}


@pytest.mark.parametrize("hardening", list(HARDENINGS))
def test_uniaxial_tension_matches_its_closed_form(hardening: str):
    """sigma = E (eps - p) with sigma = sigma_y + R(p), one scalar root per step."""
    law, R = HARDENINGS[hardening]
    path = np.linspace(0.0, 30 * EPS_Y, 40)
    res = _uniaxial(Plasticity(ELASTIC, VonMises(SIGMA_Y), law), path)

    for e, sig in zip(path[path > EPS_Y], res["stress"][path > EPS_Y, 0]):
        p = brentq(lambda p: E * (e - p) - SIGMA_Y - R(p), 0.0, e, xtol=1e-15)
        assert np.isclose(sig, E * (e - p), rtol=1e-10)


def test_perfect_plasticity_caps_the_stress():
    res = _uniaxial(Plasticity(ELASTIC, VonMises(SIGMA_Y)), [20 * EPS_Y])

    assert np.isclose(res["stress"][0, 0], SIGMA_Y, rtol=1e-9)


def test_unloading_is_elastic():
    """Unloading follows E and leaves the plastic strain behind."""
    res = _uniaxial(
        Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(H)), [5 * EPS_Y, 4 * EPS_Y]
    )

    assert np.isclose(res["stress"][0, 0] - res["stress"][1, 0], E * EPS_Y, rtol=1e-9)
    assert np.allclose(res["eps_p"][0], res["eps_p"][1], rtol=0, atol=1e-15)
    assert res["p"][1] == res["p"][0]


def test_plastic_strain_is_deviatoric_and_p_is_its_equivalent():
    res = _uniaxial(Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(H)), [5 * EPS_Y])
    eps_p = res["eps_p"][0]

    assert abs(eps_p[:3].sum()) < 1e-15
    assert np.isclose(res["p"][0], np.sqrt(2 / 3 * eps_p @ eps_p), rtol=1e-10)


def test_orthotropic_elasticity_flows_with_the_right_tangent():
    """The normal is recomputed inside the solve, so nothing assumes C is isotropic."""
    elastic = Orthotropic(
        3,
        E1=E,
        E2=E / 2,
        E3=E / 3,
        G12=E / 4,
        G13=E / 5,
        G23=E / 6,
        v12=0.3,
        v13=0.2,
        v23=0.1,
    )
    _assert_tangent(Plasticity(elastic, VonMises(SIGMA_Y), Linear(H)), EPS)


EPS_2D = np.array([4e-3, -1e-3, 5e-4])


def test_plane_stress_holds_sigma_zz_at_zero_once_flowing():
    """The 3D behavior at the solved eps_zz gives back the 2D stress and sig_zz = 0."""
    plane = Plasticity(Isotropic(2, E=E, v=nu), VonMises(SIGMA_Y), Linear(H))
    sig2, _, z = plane.Integrate(_fe(EPS_2D))
    assert _at(z["p"]) > 0.0

    eps6 = np.zeros(6)
    eps6[[0, 1, 5]] = EPS_2D
    eps6[2] = _at(z["eps_zz"])
    sig6 = _at(
        Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(H)).Integrate(_fe(eps6))[0]
    )

    assert abs(sig6[2]) < 1e-9 * SIGMA_Y
    assert np.allclose(sig6[[0, 1, 5]], _at(sig2), rtol=1e-12)


@pytest.mark.parametrize("planeStress", [False, True])
def test_2d_plastic_tangent_matches_central_difference(planeStress: bool):
    behavior = Plasticity(
        Isotropic(2, E=E, v=nu, planeStress=planeStress), VonMises(SIGMA_Y), Linear(H)
    )
    _assert_tangent(behavior, EPS_2D)


def test_plane_stress_flows_from_a_committed_plastic_state():
    """Reloading from a plastic state: eps_zz starts from the committed one, flows may switch between its iterates."""
    behavior = Plasticity(Isotropic(2, E=E, v=nu), VonMises(SIGMA_Y), Voce(Q, B))
    _, _, z = behavior.Integrate(_fe(EPS_2D))
    _assert_tangent(behavior, 1.2 * EPS_2D, z)
    # unloading from it is elastic
    _, C_alg, zNew = behavior.Integrate(_fe(0.9 * EPS_2D), z)
    assert np.allclose(_at(zNew["p"]), _at(z["p"]), rtol=0, atol=1e-14)
    assert np.allclose(_at(C_alg), Isotropic(2, E=E, v=nu, planeStress=True).C)


def _norton(A: float, n: float = 1.0, hardening=Linear(H)) -> Norton:
    return Norton(ELASTIC, VonMises(SIGMA_Y), hardening, A=A, n=n, sigma_0=SIGMA_Y)


def test_changing_a_parameter_rebuilds_the_kernel():
    behavior = Norton(ELASTIC, VonMises(SIGMA_Y), A=1.0, n=3.0, sigma_0=100.0)
    sig = _at(behavior.Integrate(_fe(EPS), dt=1.0)[0])

    behavior.A = 100.0
    sigA = _at(behavior.Integrate(_fe(EPS), dt=1.0)[0])
    assert not np.allclose(sigA, sig)

    behavior.surface = VonMises(2 * SIGMA_Y)
    assert not np.allclose(_at(behavior.Integrate(_fe(EPS), dt=1.0)[0]), sigA)


def test_norton_parameters_must_be_strictly_positive():
    with pytest.raises(AssertionError, match="> 0"):
        Norton(ELASTIC, VonMises(SIGMA_Y), A=0.0)


def test_norton_needs_a_time_increment():
    with pytest.raises(AssertionError, match="positive time increment"):
        _norton(1e-3).Integrate(_fe(EPS), dt=0.0)


def test_norton_relaxes_onto_the_rate_independent_answer():
    """Held strain: the overstress bleeds off, monotonically, down to sigma_y + H p."""
    eps_0 = 3 * EPS_Y
    sig = _uniaxial(_norton(1e-2), np.full(60, eps_0), dt=0.5)["stress"][:, 0]

    assert np.all(np.diff(sig) <= 0)
    assert np.isclose(sig[-1], (SIGMA_Y + H * eps_0) / (1 + H / E), rtol=1e-8)


def test_norton_creeps_at_its_rate_under_held_stress():
    """n = 1, no hardening: the creep rate is A (sigma - sigma_y) / sigma_0."""
    A, held, dt = 1e-2, 1.2 * SIGMA_Y, 0.5
    res = MaterialPoint(_norton(A, hardening=Linear(0.0))).Run(
        strain={"yz": np.zeros(10)}, stress={"xx": np.full(10, held)}, dt=dt
    )

    rate = np.diff(res["strain"][:, 0]) / dt
    assert np.allclose(rate, A * (held - SIGMA_Y) / SIGMA_Y, rtol=0, atol=1e-12)


def test_fast_norton_is_rate_independent():
    sig_visc = _at(_norton(1e8).Integrate(_fe(EPS), dt=1.0)[0])
    sig_plas = _at(
        Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(H)).Integrate(_fe(EPS))[0]
    )

    assert np.allclose(sig_visc, sig_plas, rtol=1e-6)


@pytest.mark.parametrize("n", [1.0, 3.0])
def test_norton_tangent_matches_central_difference(n: float):
    _assert_tangent(_norton(1e-2, n), EPS, dt=1.0)


COMPONENTS = [(60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0)]
C_KIN = 20000.0


def _chaboche(components, hardening=Perfect()) -> Chaboche:
    C_X, gamma = zip(*components)
    return Chaboche(ELASTIC, VonMises(SIGMA_Y), C_X, gamma, hardening)


def _cycle(peak: float) -> np.ndarray:
    return np.concatenate(
        [np.linspace(0.0, peak, 20), np.linspace(peak, -peak, 40)[1:]]
    )


def test_chaboche_state_is_sized_by_its_components():
    behavior = _chaboche(COMPONENTS)

    assert behavior.Virgin_internals().alpha.shape == (3, 6)
    assert behavior.Virgin_internals_e_pg(5, 4)["alpha"].shape == (5, 4, 3, 6)


def test_prager_hardens_like_linear_isotropic_in_monotonic_tension():
    path = np.linspace(0.0, 6 * EPS_Y, 20)
    kin = _uniaxial(_chaboche([(C_KIN, 0.0)]), path)["stress"][:, 0]
    iso = _uniaxial(Plasticity(ELASTIC, VonMises(SIGMA_Y), Linear(C_KIN)), path)

    assert np.allclose(kin, iso["stress"][:, 0], rtol=0, atol=1e-6)


def test_prager_keeps_the_elastic_range_at_two_sigma_y():
    """The Bauschinger effect: the surface moves instead of growing."""
    res = _uniaxial(
        _chaboche([(C_KIN, 0.0)]),
        np.concatenate([[8 * EPS_Y], np.linspace(8 * EPS_Y, -8 * EPS_Y, 400)]),
    )
    sig, p = res["stress"][:, 0], res["p"]
    resumed = int(np.argmax(p > p[0] + 1e-12))

    assert sig[0] - sig[resumed - 1] <= 2 * SIGMA_Y <= sig[0] - sig[resumed]


def test_chaboche_back_stresses_saturate_and_sit_on_the_surface():
    res = _uniaxial(_chaboche(COMPONENTS), _cycle(8 * EPS_Y))
    X = np.stack(
        [2 / 3 * C * res["alpha"][:, i, 0] for i, (C, _) in enumerate(COMPONENTS)]
    )

    for (C, gamma), Xi in zip(COMPONENTS, X):
        if gamma > 0:
            assert np.abs(Xi).max() <= 2 * C / (3 * gamma) * (1 + 1e-9)
    # gamma = 0: alpha is the plastic strain
    assert np.allclose(res["alpha"][:, 2], res["eps_p"], rtol=0, atol=1e-15)
    k = int(np.argmax(np.abs(res["stress"][:, 0])))
    assert np.isclose(
        abs(res["stress"][k, 0]), SIGMA_Y + 1.5 * abs(X[:, k].sum()), rtol=1e-10
    )


def test_chaboche_tangent_matches_central_difference():
    """Three back-stresses couple through the flow direction."""
    _assert_tangent(_chaboche(COMPONENTS, Linear(H)), EPS)
