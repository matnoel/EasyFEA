# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Core functions of Models.Behaviour, on examples checkable by hand."""

import numpy as np
import pytest

from EasyFEA import Models
from EasyFEA.FEM._linalg import FeArray
from EasyFEA.Models import _kelvin
from EasyFEA.Models.Elastic._laws import (
    Anisotropic,
    Isotropic,
    Orthotropic,
    TransverselyIsotropic,
)

E, nu = 210000.0, 0.3


def _fe(vec) -> FeArray.FeArrayALike:
    """A (1, 1, ...) field holding one value — the same code path assembly uses."""
    return FeArray.asfearray(np.asarray(vec, dtype=float)[np.newaxis, np.newaxis])


def _val(field) -> float:
    """The single value out of a (1, 1) field."""
    return float(np.asarray(field).ravel()[0])


def _elastic_laws() -> dict:
    """One 3D instance of each shipped elastic law."""
    isot = Isotropic(3, E=E, v=nu)
    return {
        "Isotropic": isot,
        "TransverselyIsotropic": TransverselyIsotropic(
            3, El=E, Et=E / 2, Gl=E / 3, vl=0.3, vt=0.2
        ),
        "Orthotropic": Orthotropic(
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
        ),
        "Anisotropic": Anisotropic(3, isot.C, useVoigtNotation=False),
    }


# ----------------------------------------------
# Kelvin algebra
# ----------------------------------------------


def test_kelvin_trace_and_split():
    """tr, spherical and deviatoric parts of a Kelvin vector."""
    A = _fe([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    assert np.allclose(np.asarray(_kelvin.Trace(A)), 9.0)
    assert np.allclose(np.asarray(_kelvin.Spherical(A))[0, 0], [3, 3, 3, 0, 0, 0])
    assert np.allclose(np.asarray(_kelvin.Trace(_kelvin.Deviator(A))), 0.0)
    assert np.allclose(
        np.asarray(_kelvin.Spherical(A) + _kelvin.Deviator(A)), np.asarray(A)
    )


def test_kelvin_idev_projects():
    """IDEV is the deviatoric projector, and it is idempotent."""
    A = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    assert np.allclose(_kelvin.IDEV @ A, np.asarray(_kelvin.Deviator(_fe(A)))[0, 0])
    assert np.allclose(_kelvin.IDEV @ _kelvin.IDEV, _kelvin.IDEV)


# ----------------------------------------------
# Elastic free energy
# ----------------------------------------------


@pytest.mark.parametrize("law", list(_elastic_laws()))
def test_sigma_is_C_eps(law: str):
    """sigma = dpsi/deps = C:eps, for every shipped elastic law."""
    elastic = _elastic_laws()[law]
    behaviour = Models.Behaviour(3, elastic)
    eps = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])

    sig = np.asarray(behaviour.Compute_sigma(_fe(eps)))[0, 0]

    assert np.allclose(sig, elastic.C @ eps)


@pytest.mark.parametrize("law", list(_elastic_laws()))
def test_psi_is_half_eps_C_eps(law: str):
    """psi = 1/2 eps:C:eps, computed directly."""
    elastic = _elastic_laws()[law]
    behaviour = Models.Behaviour(3, elastic)
    eps = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])

    psi = _val(behaviour.Compute_psi(_fe(eps))[0, 0])

    assert np.isclose(psi, 0.5 * eps @ elastic.C @ eps)


def test_psi_is_the_potential_of_sigma():
    """dpsi/deps == sigma, by central differences — the invariant the framework rests on."""
    behaviour = Models.Behaviour(3, Isotropic(3, E=E, v=nu))
    eps = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
    h = 1e-9

    dpsi = np.array(
        [
            (
                _val(behaviour.Compute_psi(_fe(eps + d))[0, 0])
                - _val(behaviour.Compute_psi(_fe(eps - d))[0, 0])
            )
            / (2 * h)
            for d in np.eye(6) * h
        ]
    )
    sig = np.asarray(behaviour.Compute_sigma(_fe(eps)))[0, 0]

    assert np.allclose(dpsi, sig, rtol=1e-6)


def test_uniaxial_stress_is_E_times_strain():
    """Pull on x with the lateral strains set to -nu*eps: sigma_xx = E*eps_xx, and nothing else."""
    behaviour = Models.Behaviour(3, Isotropic(3, E=E, v=nu))
    eps_xx = 1e-3
    eps = np.array([eps_xx, -nu * eps_xx, -nu * eps_xx, 0.0, 0.0, 0.0])

    sig = np.asarray(behaviour.Compute_sigma(_fe(eps)))[0, 0]

    assert np.isclose(sig[0], E * eps_xx)
    assert np.allclose(sig[1:], 0.0, atol=1e-9)


# ----------------------------------------------
# Integrate
# ----------------------------------------------


def test_integrate_3d_returns_C_as_the_tangent():
    """With no internal variables the tangent is C itself, and the state stays empty."""
    elastic = Isotropic(3, E=E, v=nu)
    behaviour = Models.Behaviour(3, elastic)
    eps = _fe([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])

    sig, C_alg, z, converged = behaviour.Integrate(eps)

    assert np.allclose(np.asarray(C_alg)[0, 0], elastic.C)
    assert z.shape == (1, 1, 0)
    assert converged.all()
    assert np.allclose(np.asarray(sig)[0, 0], elastic.C @ np.asarray(eps)[0, 0])


def test_integrate_tangent_matches_central_difference():
    """C_alg == dsigma/deps by central differences — catches a wrong tangent immediately."""
    behaviour = Models.Behaviour(3, Isotropic(3, E=E, v=nu))
    eps = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])
    h = 1e-8

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d), withTangent=False)[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d), withTangent=False)[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps))[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-6)


def test_plane_strain_zeroes_the_out_of_plane_strain():
    """Plane strain: eps_zz = 0, so sigma_zz = nu/(1-nu) * (sigma_xx + sigma_yy)."""
    elastic = Isotropic(3, E=E, v=nu)
    behaviour = Models.Behaviour(2, elastic, planeStress=False)
    eps2d = np.array([1e-3, 5e-4, 2e-4])

    sig, C_alg, _, _ = behaviour.Integrate(_fe(eps2d))
    eps6 = np.zeros(6)
    eps6[[0, 1, 5]] = eps2d
    sig6 = elastic.C @ eps6

    assert np.allclose(np.asarray(sig)[0, 0], sig6[[0, 1, 5]])
    assert np.allclose(np.asarray(C_alg)[0, 0], elastic.C[np.ix_([0, 1, 5], [0, 1, 5])])


# ----------------------------------------------
# Hardening
# ----------------------------------------------

SIGMA_Y, H = 250.0, 2000.0


def test_linear_hardening_closed_form():
    """psi = 1/2 H a^2, R = H a, dR = H."""
    law = Models.IsotropicHardening.Linear(H)
    a = _fe(0.01)

    assert np.isclose(_val(law.psi(a)), 0.5 * H * 0.01**2)
    assert np.isclose(_val(law.R(a)), H * 0.01)
    assert np.isclose(_val(law.dR(a)), H)


@pytest.mark.parametrize(
    "law",
    [
        Models.IsotropicHardening.Linear(H),
        Models.IsotropicHardening.Voce(150.0, 30.0),
        Models.IsotropicHardening.Swift(600.0, 0.2),
    ],
    ids=["Linear", "Voce", "Swift"],
)
def test_hardening_R_is_the_derivative_of_psi(law):
    """R == dpsi/dalpha and dR == d2psi/dalpha2, by central differences."""
    a, h = 0.01, 1e-7

    dpsi = (_val(law.psi(_fe(a + h))) - _val(law.psi(_fe(a - h)))) / (2 * h)
    d2psi = (_val(law.R(_fe(a + h))) - _val(law.R(_fe(a - h)))) / (2 * h)

    assert np.isclose(dpsi, _val(law.R(_fe(a))), rtol=1e-6)
    assert np.isclose(d2psi, _val(law.dR(_fe(a))), rtol=1e-6)


def test_hardening_starts_from_zero():
    """R(0) = 0 — the initial yield stress belongs to the surface, not to the hardening."""
    for law in (
        Models.IsotropicHardening.Linear(H),
        Models.IsotropicHardening.Voce(150.0, 30.0),
        Models.IsotropicHardening.Swift(600.0, 0.2),
    ):
        assert np.isclose(_val(law.R(_fe(0.0))), 0.0, atol=1e-12)


# ----------------------------------------------
# Yield surfaces
# ----------------------------------------------


def test_vonmises_yields_at_sigma_y():
    """Uniaxial stress at sigma_y sits exactly on the surface."""
    surface = Models.Yield.VonMises(SIGMA_Y)
    sig = _fe([SIGMA_Y, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert np.isclose(_val(surface.f(sig, _fe(0.0))), 0.0)
    assert np.isclose(_val(Models.Yield.Svm(sig)), SIGMA_Y)


def test_vonmises_normal_is_deviatoric_and_normalized():
    """N = [1, -1/2, -1/2, 0, 0, 0] under uniaxial tension, with N:N = 3/2."""
    surface = Models.Yield.VonMises(SIGMA_Y)
    sig = _fe([SIGMA_Y, 0.0, 0.0, 0.0, 0.0, 0.0])

    N = np.asarray(surface.N(sig, _fe(0.0)))[0, 0]

    assert np.allclose(N, [1.0, -0.5, -0.5, 0.0, 0.0, 0.0])
    assert np.isclose(N @ N, 1.5)


def test_vonmises_normal_is_finite_at_the_apex():
    """d(norm)/dx does not exist at zero stress; the surface clamps instead of returning NaN."""
    surface = Models.Yield.VonMises(SIGMA_Y)
    N = np.asarray(surface.N(_fe(np.zeros(6)), _fe(0.0)))

    assert np.all(np.isfinite(N))


def test_hill_defaults_to_von_mises():
    """Hill with F=G=H=1/2, L=M=N=3/2 is von Mises exactly."""
    hill = Models.Yield.Hill(SIGMA_Y)
    mises = Models.Yield.VonMises(SIGMA_Y)
    rng = np.random.default_rng(0)
    sig = _fe(rng.normal(0.0, 200.0, 6))
    R = _fe(0.0)

    assert np.isclose(_val(hill.f(sig, R)), _val(mises.f(sig, R)))
    assert np.allclose(np.asarray(hill.N(sig, R)), np.asarray(mises.N(sig, R)))
    assert np.allclose(np.asarray(hill.dNdSig(sig)), np.asarray(mises.dNdSig(sig)))


def test_hill_is_anisotropic():
    """Off-default coefficients yield at different stresses along x and y."""
    hill = Models.Yield.Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4)

    fx = _val(hill.f(_fe([SIGMA_Y, 0, 0, 0, 0, 0]), _fe(0.0)))
    fy = _val(hill.f(_fe([0, SIGMA_Y, 0, 0, 0, 0]), _fe(0.0)))

    assert not np.isclose(fx, fy)


@pytest.mark.parametrize(
    "surface",
    [
        Models.Yield.VonMises(SIGMA_Y),
        Models.Yield.DruckerPrager(SIGMA_Y, 0.2),
        Models.Yield.Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4),
    ],
    ids=["VonMises", "DruckerPrager", "Hill"],
)
def test_dNdSig_matches_central_difference(surface):
    """dNdSig == dN/dsigma — the piece the local Jacobian relies on."""
    rng = np.random.default_rng(0)
    sig = rng.normal(0.0, 200.0, 6)
    h = 1e-4

    dN_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        Np = np.asarray(surface.N(_fe(sig + d), _fe(0.0)))[0, 0]
        Nm = np.asarray(surface.N(_fe(sig - d), _fe(0.0)))[0, 0]
        dN_fd[:, j] = (Np - Nm) / (2 * h)

    dNdSig = np.asarray(surface.dNdSig(_fe(sig)))[0, 0]

    assert np.allclose(dNdSig, dN_fd, atol=1e-8)


# ----------------------------------------------
# Plasticity
# ----------------------------------------------


def _uniaxial(behaviour, eps_xx: float, z=None):
    """One strain-controlled step, lateral strains free (solved for zero lateral stress)."""
    eps = np.zeros(6)
    eps[0] = eps_xx
    # solve the 5 free components so that only sigma_xx is non-zero
    for _ in range(50):
        sig, C, z_new, ok = behaviour.Integrate(_fe(eps), z)
        r = np.asarray(sig)[0, 0][1:]
        if np.max(np.abs(r)) < 1e-9:
            break
        Csub = np.asarray(C)[0, 0][1:, 1:]
        eps[1:] -= np.linalg.solve(Csub, r)
    return np.asarray(sig)[0, 0], z_new, ok


def test_uniaxial_bar_matches_the_closed_form():
    """sigma = sigma_y + H*eps_p once yielding, with eps_p = eps - sigma/E."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    eps_y = SIGMA_Y / E

    # elastic: below yield the answer is Hooke's law
    sig, _, ok = _uniaxial(behaviour, 0.5 * eps_y)
    assert ok.all()
    assert np.isclose(sig[0], 0.5 * SIGMA_Y, rtol=1e-9)

    # plastic: the closed form for linear isotropic hardening in uniaxial tension
    eps_xx = 5 * eps_y
    sig, _, ok = _uniaxial(behaviour, eps_xx)
    assert ok.all()
    # sigma = sigma_y + H*eps_p with eps_p = eps - sigma/E  =>  sigma = E(sigma_y + H eps)/(E + H)
    assert np.isclose(sig[0], E * (SIGMA_Y + H * eps_xx) / (E + H), rtol=1e-8)


def test_perfect_plasticity_caps_the_stress():
    """With no hardening the stress never exceeds sigma_y."""
    behaviour = Models.Behaviour(
        3, Isotropic(3, E=E, v=nu), yieldSurface=Models.Yield.VonMises(SIGMA_Y)
    )
    sig, _, ok = _uniaxial(behaviour, 20 * SIGMA_Y / E)

    assert ok.all()
    assert np.isclose(sig[0], SIGMA_Y, rtol=1e-8)


def test_unloading_is_elastic():
    """After yielding, unloading follows E again and leaves the plastic strain behind."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    eps_y = SIGMA_Y / E

    sig1, z1, _ = _uniaxial(behaviour, 5 * eps_y)
    sig2, z2, _ = _uniaxial(behaviour, 4 * eps_y, z1)

    assert np.isclose(sig1[0] - sig2[0], E * eps_y, rtol=1e-6)
    assert np.allclose(np.asarray(z1), np.asarray(z2))  # nothing flowed on the way down


@pytest.mark.parametrize(
    "hardening",
    [
        Models.IsotropicHardening.Linear(H),
        Models.IsotropicHardening.Voce(150.0, 30.0),
        Models.IsotropicHardening.Swift(600.0, 0.2),
    ],
    ids=["Linear", "Voce", "Swift"],
)
def test_consistency_condition_holds(hardening):
    """A flowing point lands exactly on the surface: f(sigma, R(alpha)) = 0."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=hardening,
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    surface = Models.Yield.VonMises(SIGMA_Y)

    sig, z, ok = _uniaxial(behaviour, 20 * SIGMA_Y / E)
    alpha = np.asarray(z)[0, 0, 6]

    assert ok.all()
    assert alpha > 0.0
    f = _val(surface.f(_fe(sig), hardening.R(_fe(alpha))))
    assert abs(f) < 1e-7 * SIGMA_Y


@pytest.mark.parametrize(
    "surface",
    [
        Models.Yield.VonMises(SIGMA_Y),
        Models.Yield.DruckerPrager(SIGMA_Y, 0.2),
        Models.Yield.Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4),
    ],
    ids=["VonMises", "DruckerPrager", "Hill"],
)
def test_plastic_tangent_matches_central_difference(surface):
    """C_alg == dsigma/deps through the local solve — this is what catches a wrong dr/dz."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=surface,
    )
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d))[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d))[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps))[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("law", list(_elastic_laws()))
def test_plasticity_works_with_any_elastic_law(law: str):
    """Anisotropic elasticity flows correctly, and its tangent is right.

    The flow normal is recomputed inside the local Newton, so nothing here assumes C is
    isotropic. A frozen normal would converge quadratically onto a wrong stress with a tangent
    consistent with it, so the central-difference check is what actually catches that.
    """
    behaviour = Models.Behaviour(
        3,
        _elastic_laws()[law],
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    sig, C_alg, z, ok = behaviour.Integrate(_fe(eps))
    assert ok.all()
    assert np.asarray(z)[0, 0, 6] > 0.0  # it really did flow

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d))[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d))[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    assert np.allclose(np.asarray(C_alg)[0, 0], C_fd, rtol=1e-5, atol=1e-2)


# ----------------------------------------------
# Viscoplasticity and creep
# ----------------------------------------------


@pytest.mark.parametrize(
    "law",
    [
        Models.ViscoPlastic.Norton(1e-3, 1.0, 250.0),
        Models.ViscoPlastic.Norton(1e-3, 3.0, 250.0),
        Models.ViscoPlastic.Perzyna(100.0, 2.0, 250.0),
    ],
    ids=["Norton n=1", "Norton n=3", "Perzyna n=2"],
)
def test_rate_inverse_round_trips(law):
    """inverse(rate(f)) == f — the two directions of the same law."""
    f = 30.0

    assert np.isclose(_val(law.inverse(law.rate(_fe(f)))), f, rtol=1e-10)


@pytest.mark.parametrize(
    "law",
    [
        Models.ViscoPlastic.Norton(1e-3, 1.0, 250.0),
        Models.ViscoPlastic.Norton(1e-3, 3.0, 250.0),
        Models.ViscoPlastic.Perzyna(100.0, 2.0, 250.0),
    ],
    ids=["Norton n=1", "Norton n=3", "Perzyna n=2"],
)
def test_rate_dinverse_matches_central_difference(law):
    """dinverse == d(inverse)/d(gdot) — the piece the local Jacobian needs."""
    g, h = 1e-4, 1e-9

    fd = (_val(law.inverse(_fe(g + h))) - _val(law.inverse(_fe(g - h)))) / (2 * h)

    assert np.isclose(fd, _val(law.dinverse(_fe(g))), rtol=1e-5)


def _viscoplastic(rate):
    return Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        rate=rate,
    )


def test_rate_dependent_needs_a_time_increment():
    """dt = 0 is rejected rather than silently dividing by zero."""
    behaviour = _viscoplastic(Models.ViscoPlastic.Norton(1e-3, 1.0, SIGMA_Y))
    eps = _fe([3e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0])

    with pytest.raises(AssertionError, match="positive time increment"):
        behaviour.Integrate(eps, dt=0.0)


def test_relaxation_decays_towards_the_surface():
    """Hold the strain: the overstress bleeds off and the stress falls towards sigma_y."""
    behaviour = _viscoplastic(Models.ViscoPlastic.Norton(1e-2, 1.0, SIGMA_Y))
    eps = _fe([3e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0])

    z, history = None, []
    for _ in range(40):
        sig, _, z, ok = behaviour.Integrate(eps, z, dt=1.0)
        assert ok.all()
        history.append(np.asarray(sig)[0, 0, 0])

    assert history[-1] < history[0]  # it relaxed
    assert np.all(np.diff(history) <= 1e-9)  # monotonically


def test_faster_flow_relaxes_further():
    """A larger fluidity bleeds more overstress off in the same time."""
    eps = _fe([3e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0])

    def after_one_step(A):
        sig, _, _, ok = _viscoplastic(
            Models.ViscoPlastic.Norton(A, 1.0, SIGMA_Y)
        ).Integrate(eps, dt=1.0)
        assert ok.all()
        return np.asarray(sig)[0, 0, 0]

    assert after_one_step(1e-1) < after_one_step(1e-3)


def test_rate_independent_limit():
    """A very fast rate law reproduces rate-independent plasticity."""
    eps = _fe([3e-3, -5e-4, -5e-4, 0.0, 0.0, 0.0])

    sig_visc, _, _, ok = _viscoplastic(
        Models.ViscoPlastic.Norton(1e8, 1.0, SIGMA_Y)
    ).Integrate(eps, dt=1.0)
    assert ok.all()

    plastic = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    sig_plas = plastic.Integrate(eps)[0]

    assert np.allclose(np.asarray(sig_visc), np.asarray(sig_plas), rtol=1e-6)


def test_viscoplastic_tangent_matches_central_difference():
    """C_alg is right with a rate law too — the dr_f/ddGamma term is the only new piece."""
    behaviour = _viscoplastic(Models.ViscoPlastic.Norton(1e-2, 2.0, SIGMA_Y))
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d), dt=1.0)[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d), dt=1.0)[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps), dt=1.0)[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-5, atol=1e-2)


# ----------------------------------------------
# Viscoelasticity
# ----------------------------------------------

EPS_V = np.array([1e-3, -2e-4, 3e-4, 1e-4, -5e-5, 2e-4])


def _viscoelastic(*branches):
    return Models.Behaviour(3, Isotropic(3, E=E, v=nu), branches=branches)


def test_branch_fractions_must_leave_an_equilibrium_spring():
    """Fractions summing to 1 would leave no long-term stiffness."""
    with pytest.raises(AssertionError, match="sum to less than 1"):
        _viscoelastic(
            Models.ViscoElastic.Maxwell(0.6, 1.0), Models.ViscoElastic.Maxwell(0.4, 2.0)
        )


def test_glassy_response_is_the_full_stiffness():
    """dt = 0: no time passes, the dashpots are rigid, and the response is C."""
    behaviour = _viscoelastic(Models.ViscoElastic.Maxwell(0.3, 1.0))
    C = Isotropic(3, E=E, v=nu).C

    sig, C_alg, _, ok = behaviour.Integrate(_fe(EPS_V), dt=0.0)

    assert ok.all()
    assert np.allclose(np.asarray(sig)[0, 0], C @ EPS_V)
    assert np.allclose(np.asarray(C_alg)[0, 0], C)


def test_fully_relaxed_response_is_the_equilibrium_stiffness():
    """After many time constants only (1 - sum g)·C is left."""
    g = 0.3
    behaviour = _viscoelastic(Models.ViscoElastic.Maxwell(g, 1.0))
    C = Isotropic(3, E=E, v=nu).C
    eps = _fe(EPS_V)

    z = None
    for _ in range(200):
        sig, _, z, ok = behaviour.Integrate(eps, z, dt=1.0)
        assert ok.all()

    assert np.allclose(np.asarray(sig)[0, 0], (1 - g) * (C @ EPS_V), rtol=1e-6)


def test_relaxation_matches_the_backward_euler_closed_form():
    """sigma_n = C:eps [(1-g) + g (1 + dt/tau)^-n], exactly, for a held strain."""
    g, tau, dt, nstep = 0.3, 2.0, 0.25, 12
    behaviour = _viscoelastic(Models.ViscoElastic.Maxwell(g, tau))
    C = Isotropic(3, E=E, v=nu).C
    eps = _fe(EPS_V)

    z = None
    for _ in range(nstep):
        sig, _, z, ok = behaviour.Integrate(eps, z, dt=dt)
        assert ok.all()

    factor = (1 - g) + g * (1 + dt / tau) ** (-nstep)
    assert np.allclose(np.asarray(sig)[0, 0], factor * (C @ EPS_V), rtol=1e-10)


def test_relaxation_approaches_the_exponential():
    """Refining dt converges on the continuous solution (1-g) + g exp(-t/tau)."""
    g, tau, t = 0.3, 2.0, 2.0
    behaviour = _viscoelastic(Models.ViscoElastic.Maxwell(g, tau))
    C = Isotropic(3, E=E, v=nu).C
    eps = _fe(EPS_V)

    def relaxed(nstep):
        z = None
        for _ in range(nstep):
            sig, _, z, _ = behaviour.Integrate(eps, z, dt=t / nstep)
        return np.asarray(sig)[0, 0, 0]

    exact = ((1 - g) + g * np.exp(-t / tau)) * (C @ EPS_V)[0]

    assert abs(relaxed(2000) - exact) < abs(relaxed(20) - exact) / 10


def test_two_branches_relax_independently():
    """Each branch carries its own time constant; the slow one still holds stress."""
    fast = Models.ViscoElastic.Maxwell(0.3, 0.1)
    slow = Models.ViscoElastic.Maxwell(0.3, 1000.0)
    behaviour = _viscoelastic(fast, slow)
    C = Isotropic(3, E=E, v=nu).C
    eps = _fe(EPS_V)

    z = None
    for _ in range(20):
        sig, _, z, ok = behaviour.Integrate(eps, z, dt=0.5)
        assert ok.all()

    # the fast branch has fully relaxed, the slow one has barely moved
    assert behaviour.layout.n == 12
    assert np.asarray(sig)[0, 0, 0] < (1 - fast.g) * (C @ EPS_V)[0] + 1e-6
    assert np.asarray(sig)[0, 0, 0] > (1 - fast.g - slow.g) * (C @ EPS_V)[0]


@pytest.mark.parametrize("dt", [0.5, 5.0], ids=["dt=0.5", "dt=5"])
def test_viscoelastic_tangent_matches_central_difference(dt: float):
    behaviour = _viscoelastic(
        Models.ViscoElastic.Maxwell(0.3, 1.0), Models.ViscoElastic.Maxwell(0.2, 10.0)
    )
    h = 1e-9

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(EPS_V + d), dt=dt)[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(EPS_V - d), dt=dt)[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(EPS_V), dt=dt)[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-6)


def test_viscoelastic_plastic_tangent_matches_central_difference():
    """Both mechanisms at once — this is what exercises the coupling blocks of dr/du."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        branches=[
            Models.ViscoElastic.Maxwell(0.3, 1.0),
            Models.ViscoElastic.Maxwell(0.2, 10.0),
        ],
    )
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    sig, C_alg, z, ok = behaviour.Integrate(_fe(eps), dt=1.0)
    assert ok.all()
    assert behaviour.layout.n == 6 + 1 + 12
    assert np.asarray(z)[0, 0, 6] > 0.0  # it flowed plastically as well as relaxed

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d), dt=1.0)[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d), dt=1.0)[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    assert np.allclose(np.asarray(C_alg)[0, 0], C_fd, rtol=1e-5, atol=1e-2)


def test_psi_stays_the_potential_of_sigma_with_branches():
    """dpsi/deps == sigma still holds once the free energy carries Maxwell branches."""
    behaviour = _viscoelastic(
        Models.ViscoElastic.Maxwell(0.3, 1.0), Models.ViscoElastic.Maxwell(0.2, 10.0)
    )
    _, _, z, _ = behaviour.Integrate(_fe(EPS_V), dt=1.0)
    h = 1e-9

    dpsi = np.array(
        [
            (
                _val(behaviour.Compute_psi(_fe(EPS_V + d), z))
                - _val(behaviour.Compute_psi(_fe(EPS_V - d), z))
            )
            / (2 * h)
            for d in np.eye(6) * h
        ]
    )
    sig = np.asarray(behaviour.Compute_sigma(_fe(EPS_V), z))[0, 0]

    assert np.allclose(dpsi, sig, rtol=1e-6)


# ----------------------------------------------
# Kinematic hardening
# ----------------------------------------------

C_KIN = 20000.0


def _kinematic(gamma=0.0, hardening=None):
    return Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=hardening,
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        kinematic=Models.KinematicHardening.ArmstrongFrederick(C_KIN, gamma),
    )


def test_back_stress_is_the_derivative_of_its_energy():
    """X == dpsi/dbeta, by central differences on the state."""
    behaviour = _kinematic()
    eps = _fe(np.zeros(6))
    beta = np.array([1e-3, -5e-4, -5e-4, 1e-4, 0.0, 0.0])
    B = behaviour.layout.slots["alpha0"]
    h = 1e-9

    def psi_at(b):
        z = np.zeros((1, 1, behaviour.layout.n))
        z[0, 0, B] = b
        return _val(behaviour.Compute_psi(eps, FeArray.asfearray(z)))

    dpsi = np.array(
        [(psi_at(beta + d) - psi_at(beta - d)) / (2 * h) for d in np.eye(6) * h]
    )

    z = np.zeros((1, 1, behaviour.layout.n))
    z[0, 0, B] = beta
    X = np.asarray(behaviour.Compute_back_stress(FeArray.asfearray(z)))[0, 0]

    assert np.allclose(dpsi, X, rtol=1e-6)


def _reverse_yield(behaviour, eps_max: float, nstep: int = 800):
    """Load to eps_max, then reverse; returns (stress at eps_max, stress when flow resumes)."""
    eps_y = SIGMA_Y / E

    z = None
    for e in np.linspace(eps_y, eps_max, 30):
        sig_max, z, ok = _uniaxial(behaviour, e, z)
        assert ok.all()
    alpha = np.asarray(z)[0, 0, 6]

    for e in np.linspace(eps_max, -eps_max, nstep):
        sig, z, ok = _uniaxial(behaviour, e, z)
        assert ok.all()
        if np.asarray(z)[0, 0, 6] > alpha + 1e-9:
            return sig_max[0], sig[0]
    raise AssertionError("the reversed path never yielded again")


def test_prager_keeps_the_elastic_range_at_two_sigma_y():
    """Pure kinematic hardening translates the surface, so reversal stays elastic for exactly 2*sigma_y."""
    forward, reverse = _reverse_yield(_kinematic(), 8 * SIGMA_Y / E)

    assert np.isclose(forward - reverse, 2 * SIGMA_Y, rtol=1e-2)


def test_bauschinger_effect():
    """Kinematic hardening yields earlier in reversal than isotropic hardening does."""
    kin = _kinematic()
    # under uniaxial tension a Prager back-stress hardens like an isotropic modulus of C:
    # X_xx = (2/3) C alpha, and the deviatoric projection contributes the remaining 3/2
    iso = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(C_KIN),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
    )
    eps_max = 8 * SIGMA_Y / E

    fk, rk = _reverse_yield(kin, eps_max)
    fi, ri = _reverse_yield(iso, eps_max)

    # same forward stress, but the kinematic surface has moved with it
    assert np.isclose(fk, fi, rtol=1e-6)
    assert rk > ri  # kinematic yields at a less negative stress, i.e. sooner


def test_chaboche_back_stress_saturates():
    """The recall term bounds X at 2C/(3*gamma); without it X grows without limit."""
    gamma = 200.0
    saturated = _kinematic(gamma)
    unbounded = _kinematic(0.0)
    eps_y = SIGMA_Y / E

    def back_stress(behaviour):
        z = None
        for e in np.linspace(eps_y, 60 * eps_y, 120):
            _, z, ok = _uniaxial(behaviour, e, z)
            assert ok.all()
        B = behaviour.layout.slots["alpha0"]
        return (
            np.linalg.norm(np.asarray(behaviour.Compute_back_stress(z))[0, 0]),
            np.asarray(z)[0, 0, B],
        )

    Xs, _ = back_stress(saturated)
    Xu, _ = back_stress(unbounded)

    assert Xs < Xu  # the recall term held it back
    assert Xs < 2 * C_KIN / (3 * gamma) * 1.5  # near the saturation bound


@pytest.mark.parametrize("gamma", [0.0, 200.0], ids=["Prager", "ArmstrongFrederick"])
def test_kinematic_tangent_matches_central_difference(gamma: float):
    behaviour = _kinematic(gamma, hardening=Models.IsotropicHardening.Linear(H))
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d))[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d))[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps))[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-5, atol=1e-2)


def test_chaboche_superposes_its_components():
    """X = sum_i X_i: the total back-stress is the sum of the component contributions."""
    components = [(60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0)]
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        kinematic=Models.KinematicHardening.Chaboche(*components),
    )
    assert behaviour.layout.n == 6 + 1 + 3 * 6

    rng = np.random.default_rng(0)
    z = np.zeros((1, 1, behaviour.layout.n))
    for i in range(3):
        z[0, 0, behaviour.layout.slots[f"alpha{i}"]] = rng.normal(0.0, 1e-3, 6)
    z = FeArray.asfearray(z)

    total = np.asarray(behaviour.Compute_back_stress(z))[0, 0]
    parts = sum(
        2 / 3 * C * np.asarray(z)[0, 0, behaviour.layout.slots[f"alpha{i}"]]
        for i, (C, _) in enumerate(components)
    )

    assert np.allclose(total, parts)


def test_one_component_chaboche_is_armstrong_frederick():
    """The superposition of one component is exactly the single law."""
    eps = _fe(np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4]))

    def run(kinematic):
        law = Models.Behaviour(
            3,
            Isotropic(3, E=E, v=nu),
            yieldSurface=Models.Yield.VonMises(SIGMA_Y),
            kinematic=kinematic,
        )
        sig, _, z, ok = law.Integrate(eps)
        assert ok.all()
        return np.asarray(sig), np.asarray(z)

    single = run(Models.KinematicHardening.ArmstrongFrederick(C_KIN, 200.0))
    superposed = run(Models.KinematicHardening.Chaboche((C_KIN, 200.0)))

    assert np.allclose(single[0], superposed[0])
    assert np.allclose(single[1], superposed[1])


def test_chaboche_tangent_matches_central_difference():
    """Three back-stresses couple through the flow direction; the cross blocks must be right."""
    behaviour = Models.Behaviour(
        3,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        kinematic=Models.KinematicHardening.Chaboche(
            (60000.0, 500.0), (20000.0, 100.0), (2000.0, 0.0)
        ),
    )
    eps = np.array([3e-3, -5e-4, -5e-4, 1e-4, -2e-4, 3e-4])
    h = 1e-9

    C_fd = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps + d))[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps - d))[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps))[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("plastic", [False, True], ids=["elastic", "plastic"])
def test_plane_stress_holds_sigma_zz_at_zero(plastic: bool):
    """sig_zz = 0 must hold once the material flows, not only while it is elastic.

    The elastic closed form for eps_zz is wrong past yield and leaves a large sig_zz behind —
    a silent wrong answer, since the local solve converges perfectly on it.
    """
    behaviour = Models.Behaviour(
        2,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H) if plastic else None,
        yieldSurface=Models.Yield.VonMises(SIGMA_Y) if plastic else None,
        planeStress=True,
    )
    eps2d = _fe([4e-3, -1e-3, 5e-4])

    sig, _, z, ok = behaviour.Integrate(eps2d)
    eps6 = behaviour.Compute_strain_6d(eps2d, z)
    sig6 = np.asarray(behaviour.Compute_sigma(eps6, z))[0, 0]

    assert ok.all()
    assert abs(sig6[2]) < 1e-9 * SIGMA_Y
    if plastic:
        assert np.asarray(z)[0, 0, 6] > 0.0  # it really did yield


def test_plane_stress_plastic_tangent_matches_central_difference():
    """The condensed 2D tangent, through the plane-stress solve and the local Newton."""
    behaviour = Models.Behaviour(
        2,
        Isotropic(3, E=E, v=nu),
        hardening=Models.IsotropicHardening.Linear(H),
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        planeStress=True,
    )
    eps2d = np.array([4e-3, -1e-3, 5e-4])
    h = 1e-9

    C_fd = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = h
        sigP = np.asarray(behaviour.Integrate(_fe(eps2d + d))[0])[0, 0]
        sigM = np.asarray(behaviour.Integrate(_fe(eps2d - d))[0])[0, 0]
        C_fd[:, j] = (sigP - sigM) / (2 * h)

    C_alg = np.asarray(behaviour.Integrate(_fe(eps2d))[1])[0, 0]

    assert np.allclose(C_alg, C_fd, rtol=1e-5, atol=1e-2)


def test_plane_stress_gives_zero_out_of_plane_stress():
    """Plane stress: eps_zz is solved so that sigma_zz = 0, and the tangent is condensed."""
    behaviour = Models.Behaviour(2, Isotropic(3, E=E, v=nu), planeStress=True)
    eps2d = _fe([1e-3, 5e-4, 2e-4])

    sig, C_alg, _, _ = behaviour.Integrate(eps2d)

    # the condensed 2D tangent must equal the elastic law's own plane-stress C
    C_2d = Isotropic(2, E=E, v=nu, planeStress=True).C
    assert np.allclose(np.asarray(C_alg)[0, 0], C_2d)
    assert np.allclose(np.asarray(sig)[0, 0], C_2d @ np.asarray(eps2d)[0, 0])


# ------------------------------------------------------------------------------------------
# The two solvers must agree wherever both apply. That equivalence is what makes a second
# implementation safe rather than a second thing to be wrong.
# ------------------------------------------------------------------------------------------

_EQUIVALENT = [
    (
        "VonMises + Linear",
        Models.Yield.VonMises(SIGMA_Y),
        Models.IsotropicHardening.Linear(H),
    ),
    (
        "VonMises + Voce",
        Models.Yield.VonMises(SIGMA_Y),
        Models.IsotropicHardening.Voce(150.0, 30.0),
    ),
    (
        "VonMises + Swift",
        Models.Yield.VonMises(SIGMA_Y),
        Models.IsotropicHardening.Swift(600.0, 0.2),
    ),
    ("VonMises perfect", Models.Yield.VonMises(SIGMA_Y), None),
    (
        "Hill + Linear",
        Models.Yield.Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4),
        Models.IsotropicHardening.Linear(H),
    ),
    (
        "Hill + Voce",
        Models.Yield.Hill(SIGMA_Y, F=0.7, G=0.4, H=0.6, L=1.8, M=1.2, N=1.4),
        Models.IsotropicHardening.Voce(150.0, 30.0),
    ),
]


def _anisotropy() -> dict:
    isot = Isotropic(3, E=E, v=nu)
    return {
        "Isotropic": isot,
        "Orthotropic": Orthotropic(
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
        ),
    }


@pytest.mark.parametrize("law", list(_anisotropy()))
@pytest.mark.parametrize(
    "name,surface,hardening", _EQUIVALENT, ids=[c[0] for c in _EQUIVALENT]
)
def test_the_two_solvers_agree(law, name, surface, hardening):
    """Same stress, same state, same tangent -- across surfaces, hardening laws and anisotropy."""
    elastic = _anisotropy()[law]
    kwargs = dict(yieldSurface=surface, hardening=hardening)
    fast = Models.Behaviour(3, elastic, **kwargs)
    slow = Models.Behaviour(3, elastic, solver="newton", **kwargs)
    assert fast._Behaviour__eigen is not None and slow._Behaviour__eigen is None

    rng = np.random.default_rng(0)
    eps = FeArray.asfearray(rng.normal(0.0, 4e-3, (6, 3, 6)))

    sigF, CF, zF, okF = fast.Integrate(eps)
    sigS, CS, zS, okS = slow.Integrate(eps)

    assert okF.all() and okS.all()
    scale = np.linalg.norm(np.asarray(sigS))
    assert np.linalg.norm(np.asarray(sigF) - np.asarray(sigS)) / scale < 1e-8
    assert np.max(np.abs(np.asarray(zF) - np.asarray(zS))) < 1e-10
    assert (
        np.linalg.norm(np.asarray(CF) - np.asarray(CS)) / np.linalg.norm(np.asarray(CS))
        < 1e-6
    )
    # the sample really did yield, so this is not vacuous
    assert np.max(np.asarray(zF)[..., fast.layout.slots["p"]]) > 1e-6


@pytest.mark.parametrize("planeStress", [False, True])
def test_the_two_solvers_agree_in_2d(planeStress: bool):
    """Plane stress wraps the 3D return in an outer Newton on eps_zz; both paths must survive it."""
    kwargs = dict(
        yieldSurface=Models.Yield.VonMises(SIGMA_Y),
        hardening=Models.IsotropicHardening.Linear(H),
        planeStress=planeStress,
    )
    elastic = Isotropic(3, E=E, v=nu)
    fast = Models.Behaviour(2, elastic, **kwargs)
    slow = Models.Behaviour(2, elastic, solver="newton", **kwargs)

    rng = np.random.default_rng(3)
    eps = FeArray.asfearray(rng.normal(0.0, 4e-3, (5, 4, 3)))

    sigF, CF, zF, _ = fast.Integrate(eps)
    sigS, CS, zS, _ = slow.Integrate(eps)

    assert (
        np.linalg.norm(np.asarray(sigF) - np.asarray(sigS))
        / np.linalg.norm(np.asarray(sigS))
        < 1e-7
    )
    assert (
        np.linalg.norm(np.asarray(CF) - np.asarray(CS)) / np.linalg.norm(np.asarray(CS))
        < 1e-5
    )
