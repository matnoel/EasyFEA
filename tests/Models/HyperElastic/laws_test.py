# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Each shipped potential is restated once in jax; its derivatives are checked against autodiff. Skipped whole when jax is absent."""

import pickle

import numpy as np
import pytest

from EasyFEA import ElemType, MatrixType
from EasyFEA.FEM import FeArray, Kinematics
from EasyFEA.Geoms import Domain, Line
from EasyFEA.Models.HyperElastic import (
    AutoDiff,
    CiarletGeymonat,
    HolzapfelOgden,
    HyperElasticPotential,
    MooneyRivlin,
    NeoHookean,
    SaintVenantKirchhoff,
    _HyperElastic,
)

from _numerics import mandel, rel_error

from ..conftest import F_3D, affine_kinematics, at_every_point

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

# every comparison below sits between 1e-16 and 1e-14; 1e-12 leaves two orders of margin
TOL = 1e-12

LMBDA, MU = 121153.0, 80769.0
E = (F_3D.T @ F_3D - np.eye(3)) / 2
I_KELVIN = mandel(np.eye(3))
SVK = SaintVenantKirchhoff(3, LMBDA, MU)


def _kinematics(dim: int, seed: int = 0) -> Kinematics:
    """Six one-point elements under a random displacement: one (Ne, nPg) in every dimension, so jax compiles each potential once."""
    if dim == 1:
        line = Line((0, 0), (1, 0), 1 / 6)
        mesh = line.Mesh_1D(ElemType.SEG2)
    elif dim == 2:
        strip = Domain((0, 0), (3, 1), 1)
        mesh = strip.Mesh_2D([], ElemType.TRI3, isOrganised=True)
    else:
        cube = Domain((0, 0), (1, 1), 1)
        mesh = cube.Mesh_Extrude(
            [],
            [0, 0, 1],
            [1],
            ElemType.TETRA4,
            isOrganised=True,
        )
    u = np.random.default_rng(seed).normal(0.0, 0.02, mesh.Nn * dim)
    return Kinematics(mesh.groupElem, u, MatrixType.rigi)


@pytest.fixture(scope="module")
def kinematics() -> Kinematics:
    """The 3D `_kinematics`, for the tests that do not vary the dimension."""
    return _kinematics(3)


@pytest.fixture(scope="module")
def affine3() -> Kinematics:
    """3D affine kinematics, whose Green-Lagrange strain is `E` above."""
    return affine_kinematics(3).kinematics


def _fibers(kinematics: Kinematics, scale: float = 1.0) -> tuple[FeArray, FeArray]:
    """In-plane fibre pairs rotating from element to element, `scale` long."""
    Ne, nPg = kinematics.Compute_J().shape
    angle = np.linspace(-np.pi / 3, np.pi / 3, Ne)[:, None] * np.ones((1, nPg))
    zero = np.zeros_like(angle)
    T1_e_pg = np.stack([np.cos(angle), np.sin(angle), zero], -1)
    T2_e_pg = np.stack([-np.sin(angle), np.cos(angle), zero], -1)
    return FeArray.asfearray(scale * T1_e_pg), FeArray.asfearray(scale * T2_e_pg)


def test_SaintVenantKirchhoff_Compute_W_is_its_closed_form(affine3: Kinematics):
    W = LMBDA / 2 * np.trace(E) ** 2 + MU * np.trace(E @ E)
    assert at_every_point(SVK.Compute_W(affine3), W) < TOL


def test_SaintVenantKirchhoff_Compute_dWde_is_its_closed_form(affine3: Kinematics):
    dW = LMBDA * np.trace(E) * I_KELVIN + 2 * MU * mandel(E)
    assert at_every_point(SVK.Compute_dWde(affine3), dW) < TOL


def test_SaintVenantKirchhoff_Compute_d2Wde_is_its_closed_form(affine3: Kinematics):
    d2W = LMBDA * np.outer(I_KELVIN, I_KELVIN) + 2 * MU * np.eye(6)
    assert at_every_point(SVK.Compute_d2Wde(affine3), d2W) < TOL


def test_HyperElastic_starts_undamped_and_passive():
    assert (SVK.eta, SVK.active_stress) == (0.0, 0.0)


def test_HyperElastic_Available_Laws_lists_every_shipped_law():
    """Every concrete law except AutoDiff, which runs a potential the user supplies."""
    assert _HyperElastic.Available_Laws() == [
        NeoHookean,
        MooneyRivlin,
        CiarletGeymonat,
        SaintVenantKirchhoff,
        HolzapfelOgden,
    ]


# same names as ComputeHyperelasticLaws.py, the sympy source of the hand-written derivatives
def _invariants(C):
    """``I1, I2, I3`` of ``C``."""
    I1 = jnp.trace(C)
    return I1, (I1**2 - jnp.trace(C @ C)) / 2, jnp.linalg.det(C)


def _reduced(I1, I2, I3):
    """``J1 = I1 I3^(-1/3)``, ``J2 = I2 I3^(-2/3)``, ``J = sqrt(I3)``."""
    return I1 * I3 ** (-1 / 3), I2 * I3 ** (-2 / 3), jnp.sqrt(I3)


K, K1, K2 = 200.0, 30.0, 8.0

# passive myocardium, Holzapfel & Ogden (2009), in Pa
HO = dict(
    C0=19.0 / 2 / 8.023,
    C1=8.023,
    C2=6157.0 / 2 / 16.026,
    C3=16.026,
    C4=827.0 / 2 / 11.12,
    C5=11.12,
    C6=72.0 / 2 / 11.436,
    C7=11.436,
    K=1e6,
    Mu1=0.0,
    Mu2=0.0,
    ks=100.0,
)
T1 = np.array([1.0, 0.0, 0.0])
T2 = np.array([0.0, 1.0, 0.0])


def _neo_hookean(C):
    J1, _, _ = _reduced(*_invariants(C))
    return K * (J1 - 3)


def _mooney_rivlin(C):
    J1, J2, J = _reduced(*_invariants(C))
    return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1) ** 2


def _ciarlet_geymonat(C):
    J1, J2, J = _reduced(*_invariants(C))
    return K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1 - jnp.log(J))


def _saint_venant_kirchhoff(C):
    I1, I2, I3 = _invariants(C)
    return (
        I1**2 * (LMBDA / 8 + MU / 4)
        - I1 * (3 * LMBDA / 4 + MU / 2)
        - I2 * MU / 2
        + 0.5 * K * (I3 - 1) ** 2
        + 9 * LMBDA / 8
        + 3 * MU / 4
    )


def _holzapfel_ogden(C, T1, T2):
    """Two fibre directions, three anisotropic invariants, and a sigmoid tension/compression switch."""
    J1, J2, J = _reduced(*_invariants(C))
    I4 = T1 @ C @ T1
    I6 = T2 @ C @ T2
    I8 = T1 @ C @ T2

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


LAWS = {
    "HolzapfelOgden": (
        _holzapfel_ogden,
        lambda dim: HolzapfelOgden(dim, T1=T1, T2=T2, **HO),
        (T1, T2),
        (0, None, None),
    ),
    "NeoHookean": (
        _neo_hookean,
        lambda dim: NeoHookean(dim, K=K),
        (),
        0,
    ),
    "MooneyRivlin": (
        _mooney_rivlin,
        lambda dim: MooneyRivlin(dim, K1=K1, K2=K2, K=K),
        (),
        0,
    ),
    "CiarletGeymonat": (
        _ciarlet_geymonat,
        lambda dim: CiarletGeymonat(dim, K1=K1, K2=K2, K=K),
        (),
        0,
    ),
    "SaintVenantKirchhoff": (
        _saint_venant_kirchhoff,
        lambda dim: SaintVenantKirchhoff(dim, lmbda=LMBDA, mu=MU, K=K),
        (),
        0,
    ),
}


C_3D = F_3D.T @ F_3D
I1, I2, I3 = (
    np.trace(C_3D),
    (np.trace(C_3D) ** 2 - np.trace(C_3D @ C_3D)) / 2,
    np.linalg.det(C_3D),
)
J1, J2, J = I1 * I3 ** (-1 / 3), I2 * I3 ** (-2 / 3), np.sqrt(I3)


def test_NeoHookean_Compute_W_is_K_J1_minus_3(affine3: Kinematics):
    assert at_every_point(NeoHookean(3, K).Compute_W(affine3), K * (J1 - 3)) < TOL


def test_MooneyRivlin_Compute_W_is_its_closed_form(affine3: Kinematics):
    W = K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1) ** 2

    assert at_every_point(MooneyRivlin(3, K1, K2, K).Compute_W(affine3), W) < TOL


def test_CiarletGeymonat_Compute_W_is_its_closed_form(affine3: Kinematics):
    W = K1 * (J1 - 3) + K2 * (J2 - 3) + K * (J - 1 - np.log(J))

    assert at_every_point(CiarletGeymonat(3, K1, K2, K).Compute_W(affine3), W) < TOL


METHODS = ["Compute_W", "Compute_dWde", "Compute_d2Wde"]
DIMS = (1, 2, 3)


def _assert_matches_autodiff(subtests, method: str):
    """`method` of every law, in every dimension, is the autodiff of its jax potential."""
    kinematics = {dim: _kinematics(dim) for dim in DIMS}
    for law, (W_point, Material, aux, in_axes) in LAWS.items():
        reference = HyperElasticPotential(W_point, in_axes)[METHODS.index(method)]
        for dim in DIMS:
            with subtests.test(law=law, dim=dim):
                actual = getattr(Material(dim), method)(kinematics[dim])
                assert rel_error(reference(kinematics[dim], *aux), actual) < TOL


def test_HyperElastic_Compute_W_matches_autodiff_of_the_jax_potential(subtests):
    _assert_matches_autodiff(subtests, "Compute_W")


def test_HyperElastic_Compute_dWde_matches_autodiff_of_the_jax_potential(subtests):
    _assert_matches_autodiff(subtests, "Compute_dWde")


def test_HyperElastic_Compute_d2Wde_matches_autodiff_of_the_jax_potential(subtests):
    _assert_matches_autodiff(subtests, "Compute_d2Wde")


def test_HyperElastic_Compute_W_is_zero_at_rest(subtests):
    kinematics = affine_kinematics(3, scale=0.0).kinematics
    for name in ("NeoHookean", "MooneyRivlin", "CiarletGeymonat"):
        with subtests.test(law=name):
            W = LAWS[name][1](3).Compute_W(kinematics)
            assert at_every_point(W, 0.0) < TOL


# ----------------------------------------------
# Active stress
# ----------------------------------------------

TAU = 0.5
FIBER = np.array([1.0, 2.0, 2.0])


def _active(kinematics: Kinematics, tau=TAU) -> _HyperElastic:
    """SVK with a fibre along `FIBER`, 3 long, under the active stress `tau`."""
    Ne, nPg = kinematics.Compute_J().shape
    material = SaintVenantKirchhoff(kinematics.dim, LMBDA, MU)
    material.Set_active_stress_vec(
        FeArray.asfearray(np.broadcast_to(FIBER, (Ne, nPg, 3)))
    )
    material.active_stress = tau
    return material


def test_HyperElastic_Set_active_stress_vec_normalizes_the_direction(kinematics):
    """|FIBER| = 3, so T^ (x) T^ = FIBER (x) FIBER / 9."""
    expected = TAU * mandel(np.outer(FIBER, FIBER) / 9)

    assert (
        at_every_point(_active(kinematics).Compute_active_stress(kinematics), expected)
        < TOL
    )


def test_HyperElastic_Compute_active_stress_is_tau_T_T_in_the_solution_dimension(
    subtests,
):
    for dim in (2, 3):
        with subtests.test(dim=dim):
            kinematics = _kinematics(dim)
            expected = TAU * mandel(np.outer(FIBER, FIBER) / 9)
            if dim == 2:
                expected = expected[[0, 1, 5]]

            active = _active(kinematics).Compute_active_stress(kinematics)
            assert at_every_point(active, expected) < TOL


def test_HyperElastic_Compute_active_stress_scales_by_a_field(kinematics, subtests):
    Ne, nPg = kinematics.Compute_J().shape
    fields = {
        "element": np.linspace(1.0, 2.0, Ne),
        "point": np.linspace(1.0, 2.0, Ne * nPg).reshape(Ne, nPg),
    }
    for per, tau in fields.items():
        with subtests.test(per=per):
            material = _active(kinematics, tau)
            active = np.asarray(material.Compute_active_stress(kinematics))
            tau_e_pg = np.broadcast_to(tau.reshape(Ne, -1), (Ne, nPg))
            expected = tau_e_pg[..., None] * mandel(np.outer(FIBER, FIBER) / 9)
            assert rel_error(active, expected) < TOL


def test_HyperElastic_Compute_active_stress_rejects_a_field_of_another_mesh(kinematics):
    Ne, _ = kinematics.Compute_J().shape
    material = _active(kinematics, np.ones(Ne + 1))

    with pytest.raises(AssertionError, match="active_stress is"):
        material.Compute_active_stress(kinematics)


def test_HyperElastic_Compute_active_stress_needs_a_direction(kinematics):
    material = SaintVenantKirchhoff(3, LMBDA, MU)
    material.active_stress = TAU

    with pytest.raises(AssertionError, match="Set_active_stress_vec"):
        material.Compute_active_stress(kinematics)


def test_HyperElastic_Compute_dWde_ignores_the_active_stress(kinematics):
    """Compute_dWde stays dW/de: an active term would do no work in a discrete gradient."""
    assert np.array_equal(
        _active(kinematics).Compute_dWde(kinematics),
        SaintVenantKirchhoff(3, LMBDA, MU).Compute_dWde(kinematics),
    )


# ----------------------------------------------
# Fibres
# ----------------------------------------------


def test_HolzapfelOgden_accepts_fibre_fields(kinematics: Kinematics, subtests):
    """The fibre directions may vary per Gauss point, as a heart mesh supplies them; `AutoDiff` carries them to its kernel as `aux`."""
    T1_e_pg, T2_e_pg = _fibers(kinematics)
    material = HolzapfelOgden(3, T1=T1_e_pg, T2=T2_e_pg, **HO)
    autodiff = AutoDiff(3, _holzapfel_ogden, (T1_e_pg, T2_e_pg))
    for method in METHODS:
        with subtests.test(method=method):
            expected = getattr(autodiff, method)(kinematics)
            assert rel_error(getattr(material, method)(kinematics), expected) < TOL


def test_HolzapfelOgden_normalizes_its_fibres(kinematics: Kinematics):
    T1_e_pg, T2_e_pg = _fibers(kinematics)
    unit = HolzapfelOgden(3, T1=T1_e_pg, T2=T2_e_pg, **HO)
    long = HolzapfelOgden(3, T1=3 * T1_e_pg, T2=3 * T2_e_pg, **HO)

    assert rel_error(long.T1, unit.T1) < TOL and rel_error(long.T2, unit.T2) < TOL
    assert rel_error(long.Compute_dWde(kinematics), unit.Compute_dWde(kinematics)) < TOL


def test_HolzapfelOgden_fibre_term_switches_off_in_compression(subtests):
    """chi(I4), a steep sigmoid, keeps the fibre energy only where the fibre is stretched."""
    material = HolzapfelOgden(3, T1=T1, T2=T2, **HO)
    without = HolzapfelOgden(3, T1=T1, T2=T2, **{**HO, "C2": 0.0})
    for scale, active in [(5.0, True), (-5.0, False)]:
        with subtests.test(scale=scale):
            stretched = affine_kinematics(3, scale=scale).kinematics
            W = material.Compute_W(stretched)
            W_without = without.Compute_W(stretched)

            assert (rel_error(W, W_without) > 1e-3) is active
            assert active or rel_error(W, W_without) < 1e-6


# ----------------------------------------------
# AutoDiff
# ----------------------------------------------


def test_HyperElasticPotential_is_the_W_dWde_d2Wde_of_a_point_potential(
    kinematics: Kinematics, subtests
):
    potential = HyperElasticPotential(_saint_venant_kirchhoff)
    material = LAWS["SaintVenantKirchhoff"][1](3)
    for reference, method in zip(potential, METHODS):
        with subtests.test(method=method):
            actual = getattr(material, method)(kinematics)
            assert rel_error(reference(kinematics), actual) < TOL


def test_AutoDiff_survives_a_pickle_round_trip(kinematics: Kinematics, subtests):
    """`Simu.Save` pickles the whole simulation, material included."""
    law = AutoDiff(3, _ciarlet_geymonat)
    reloaded = pickle.loads(pickle.dumps(law))
    shipped = CiarletGeymonat(3, K1=K1, K2=K2, K=K)
    for method in METHODS:
        with subtests.test(method=method):
            expected = getattr(shipped, method)(kinematics)
            assert rel_error(getattr(reloaded, method)(kinematics), expected) < TOL
