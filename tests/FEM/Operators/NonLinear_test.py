# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""A wrong tangent only slows Newton, never shows as a wrong answer: each one is checked against finite differences of the residual it linearizes."""

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, MatrixType, FeArray, Kinematics, Operators

# the operators integrate a hyperelastic law, and laws live in Models (rule 7 exception)
from EasyFEA.Models.HyperElastic import NeoHookean

from _numerics import (
    G3,
    H3,
    affine,
    assert_operator_tangent_matches_fd,
    element_dot,
    mandel,
    rel_error,
)
from ._reference import assert_restricts, element_mesh

NonLinear = Operators.NonLinear

TYPES = {2: ElemType.TRI3, 3: ElemType.TETRA4}


def _fields(dim: int, elemType: ElemType, seed: int = 0, layers: int = 1):
    """(groupElem, u_n, u_{n+1}): two random configurations a small step apart."""
    mesh = element_mesh(dim, elemType, layers)
    rng = np.random.default_rng(seed)
    u_n = rng.standard_normal(mesh.Nn * dim) * 0.02
    u_np1 = u_n + rng.standard_normal(mesh.Nn * dim) * 0.02
    return mesh.groupElem, u_n, u_np1


def _kinematics(groupElem, u) -> Kinematics:
    return Kinematics(groupElem, u, MatrixType.rigi)


FIBER = (1.0, 2.0, 2.0)


def _material(groupElem, eta: float = 0.0, tau: float = 0.0) -> NeoHookean:
    """Neo-Hookean, viscous under `eta`, active along `FIBER` under `tau`."""
    material = NeoHookean(groupElem.dim, K=1.0)
    material.eta = eta
    if tau:
        nPg = groupElem.Get_gauss(MatrixType.rigi).nPg
        fiber_e_pg = np.broadcast_to(FIBER, (groupElem.Ne, nPg, 3))
        material.Set_active_stress_vec(FeArray.asfearray(fiber_e_pg))
        material.active_stress = tau
    return material


def _energy_defect(groupElem, mat, u_n, u_np1, R) -> tuple[float, float]:
    """(R · Δu - ΔW, ΔW)"""
    work = element_dot(groupElem, R, u_np1 - u_n)
    wJ = np.asarray(groupElem.Get_weightedJacobian_e_pg(MatrixType.rigi))
    W_n = mat.Compute_W(_kinematics(groupElem, u_n))
    W_np1 = mat.Compute_W(_kinematics(groupElem, u_np1))
    dW = float(np.sum(wJ * np.asarray(W_np1 - W_n)))
    return work - dW, dW


# ----------------------------------------------------------------------------
# SecondPiolaKirchhoffStressTensor
# ----------------------------------------------------------------------------


def test_SecondPiolaKirchhoffStressTensor_tangent_matches_finite_differences(subtests):
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u, _ = _fields(dim, elemType)
            mat = _material(groupElem)

            def spk(u_):
                return NonLinear.SecondPiolaKirchhoffStressTensor(
                    mat, _kinematics(groupElem, u_)
                )

            assert_operator_tangent_matches_fd(spk, u, groupElem)


def test_SecondPiolaKirchhoffStressTensor_tangent_is_symmetric(subtests):
    """A hyperelastic tangent without follower loads is symmetric."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u, _ = _fields(dim, elemType)

            K, _ = NonLinear.SecondPiolaKirchhoffStressTensor(
                _material(groupElem), _kinematics(groupElem, u)
            )

            assert rel_error(K, K.transpose(0, 2, 1)) < 1e-10


def test_SecondPiolaKirchhoffStressTensor_residual_does_the_virtual_work_of_S(subtests):
    """u = G x: R · δu = area · S : sym(Fᵀ H) for δu = H x."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem = element_mesh(dim, elemType).groupElem
            kinematics = _kinematics(groupElem, affine(groupElem.coord, G3[:dim, :dim]))
            mat = _material(groupElem)

            _, R = NonLinear.SecondPiolaKirchhoffStressTensor(mat, kinematics)

            F = np.eye(3)
            F[:dim, :dim] += G3[:dim, :dim]
            H = np.zeros((3, 3))
            H[:dim, :dim] = H3[:dim, :dim]
            dE = mandel(F.T @ H, dim)
            S = np.asarray(mat.Compute_dWde(kinematics))[0, 0]
            assert (
                rel_error(
                    element_dot(groupElem, R, affine(groupElem.coord, H3[:dim, :dim])),
                    S @ np.asarray(dE),
                )
                < 1e-10
            )


# ----------------------------------------------------------------------------
# GonzalezStressTensor
# ----------------------------------------------------------------------------


def _gonzalez(groupElem, mat, u_n, u_np1, useConsistentTangent=True):
    return NonLinear.GonzalezStressTensor(
        mat,
        _kinematics(groupElem, u_n),
        _kinematics(groupElem, (u_n + u_np1) / 2),
        _kinematics(groupElem, u_np1),
        useConsistentTangent,
    )


def test_GonzalezStressTensor_tangent_matches_finite_differences(subtests):
    """Built for coefK = ½: the Jacobian ∂R/∂u_{n+1} is ½ K."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u_n, u_np1 = _fields(dim, elemType, seed=10)
            mat = _material(groupElem)

            def gonzalez(u_):
                return _gonzalez(groupElem, mat, u_n, u_)

            assert_operator_tangent_matches_fd(gonzalez, u_np1, groupElem, 0.5)


def test_GonzalezStressTensor_residual_works_the_stored_energy_increment(subtests):
    """R · Δu = ΔW: the discrete gradient conserves energy."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u_n, u_np1 = _fields(dim, elemType, seed=7)
            mat = _material(groupElem)

            _, R = _gonzalez(groupElem, mat, u_n, u_np1)

            defect, dW = _energy_defect(groupElem, mat, u_n, u_np1, R)
            assert abs(defect) < 1e-9 * abs(dW)


def test_GonzalezStressTensor_approximate_tangent_is_symmetric_with_the_same_residual(
    subtests,
):
    """Only the consistent tangent carries the rank-1 discrete-gradient term, which breaks symmetry."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u_n, u_np1 = _fields(dim, elemType, seed=11)
            mat = _material(groupElem)

            K_con, R_con = _gonzalez(groupElem, mat, u_n, u_np1, True)
            K_apx, R_apx = _gonzalez(groupElem, mat, u_n, u_np1, False)

            assert np.array_equal(R_con, R_apx)
            assert rel_error(K_apx, K_apx.transpose(0, 2, 1)) < 1e-10
            assert rel_error(K_con, K_con.transpose(0, 2, 1)) > 1e-6


# ----------------------------------------------------------------------------
# TimeQuadratureStressTensor
# ----------------------------------------------------------------------------


def _time_quadrature(groupElem, mat, u_n, u_np1, coefK, **kwargs):
    u_t = u_n + coefK * (u_np1 - u_n)
    return NonLinear.TimeQuadratureStressTensor(
        mat,
        _kinematics(groupElem, u_n),
        _kinematics(groupElem, u_t),
        _kinematics(groupElem, u_np1),
        coefK,
        **kwargs,
    )


def test_TimeQuadratureStressTensor_tangent_matches_finite_differences(subtests):
    """coefK · K = ∂R/∂u_{n+1}, with u_t = u_n + coefK Δu."""
    for dim, elemType in TYPES.items():
        groupElem, u_n, u_np1 = _fields(dim, elemType, seed=3)
        mat = _material(groupElem)
        for coefK in [0.5, 1.0]:
            for nPoints in [1, 2, 3]:
                with subtests.test(
                    elemType=str(elemType), coefK=coefK, nPoints=nPoints
                ):

                    def quadrature(u_):
                        return _time_quadrature(
                            groupElem, mat, u_n, u_, coefK, nPoints=nPoints
                        )

                    assert_operator_tangent_matches_fd(
                        quadrature, u_np1, groupElem, coefK
                    )


def test_TimeQuadratureStressTensor_converges_to_the_energy_increment_at_midpoint():
    """At midpoint the defect is only the quadrature error of the strain-path integral."""
    groupElem, u_n, u_np1 = _fields(2, ElemType.QUAD4, seed=5, layers=2)
    u_np1 = u_n + 3 * (u_np1 - u_n)
    mat = _material(groupElem)

    defects = []
    for nPoints in [1, 3, 9]:
        _, R, nPts_e = _time_quadrature(
            groupElem, mat, u_n, u_np1, 0.5, nPoints=nPoints
        )
        defect, dW = _energy_defect(groupElem, mat, u_n, u_np1, R)
        defects.append(abs(defect / dW))
        assert np.all(nPts_e == nPoints)

    assert defects[0] > 10 * defects[1] > 1e3 * defects[2]
    assert defects[2] < 1e-9


def test_TimeQuadratureStressTensor_adaptive_rule_meets_its_tolerance():
    groupElem, u_n, u_np1 = _fields(2, ElemType.QUAD4, seed=5, layers=2)
    u_np1 = u_n + 3 * (u_np1 - u_n)
    mat = _material(groupElem)

    _, R, nPts_e = _time_quadrature(
        groupElem, mat, u_n, u_np1, 0.5, nPoints=1, tol=1e-8
    )

    defect, dW = _energy_defect(groupElem, mat, u_n, u_np1, R)
    assert abs(defect) <= 1e-8 * abs(dW)
    assert nPts_e.max() > 1
    _, _, capped_e = _time_quadrature(
        groupElem, mat, u_n, u_np1, 0.5, nPoints=1, tol=1e-14, maxPoints=3
    )
    assert np.all(capped_e <= 3)


# ----------------------------------------------------------------------------
# ActiveStressTensor
# ----------------------------------------------------------------------------


def test_ActiveStressTensor_tangent_matches_finite_differences(subtests):
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u, _ = _fields(dim, elemType, seed=6)
            mat = _material(groupElem, tau=0.5)

            def active(u_):
                return NonLinear.ActiveStressTensor(mat, _kinematics(groupElem, u_))

            assert_operator_tangent_matches_fd(active, u, groupElem)


def test_ActiveStressTensor_residual_at_rest_does_the_fiber_work(subtests):
    """u = 0: R · δu = area · τ T̂ · sym(H) T̂ for δu = H x."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem = element_mesh(dim, elemType).groupElem
            tau = 0.5
            mat = _material(groupElem, tau=tau)

            _, R = NonLinear.ActiveStressTensor(
                mat, _kinematics(groupElem, np.zeros(groupElem.Ncoords * dim))
            )

            T = np.array(FIBER) / 3
            T[dim:] = 0
            H = np.zeros((3, 3))
            H[:dim, :dim] = H3[:dim, :dim]
            assert (
                rel_error(
                    element_dot(groupElem, R, affine(groupElem.coord, H3[:dim, :dim])),
                    tau * T @ H @ T,
                )
                < 1e-12
            )


def test_ActiveStressTensor_without_active_stress_returns_nothing():
    groupElem = element_mesh(3, ElemType.HEXA8).groupElem
    kinematics = _kinematics(groupElem, np.zeros(groupElem.Ncoords * 3))

    assert NonLinear.ActiveStressTensor(_material(groupElem), kinematics) == (
        None,
        None,
    )


# ----------------------------------------------------------------------------
# KelvinVoigtDamping
# ----------------------------------------------------------------------------


def test_KelvinVoigtDamping_tangent_matches_finite_differences(subtests):
    """Kgeo = ∂R/∂u at fixed velocity."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u, v = _fields(dim, elemType, seed=0)
            mat = _material(groupElem, eta=0.7)

            def damping(u_):
                return NonLinear.KelvinVoigtDamping(mat, _kinematics(groupElem, u_), v)

            assert_operator_tangent_matches_fd(damping, u, groupElem)


def test_KelvinVoigtDamping_damping_matrix_is_symmetric_positive(subtests):
    """C = η ∫ BᵀB."""
    for dim, elemType in TYPES.items():
        with subtests.test(elemType=str(elemType)):
            groupElem, u, v = _fields(dim, elemType, seed=6)

            _, _, C = NonLinear.KelvinVoigtDamping(
                _material(groupElem, eta=0.7), _kinematics(groupElem, u), v
            )

            assert rel_error(C, C.transpose(0, 2, 1)) < 1e-12
            assert np.linalg.eigvalsh(C).min() > -1e-9 * np.abs(C).max()


def test_KelvinVoigtDamping_inactive_returns_nothing(subtests):
    groupElem = element_mesh(2, ElemType.QUAD4).groupElem
    u = np.zeros(groupElem.Ncoords * 2)
    kinematics = _kinematics(groupElem, u)
    for name, eta, v in [("no viscosity", 0.0, u.copy()), ("no velocity", 0.7, None)]:
        with subtests.test(name):
            mat = _material(groupElem, eta=eta)

            assert NonLinear.KelvinVoigtDamping(mat, kinematics, v) == (None,) * 3


# ----------------------------------------------------------------------------
# FollowingPressure
# ----------------------------------------------------------------------------


@pytest.fixture
def faces():
    """Surface group of a unit cube, its displacement, and its z = 1 face elements."""
    mesh = element_mesh(3, ElemType.TETRA4, layers=2)
    groupElem = mesh.Get_list_groupElem(2)[0]
    top = groupElem.Get_Elements_Nodes(mesh.Nodes_Conditions(lambda x, y, z: z == 1))
    u = np.random.default_rng(4).standard_normal(mesh.Nn * 3) * 0.02
    return groupElem, u, top


def test_FollowingPressure_tangent_matches_finite_differences(faces):
    groupElem, u, _ = faces

    def pressure(u_):
        return NonLinear.FollowingPressure(groupElem, u_, 2.0)

    assert_operator_tangent_matches_fd(pressure, u, groupElem)


def test_FollowingPressure_on_a_closed_surface_has_no_resultant(faces):
    groupElem, u, _ = faces

    _, R = NonLinear.FollowingPressure(groupElem, u, 2.0)

    assert np.abs(R.reshape(groupElem.Ne, -1, 3).sum((0, 1))).max() < 1e-12


def test_FollowingPressure_on_a_face_pushes_pressure_times_area(faces):
    groupElem, _, top = faces
    u = np.zeros(groupElem.Ncoords * 3)

    _, R = NonLinear.FollowingPressure(groupElem, u, 2.0, top)

    resultant = R.reshape(groupElem.Ne, -1, 3).sum((0, 1))
    assert rel_error(np.abs(resultant), np.array([0, 0, 2.0])) < 1e-12
    others = np.setdiff1d(np.arange(groupElem.Ne), top)
    assert not np.any(R[others])


def test_FollowingPressure_without_pressure_or_elements_is_zero(faces):
    groupElem, u, _ = faces

    for K, R in [
        NonLinear.FollowingPressure(groupElem, u, 0.0),
        NonLinear.FollowingPressure(groupElem, u, 2.0, np.array([], dtype=int)),
    ]:
        assert not np.any(K) and not np.any(R)


# ----------------------------------------------------------------------------
# PenaltyContact
# ----------------------------------------------------------------------------

NORMAL = np.array([0.1, 1.0, 0.0]) / np.linalg.norm([0.1, 1.0])
OBSTACLE = np.array([0.0, 0.05, 0.0])
"""a plane through OBSTACLE with outward normal NORMAL: the bottom edge sits in it"""


@pytest.fixture
def edges():
    mesh = element_mesh(2, ElemType.TRI3, layers=2)
    groupElem = mesh.Get_list_groupElem(1)[0]
    u = np.random.default_rng(8).standard_normal(mesh.Nn * 2) * 1e-3
    return groupElem, u


def _gap(groupElem, u) -> tuple[FeArray, FeArray]:
    """Signed distance of the displaced Gauss points to the plane, and its normal."""
    x_e_pg = np.asarray(groupElem.Get_GaussCoordinates_e_pg(MatrixType.mass))
    u_e_pg = np.asarray(groupElem.Interpolate_e_pg(u, MatrixType.mass))
    x_e_pg[..., :2] += u_e_pg
    gap_e_pg = (x_e_pg - OBSTACLE) @ NORMAL
    normal_e_pg = np.broadcast_to(NORMAL, x_e_pg.shape)
    return FeArray.asfearray(gap_e_pg), FeArray.asfearray(normal_e_pg)


def _contact(groupElem, u, penalty=100.0, elements=None):
    return NonLinear.PenaltyContact(groupElem, penalty, *_gap(groupElem, u), elements)


def test_PenaltyContact_tangent_matches_finite_differences(edges):
    groupElem, u = edges

    assert_operator_tangent_matches_fd(lambda u_: _contact(groupElem, u_), u, groupElem)


def test_PenaltyContact_pushes_only_where_it_penetrates(edges):
    groupElem, u = edges
    gap_e_pg, _ = _gap(groupElem, u)
    penetrating = np.any(np.asarray(gap_e_pg) < 0, axis=1)

    _, R = _contact(groupElem, u)

    assert penetrating.any() and not penetrating.all()
    assert not np.any(R[~penetrating])
    # internal force against the outward push: each element's resultant R · n < 0
    resultants = R[penetrating].reshape(-1, groupElem.nPe, 2).sum(1)
    assert np.all(resultants @ NORMAL[:2] < 0)


def test_PenaltyContact_without_penalty_or_elements_is_zero(edges):
    groupElem, u = edges

    for K, R in [
        _contact(groupElem, u, penalty=0.0),
        _contact(groupElem, u, elements=np.array([], dtype=int)),
    ]:
        assert not np.any(K) and not np.any(R)


def test_PenaltyContact_restricted_to_elements(edges):
    groupElem, u = edges
    _, R = _contact(groupElem, u)
    hit = np.flatnonzero(np.any(R, axis=1))

    assert_restricts(
        lambda elements: _contact(groupElem, u, elements=elements)[1], hit[:1]
    )
