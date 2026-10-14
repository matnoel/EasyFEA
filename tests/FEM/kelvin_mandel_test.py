# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.FEM import Get_Pmat, Apply_Pmat

from _numerics import isotropic_stiffness, mandel, rel_error


def _frame(dim: int, lead: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random rotations R (…, dim, dim) whose columns are the material axes."""
    rng = np.random.default_rng(dim + len(lead))
    Q, _ = np.linalg.qr(rng.standard_normal((*lead, dim, dim)))
    return Q, Q[..., :, 0], Q[..., :, 1]


def _symmetric(dim: int, lead: tuple) -> np.ndarray:
    S = np.random.default_rng(1).standard_normal((*lead, dim, dim))
    return S + np.swapaxes(S, -1, -2)


LEADS = {"()": (), "(Ne,)": (4,), "(Ne, nPg)": (4, 3)}


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("lead", LEADS.values(), ids=LEADS.keys())
def test_Get_Pmat_rotates_a_mandel_vector_like_its_tensor(dim: int, lead: tuple):
    """σ_global = Pm σ_material ⟺ S_global = R S_material Rᵀ."""
    R, axis_1, axis_2 = _frame(dim, lead)
    if dim == 3:
        # the third axis is axis_1 × axis_2
        R[..., :, 2] = np.cross(axis_1, axis_2)
    S = _symmetric(dim, lead)

    Pm = Get_Pmat(axis_1, axis_2)

    expected = mandel(R @ S @ np.swapaxes(R, -1, -2))
    assert rel_error(np.einsum("...ij,...j->...i", Pm, mandel(S)), expected) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_Get_Pmat_mandel_is_orthogonal(dim: int):
    _, axis_1, axis_2 = _frame(dim, (5,))

    Pm = Get_Pmat(axis_1, axis_2)

    identity = np.broadcast_to(np.eye(Pm.shape[-1]), Pm.shape)
    assert rel_error(Pm @ np.swapaxes(Pm, -1, -2), identity) < 1e-12


@pytest.mark.parametrize("dim", [2, 3])
def test_Get_Pmat_voigt_inverse_of_Ps_is_Pe_transposed(dim: int):
    _, axis_1, axis_2 = _frame(dim, ())

    Ps, Pe = Get_Pmat(axis_1, axis_2, useMandel=False)

    assert rel_error(Ps @ Pe.T, np.eye(Ps.shape[-1])) < 1e-12


def test_Get_Pmat_of_the_global_frame_is_the_identity():
    Pm = Get_Pmat([2, 0, 0], [0, 3, 0])

    assert np.array_equal(Pm, np.eye(6))


@pytest.mark.parametrize(
    "axis_1, axis_2",
    [([1, 0, 0], [1, 1, 0]), ([1, 0], [0, 1, 0]), ([1, 0, 0, 0], [0, 1, 0, 0])],
    ids=["not perpendicular", "shapes differ", "4d"],
)
def test_Get_Pmat_rejects_an_invalid_frame(axis_1, axis_2):
    with pytest.raises(ValueError):
        Get_Pmat(axis_1, axis_2)


def test_Apply_Pmat_leaves_an_isotropic_stiffness_unchanged():
    _, axis_1, axis_2 = _frame(3, (4,))
    P = Get_Pmat(axis_1, axis_2)
    C = isotropic_stiffness()

    C_global = Apply_Pmat(P, C)

    assert C_global.shape == (4, 6, 6)
    assert rel_error(C_global, np.broadcast_to(C, (4, 6, 6))) < 1e-12


@pytest.mark.parametrize("lead", LEADS.values(), ids=LEADS.keys())
def test_Apply_Pmat_back_to_the_material_frame_undoes_it(lead: tuple):
    _, axis_1, axis_2 = _frame(3, lead)
    P = Get_Pmat(axis_1, axis_2)
    M = np.random.default_rng(2).standard_normal((*lead, 6, 6))

    M_global = Apply_Pmat(P, M)

    assert rel_error(Apply_Pmat(P, M_global, toGlobal=False), M) < 1e-12
    assert rel_error(M_global, P @ M @ np.swapaxes(P, -1, -2)) < 1e-12


def test_Apply_Pmat_takes_a_per_point_matrix_with_a_constant_rotation():
    _, axis_1, axis_2 = _frame(3, ())
    P = Get_Pmat(axis_1, axis_2)
    M = np.random.default_rng(3).standard_normal((4, 3, 6, 6))

    assert rel_error(Apply_Pmat(P, M), P @ M @ P.T) < 1e-12


def test_Apply_Pmat_rejects_a_higher_rank_matrix():
    P = Get_Pmat([1, 0, 0], [0, 1, 0])

    with pytest.raises(Exception):
        Apply_Pmat(P, np.ones((2, 2, 2, 6, 6)))
