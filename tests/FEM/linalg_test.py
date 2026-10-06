# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.FEM._linalg import FeArray, Transpose, Trace, Det, Inv, TensorProd


@pytest.fixture
def setup_matrices() -> list[np.ndarray]:

    mat0 = np.ones((1, 1))
    mat1 = np.eye(3)
    mat2 = mat1[np.newaxis, :, :].repeat(10, axis=0)
    mat3 = np.random.random((10, 4, 1, 1))
    mat4 = np.random.random((10, 4, 2, 2))
    mat5 = np.random.random((10, 4, 3, 3))
    mat6 = np.random.random((10, 4, 3, 5, 5))

    list_Mat = [mat0, mat1, mat2, mat3, mat4, mat5, mat6]

    return list_Mat


def Check(mat: np.ndarray, matVerif: np.ndarray):

    diff = mat - matVerif

    test = np.linalg.norm(diff) / np.linalg.norm(matVerif)

    assert test < 1e-12


class TestLinalg:

    def test_Tranpose(self, setup_matrices):

        mat0, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Transpose(mat0), mat0)
        Check(Transpose(mat1), mat1)
        Check(Transpose(mat2), mat2)
        Check(Transpose(mat3), np.transpose(mat3, (0, 1, 3, 2)))
        Check(Transpose(mat4), np.transpose(mat4, (0, 1, 3, 2)))
        Check(Transpose(mat5), np.transpose(mat5, (0, 1, 3, 2)))
        Check(Transpose(mat6), np.transpose(mat6, (0, 1, 2, 4, 3)))

    def test_Trace(self, setup_matrices):

        _, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Trace(mat1), np.trace(mat1, axis1=-2, axis2=-1))
        Check(Trace(mat2), np.trace(mat2, axis1=-2, axis2=-1))
        Check(Trace(mat3), np.trace(mat3, axis1=-2, axis2=-1))
        Check(Trace(mat4), np.trace(mat4, axis1=-2, axis2=-1))
        Check(Trace(mat5), np.trace(mat5, axis1=-2, axis2=-1))
        Check(Trace(mat6), np.trace(mat6, axis1=-2, axis2=-1))

    def test_Det(self, setup_matrices):

        mat0, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Det(mat0), mat0[0, 0])
        Check(Det(mat1), np.linalg.det(mat1))
        Check(Det(mat2), np.linalg.det(mat2))
        Check(Det(mat3), np.linalg.det(mat3))
        Check(Det(mat4), np.linalg.det(mat4))
        Check(Det(mat5), np.linalg.det(mat5))
        Check(Det(mat6), np.linalg.det(mat6))

    def test_Inv(self, setup_matrices):

        mat0, mat1, mat2 = setup_matrices[:3]

        mat3 = np.array([[2, 1], [1, 2]])

        mat4 = np.array([[4, 3, 8], [6, 2, 5], [1, 5, 9]])

        Check(Inv(mat0), 1 / mat0)
        Check(Inv(mat1), np.linalg.inv(mat1))
        Check(Inv(mat2), np.linalg.inv(mat2))
        Check(Inv(mat3), np.linalg.inv(mat3))
        Check(Inv(mat4), np.linalg.inv(mat4))


def test_a_plain_array_is_always_a_constant_tensor():
    """No operand's rank is ever in doubt: the FeArray is a field, the plain array is not.

    A plain array shaped like the field is still a constant tensor, so it multiplies out
    rather than pairing off elementwise. Code that meant a field says so with asfearray.
    """
    scalar_field = FeArray.asfearray(np.ones((4, 3)))
    plain = np.full((4, 3), 2.0)

    assert (scalar_field * plain).shape == (4, 3, 4, 3)
    assert (scalar_field * FeArray.asfearray(plain)).shape == (4, 3)

    assert (scalar_field * np.ones(6)).shape == (4, 3, 6)
    assert (scalar_field * np.ones((6, 6))).shape == (4, 3, 6, 6)
    assert (scalar_field * 2.0).shape == (4, 3)


# Ne and nPg deliberately collide with the tensor dimension, so every shape coincidence the
# rules could be tempted to read something into is exercised rather than avoided.
N = 6


def _fields():
    """scalar, vector and matrix fields on a (N, N) mesh whose tensors are also N wide"""
    rng = np.random.default_rng(0)
    return tuple(
        FeArray.asfearray(rng.random(shape))
        for shape in [(N, N), (N, N, N), (N, N, N, N)]
    )


def test_a_field_is_never_re_read_from_a_shape_coincidence():
    """A (Ne, nPg) FeArray is a scalar field even where its dimensions match a tensor's."""
    scalar, _, matrix = _fields()

    assert scalar._ndim == 0 and matrix._ndim == 2
    # Ne == nPg == N, so numpy alone would read these elementwise; the field reading wins
    assert (scalar * np.ones(N)).shape == (N, N, N)
    assert (scalar * np.ones((N, N))).shape == (N, N, N, N)
    assert (scalar * matrix).shape == (N, N, N, N)
    Check(
        np.asarray(scalar * matrix),
        np.asarray(matrix) * np.asarray(scalar)[..., None, None],
    )


def test_field_ranks_pad_and_constants_right_align():
    scalar, vector, matrix = _fields()

    assert (matrix * vector).shape == (N, N, N, N)
    assert (vector * scalar).shape == (N, N, N)
    assert (scalar[..., None, None] * matrix).shape == (N, N, N, N)
    assert (matrix - np.eye(N)).shape == (N, N, N, N)
    assert (scalar * 2.0).shape == (N, N)

    # value, not just shape: the vector scales each row of the matrix
    Check(
        np.asarray(matrix * vector),
        np.asarray(matrix) * np.asarray(vector)[:, :, None, :],
    )


def test_the_rank_rule_holds_even_when_every_dimension_collides():
    """Ne == nPg == N, so nothing can be told apart by shape; the rule decides regardless."""
    scalar, vector, _ = _fields()

    # a constant (N, N) tensor is shaped exactly like the scalar field and is still a constant
    assert (scalar * np.ones((N, N))).shape == (N, N, N, N)
    Check(
        np.asarray(scalar * np.ones((N, N))),
        np.asarray(scalar)[..., None, None] * np.ones((N, N)),
    )

    # and saying "field" gives the elementwise answer instead
    for field in (scalar, vector):
        assert (field * FeArray.asfearray(np.asarray(field))).shape == field.shape


def test_dispatched_reductions_read_the_axis_too():
    """np.mean(x, 1) reduces a Gauss-point axis; the result is not a field however it is shaped.

    numpy runs a dispatched reduction on the stripped array, so the method wrapper never sees
    it. With Ne == nPg == N the result shape matches (Ne, nPg) exactly, so only the axis says.
    """
    _, vector, matrix = _fields()

    assert not isinstance(np.mean(vector, 1), FeArray)
    assert not isinstance(np.sum(vector, axis=0), FeArray)
    assert isinstance(np.sum(matrix, axis=-1), FeArray)


def test_reductions_read_the_axis_not_the_shape():
    """A sum over tensor axes is still a field; one over elements or Gauss points is not.

    With Ne == nPg == tensor width every reduction returns a shape that *looks* like a field,
    so anything inferring from the result shape gets these wrong.
    """
    _, vector, matrix = _fields()

    assert isinstance(vector.sum(axis=-1), FeArray)
    assert isinstance(matrix.sum(axis=(2, 3)), FeArray)
    assert not isinstance(vector.sum(axis=0), FeArray)
    assert not isinstance(vector.mean(axis=1), FeArray)
    assert not isinstance(vector.sum(), FeArray)


def test_the_type_survives_exactly_where_the_fe_axes_do():
    scalar, vector, matrix = _fields()

    for kept in [
        np.einsum("...i,...i->...", vector, vector),
        np.where(scalar > 0.5, scalar, 0.0),
        np.linalg.solve(matrix + N * np.eye(N), vector[..., None]),
        np.concatenate([vector, vector], axis=-1),
        np.maximum(matrix, vector),
        matrix > scalar,
        matrix.T,
    ]:
        assert isinstance(kept, FeArray), np.shape(kept)

    for dropped in [
        np.reshape(matrix, (N * N, N, N)),
        np.ravel(vector),
        vector.reshape(N * N, N),
        vector.integrate(),
    ]:
        assert not isinstance(dropped, FeArray), np.shape(dropped)


def test_a_constant_field_keeps_its_type_against_a_mesh_field():
    """Broadcast constants meet real fields and must stay fields."""
    const = FeArray.broadcast(np.eye(3), 7, 2, tensor_shape=(3, 3))
    mesh = FeArray.asfearray(np.ones((7, 2, 3, 3)))

    assert isinstance(const @ mesh, FeArray)
    assert (const @ mesh).shape == (7, 2, 3, 3)
    # but reshaping one out of field-land must not be undone
    assert not isinstance(np.reshape(const, (14, 3, 3)), FeArray)


def test_out_and_where_do_not_recurse():
    scalar, _, matrix = _fields()

    got = np.zeros_like(matrix)
    np.multiply(matrix, scalar, out=got)
    Check(np.asarray(got), np.asarray(matrix) * np.asarray(scalar)[:, :, None, None])

    safe = np.divide(1.0, scalar, out=np.zeros_like(scalar), where=scalar > 0.5)
    assert isinstance(safe, FeArray) and safe.shape == (N, N)


def test_reflected_operands_agree():
    scalar, _, matrix = _fields()

    Check(np.asarray(np.eye(N) * matrix), np.asarray(matrix * np.eye(N)))
    Check(np.asarray(2.0 * scalar), np.asarray(scalar * 2.0))


def test_an_array_without_fe_axes_cannot_become_one():
    with pytest.raises(ValueError, match="no \\(Ne, nPg\\) axes"):
        FeArray.asfearray(np.ones(N))
    # the deliberate way to hold a constant at every Gauss point
    assert FeArray.broadcast(np.ones(N), 2, 3, tensor_shape=(N,)).shape == (2, 3, N)


def test_degenerate_fearray_is_reported():
    """A view that bypasses `__new__` without the (Ne, nPg) axes must not answer as a FeArray."""
    fe = FeArray.asfearray(np.ones((4, 3)))  # scalar field
    assert fe._ndim == 0

    with pytest.raises(ValueError, match="lost the leading"):
        np.ones(3).view(FeArray)._ndim

    # and the intended escape hatch still works
    assert np.asarray(fe)[0].shape == (3,)


def test_an_integer_on_a_fe_axis_gives_a_plain_array():
    """``sig[:, 0]`` or ``sig[e, 0]`` is no longer laid out ``(Ne, nPg, ...)``."""
    sig = FeArray.asfearray(np.arange(48.0).reshape(4, 3, 2, 2))
    mask = np.asarray(sig[:, :, 0, 0]) > 20

    for consumed in [sig[0], sig[:, 0], sig[1, 2], sig[..., 0, :, :], sig[np.int64(0)], sig[mask], sig[:, None], sig[None]]:  # fmt: skip
        assert type(consumed) is np.ndarray, np.shape(consumed)
    for kept in [
        sig[:, :, 0],
        sig[..., 0, 1],
        sig[[0, 2]],
        sig[1:3],
        sig[mask[:, 0]],
        sig[:, :, None],
        sig[..., None],
    ]:
        assert isinstance(kept, FeArray), np.shape(kept)


@pytest.mark.parametrize("nPg", [4, 3], ids=["nPg!=n", "nPg==n"])
def test_a_plain_matrix_applies_at_every_point(nPg: int):
    """``R @ u`` with a plain ``R``: a constant tensor, never summed over Gauss points."""
    rng = np.random.default_rng(0)
    R = rng.random((3, 3))
    u = FeArray.asfearray(rng.random((5, nPg, 3)))
    M = FeArray.asfearray(rng.random((5, nPg, 3, 3)))

    Ru = R @ u
    assert isinstance(Ru, FeArray) and Ru.shape == (5, nPg, 3)
    Check(np.asarray(Ru), np.einsum("ij,epj->epi", R, np.asarray(u)))
    Check(np.asarray(R @ M), np.einsum("ij,epjk->epik", R, np.asarray(M)))
    Check(np.asarray(u @ R), np.einsum("epi,ij->epj", np.asarray(u), R))


@pytest.mark.parametrize("nPg", [4, 3], ids=["nPg!=n", "nPg==n"])
def test_tensor_product_with_a_plain_operand(nPg: int):
    rng = np.random.default_rng(0)
    a = rng.random(3)
    u = FeArray.asfearray(rng.random((5, nPg, 3)))

    au = TensorProd(a, u)
    assert isinstance(au, FeArray) and au.shape == (5, nPg, 3, 3)
    Check(np.asarray(au), np.einsum("i,epj->epij", a, np.asarray(u)))
    Check(np.asarray(TensorProd(u, a)), np.einsum("epi,j->epij", np.asarray(u), a))
