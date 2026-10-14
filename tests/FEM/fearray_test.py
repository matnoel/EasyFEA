# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import operator

import pytest
import numpy as np

from EasyFEA.FEM import (
    FeArray,
    Transpose,
    Trace,
    Det,
    Inv,
    TensorProd,
    Norm,
    Normalize,
)

from _numerics import rel_error

RNG = np.random.default_rng(0)

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


TENSOR_SHAPES = {"scalar": (), "vector": (3,), "matrix": (3, 3), "tensor": (3,) * 4}


def _field(tensor: str, Ne: int = 5, nPg: int = 4) -> FeArray:
    return FeArray.asfearray(RNG.random((Ne, nPg, *TENSOR_SHAPES[tensor])) + 0.5)


def _plain(tensor: str) -> np.ndarray:
    return RNG.random(TENSOR_SHAPES[tensor]) + 0.5


def _padded(value, rank: int) -> np.ndarray:
    """`value` with a field's tensor rank padded to `rank`, tensor axes right-aligned; plain arrays are left as is."""
    if not isinstance(value, FeArray):
        return np.asarray(value)
    return np.asarray(value)[
        (slice(None), slice(None)) + (None,) * (rank - value._ndim)
    ]


# ----------------------------------------------------------------------------
# FeArray
# ----------------------------------------------------------------------------


def test_FeArray_needs_the_two_fe_axes():
    assert FeArray(np.ones((2, 3))).shape == (2, 3)
    for value in [0, [0], np.ones(3)]:
        with pytest.raises(ValueError):
            FeArray(value)


OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def test_FeArray_elementwise_operators_pad_the_field_ranks(subtests):
    """A field is padded to the widest rank and a plain array is a constant tensor: tensor axes right-aligned."""
    for symbol, op in OPERATORS.items():
        for left in TENSOR_SHAPES:
            for right in TENSOR_SHAPES:
                with subtests.test(op=symbol, left=left, right=right):
                    a, b = _field(left), _plain(right)
                    rank = max(a._ndim, b.ndim)

                    for res, expected in [
                        (op(a, b), op(_padded(a, rank), b)),
                        (op(b, a), op(b, _padded(a, rank))),
                    ]:
                        assert isinstance(res, FeArray)
                        assert rel_error(np.asarray(res), expected) < 1e-12


def test_FeArray_a_plain_array_is_always_a_constant_tensor():
    """A plain array shaped like the field is still a constant tensor: code that means a field says so with asfearray."""
    scalar_field = FeArray.asfearray(np.ones((4, 3)))
    plain = np.full((4, 3), 2.0)

    assert (scalar_field * plain).shape == (4, 3, 4, 3)
    assert (scalar_field * FeArray.asfearray(plain)).shape == (4, 3)

    assert (scalar_field * np.ones(6)).shape == (4, 3, 6)
    assert (scalar_field * np.ones((6, 6))).shape == (4, 3, 6, 6)
    assert (scalar_field * 2.0).shape == (4, 3)


def test_FeArray_a_field_is_never_re_read_from_a_shape_coincidence():
    """A (Ne, nPg) FeArray is a scalar field even where its dimensions match a tensor's."""
    scalar, _, matrix = _fields()

    # Ne == nPg == N, so numpy alone would read these elementwise; the field reading wins
    assert (scalar * np.ones(N)).shape == (N, N, N)
    assert (scalar * np.ones((N, N))).shape == (N, N, N, N)
    assert (scalar * matrix).shape == (N, N, N, N)
    assert (
        rel_error(
            np.asarray(scalar * matrix),
            np.asarray(matrix) * np.asarray(scalar)[..., None, None],
        )
        < 1e-12
    )


def test_FeArray_the_rank_rule_holds_even_when_every_dimension_collides():
    """Ne == nPg == N, so nothing can be told apart by shape; the rule decides regardless."""
    scalar, vector, _ = _fields()

    # a constant (N, N) tensor is shaped exactly like the scalar field and is still a constant
    assert (scalar * np.ones((N, N))).shape == (N, N, N, N)
    assert (
        rel_error(
            np.asarray(scalar * np.ones((N, N))),
            np.asarray(scalar)[..., None, None] * np.ones((N, N)),
        )
        < 1e-12
    )

    # and saying "field" gives the elementwise answer instead
    assert (scalar * FeArray.asfearray(np.asarray(scalar))).shape == scalar.shape
    assert (vector * FeArray.asfearray(np.asarray(vector))).shape == vector.shape


def test_FeArray_dispatched_reductions_read_the_axis_too():
    """np.mean(x, 1) reduces a Gauss-point axis; the result is not a field however it is shaped."""
    _, vector, matrix = _fields()

    assert not isinstance(np.mean(vector, 1), FeArray)
    assert not isinstance(np.sum(vector, axis=0), FeArray)
    assert isinstance(np.sum(matrix, axis=-1), FeArray)


def test_FeArray_the_type_survives_exactly_where_the_fe_axes_do():
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
        vector.integrate(),
    ]:
        assert not isinstance(dropped, FeArray), np.shape(dropped)


def test_FeArray_a_broadcast_constant_keeps_its_type_against_a_mesh_field():
    const = FeArray.broadcast(np.eye(3), 7, 2, tensor_shape=(3, 3))
    mesh = FeArray.asfearray(np.ones((7, 2, 3, 3)))

    assert isinstance(const @ mesh, FeArray)
    assert (const @ mesh).shape == (7, 2, 3, 3)
    # but reshaping one out of field-land must not be undone
    assert not isinstance(np.reshape(const, (14, 3, 3)), FeArray)


def test_FeArray_out_and_where_do_not_recurse():
    scalar, _, matrix = _fields()

    got = np.zeros_like(matrix)
    np.multiply(matrix, scalar, out=got)
    expected = np.asarray(matrix) * np.asarray(scalar)[:, :, None, None]
    assert rel_error(np.asarray(got), expected) < 1e-12

    safe = np.divide(1.0, scalar, out=np.zeros_like(scalar), where=scalar > 0.5)
    assert isinstance(safe, FeArray) and safe.shape == (N, N)


def test_FeArray_reflected_operands_agree():
    scalar, _, matrix = _fields()

    assert (
        rel_error(np.asarray(np.eye(N) * matrix), np.asarray(matrix * np.eye(N)))
        < 1e-12
    )
    assert rel_error(np.asarray(2.0 * scalar), np.asarray(scalar * 2.0)) < 1e-12


def test_FeArray_an_integer_on_a_fe_axis_gives_a_plain_array():
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


MATMUL = {
    "i,i": ("vector", "vector", "...i,...i->..."),
    "i,ij": ("vector", "matrix", "...i,...ij->...j"),
    "i,ijkl": ("vector", "tensor", "...i,...ijkl->...jkl"),
    "ij,j": ("matrix", "vector", "...ij,...j->...i"),
    "ij,jk": ("matrix", "matrix", "...ij,...jk->...ik"),
    "ij,jklm": ("matrix", "tensor", "...ij,...jklm->...iklm"),
    "ijkl,l": ("tensor", "vector", "...ijkl,...l->...ijk"),
    "ijkl,lm": ("tensor", "matrix", "...ijkl,...lm->...ijkm"),
    "ijkl,lmno": ("tensor", "tensor", "...ijkl,...lmno->...ijkmno"),
}


def test_FeArray_matmul_contracts_the_inner_index(subtests):
    for name, (left, right, subscripts) in MATMUL.items():
        with subtests.test(subscripts=name):
            a, b = _field(left), _field(right)

            res = a @ b

            assert isinstance(res, FeArray)
            assert rel_error(np.asarray(res), np.einsum(subscripts, a, b)) < 1e-12


def test_FeArray_matmul_applies_a_plain_matrix_at_every_point(subtests):
    """``R @ u`` with a plain ``R``: a constant tensor, never summed over Gauss points."""
    for nPg in [4, 3]:
        with subtests.test(nPg=nPg):
            rng = np.random.default_rng(0)
            R = rng.random((3, 3))
            u = FeArray.asfearray(rng.random((5, nPg, 3)))
            M = FeArray.asfearray(rng.random((5, nPg, 3, 3)))

            Ru = R @ u
            assert isinstance(Ru, FeArray) and Ru.shape == (5, nPg, 3)
            assert (
                rel_error(np.asarray(Ru), np.einsum("ij,epj->epi", R, np.asarray(u)))
                < 1e-12
            )
            assert (
                rel_error(
                    np.asarray(R @ M), np.einsum("ij,epjk->epik", R, np.asarray(M))
                )
                < 1e-12
            )
            assert (
                rel_error(np.asarray(u @ R), np.einsum("epi,ij->epj", np.asarray(u), R))
                < 1e-12
            )


def test_FeArray_matmul_rejects_a_non_array():
    with pytest.raises(TypeError):
        _field("vector") @ [1.0, 2.0, 3.0]


def test_FeArray__shape_is_the_tensor_shape():
    assert _field("scalar")._shape == ()
    assert _field("matrix")._shape == (3, 3)


def test_FeArray__ndim_is_the_tensor_rank():
    assert [_field(t)._ndim for t in TENSOR_SHAPES] == [0, 1, 2, 4]


def test_FeArray__ndim_reports_a_view_that_lost_the_fe_axes():
    with pytest.raises(ValueError, match="lost the leading"):
        np.ones(3).view(FeArray)._ndim


def test_FeArray__align_pads_field_ranks_and_right_aligns_constants():
    scalar, vector, matrix = _fields()

    assert FeArray._align((matrix, matrix)) == (matrix, matrix)
    assert [np.shape(op) for op in FeArray._align((scalar, vector, np.eye(N)))] == [
        (N, N, 1, 1),
        (N, N, 1, N),
        (N, N),
    ]

    # value, not just shape: the vector scales each row of the matrix
    assert (
        rel_error(
            np.asarray(matrix * vector),
            np.asarray(matrix) * np.asarray(vector)[:, :, None, :],
        )
        < 1e-12
    )


def test_FeArray__lift_plain_holds_plain_operands_at_every_point():
    vector = _field("vector")
    plain = _plain("matrix")

    assert FeArray._lift_plain((vector, 2.0)) is None
    assert FeArray._lift_plain((plain, plain)) is None

    lifted, kept = FeArray._lift_plain((plain, vector))
    assert kept is vector
    assert lifted.shape == (5, 4, 3, 3)
    assert np.array_equal(np.asarray(lifted)[2, 1], plain)


def test_FeArray_T_reverses_the_tensor_axes():
    scalar, vector, matrix, tensor = (_field(t) for t in TENSOR_SHAPES)

    assert np.array_equal(scalar.T, scalar)
    assert np.array_equal(vector.T, vector)
    assert np.array_equal(matrix.T, np.asarray(matrix).transpose((0, 1, 3, 2)))
    assert np.array_equal(tensor.T, np.asarray(tensor).transpose((0, 1, 5, 4, 3, 2)))


def test_FeArray__dot_subscript_contracts_the_last_index_with_the_first():
    assert FeArray._dot_subscript(1, 1) == "...i,...i->..."
    assert FeArray._dot_subscript(2, 2) == "...ij,...jk->...ik"
    assert FeArray._dot_subscript(4, 1) == "...ijkl,...l->...ijk"


def test_FeArray__ddot_subscript_contracts_the_last_two_indices():
    assert FeArray._ddot_subscript(2, 2) == "...ij,...ij->..."
    assert FeArray._ddot_subscript(2, 4) == "...ij,...ijkl->...kl"
    assert FeArray._ddot_subscript(4, 2) == "...ijkl,...kl->...ij"


def test_FeArray_dot_contracts_the_inner_index(subtests):
    for name, (left, right, subscripts) in MATMUL.items():
        with subtests.test(subscripts=name):
            a, b = _field(left), _field(right)

            assert rel_error(np.asarray(a.dot(b)), np.einsum(subscripts, a, b)) < 1e-12


def test_FeArray_dot_holds_a_plain_operand_constant(subtests):
    for name, (left, right, subscripts) in MATMUL.items():
        with subtests.test(subscripts=name):
            a, b = _field(left), _plain(right)

            expected = np.einsum(subscripts, a, np.broadcast_to(b, (5, 4, *b.shape)))
            assert rel_error(np.asarray(a.dot(b)), expected) < 1e-12


def test_FeArray_dot_needs_two_tensors():
    for a, b in [
        (_field("scalar"), _field("vector")),
        (_field("vector"), _field("scalar")),
    ]:
        with pytest.raises(ValueError):
            a.dot(b)
    with pytest.raises(TypeError):
        _field("vector").dot([1.0, 2.0, 3.0])


DDOT = {
    "ij,ij": ("matrix", "matrix", "...ij,...ij->..."),
    "ij,ijkl": ("matrix", "tensor", "...ij,...ijkl->...kl"),
    "ijkl,kl": ("tensor", "matrix", "...ijkl,...kl->...ij"),
    "ijkl,klmn": ("tensor", "tensor", "...ijkl,...klmn->...ijmn"),
}


def test_FeArray_ddot_contracts_the_two_inner_indices(subtests):
    for name, (left, right, subscripts) in DDOT.items():
        with subtests.test(subscripts=name):
            a, b = _field(left), _field(right)

            plain = _plain(right)

            assert rel_error(np.asarray(a.ddot(b)), np.einsum(subscripts, a, b)) < 1e-12
            expected = np.einsum(subscripts, a, np.broadcast_to(plain, b.shape))
            assert rel_error(np.asarray(a.ddot(plain)), expected) < 1e-12


def test_FeArray_ddot_needs_two_matrices_at_least(subtests):
    for left, right in [
        ("vector", "matrix"),
        ("matrix", "vector"),
        ("tensor", "vector"),
    ]:
        with subtests.test(left=left, right=right):
            with pytest.raises(ValueError):
                _field(left).ddot(_field(right))


def test_FeArray__make_reducer_keeps_the_type_over_tensor_axes_only():
    """With Ne == nPg == tensor width every reduction looks like a field: only the axis says."""
    _, vector, matrix = _fields()

    assert isinstance(vector.sum(axis=-1), FeArray)
    assert isinstance(matrix.max(axis=(2, 3)), FeArray)
    assert not isinstance(vector.sum(axis=0), FeArray)
    assert not isinstance(vector.mean(axis=1), FeArray)
    assert not isinstance(vector.sum(), FeArray)
    assert not isinstance(vector.ravel(), FeArray)


def test_FeArray_reshape_keeps_the_type_with_the_fe_axes():
    matrix = _field("matrix")

    assert isinstance(matrix.reshape(5, 4, 9), FeArray)
    assert not isinstance(matrix.reshape(20, 3, 3), FeArray)


def test_FeArray_integrate_sums_over_the_gauss_points():
    values = FeArray.asfearray(np.arange(24.0).reshape(2, 3, 4))

    res = values.integrate()

    assert type(res) is np.ndarray
    assert np.array_equal(res, np.asarray(values).sum(axis=1))


def test_FeArray__get_idx_spans_elements_points_and_the_given_tensor_indices():
    matrix = FeArray.zeros(5, 4, 3, 3)

    idx = matrix._get_idx(np.array([0, 2]), np.array([1]))

    assert [i.shape for i in idx] == [
        (5, 1, 1, 1),
        (1, 4, 1, 1),
        (1, 1, 2, 1),
        (1, 1, 1, 1),
    ]


def test_FeArray__assemble_writes_the_block_at_every_point():
    matrix = FeArray.zeros(5, 4, 3, 3)

    matrix._assemble(np.array([0, 2]), np.array([1, 2]), value=1.0)

    expected = np.zeros((3, 3))
    expected[np.ix_([0, 2], [1, 2])] = 1.0
    assert np.array_equal(np.asarray(matrix), np.broadcast_to(expected, (5, 4, 3, 3)))


def test_FeArray_asfearray_views_without_copying():
    values = np.ones((2, 3, 3))

    res = FeArray.asfearray(values)

    assert isinstance(res, FeArray) and np.shares_memory(res, values)
    assert isinstance(FeArray.asfearray([[1.0, 2.0]]), FeArray)


def test_FeArray_asfearray_rejects_an_array_without_fe_axes():
    with pytest.raises(ValueError, match="no \\(Ne, nPg\\) axes"):
        FeArray.asfearray(np.ones(N))


def test_FeArray_from_pg_repeats_the_reference_values_over_the_elements():
    value_pg = np.arange(6.0).reshape(3, 2)

    res = FeArray.from_pg(value_pg, 4)

    assert res.shape == (4, 3, 2) and not res.flags.writeable
    assert all(np.array_equal(res[e], value_pg) for e in range(4))


def test_FeArray_from_e_repeats_the_element_values_over_the_points():
    value_e = np.arange(6.0).reshape(3, 2)

    res = FeArray.from_e(value_e, 4)

    assert res.shape == (3, 4, 2) and not res.flags.writeable
    assert all(np.array_equal(np.asarray(res)[:, p], value_e) for p in range(4))


# (value, tensor shape, accepted, meaning): value "" is a float, "fe:..." a FeArray;
# meaning "full" (constant or already per point), "e" per element, "pg" per point (the hole), None ill-formed
S, V, M = "", "n", "n, n"
CASES = [
    ("", S, True, "full"),
    ("Ne", S, True, "e"),
    ("Ne, nPg", S, True, "full"),
    ("fe:Ne, nPg", S, True, "full"),
    ("nPg", S, False, "pg"),
    ("Ne, 1", S, False, "full"),
    ("1, nPg", S, False, "full"),
    ("Ne, nPg + 2", S, False, None),
    ("Ne + 1", S, False, None),
    ("n", S, False, None),
    ("Ne, nPg, n", S, False, None),
    ("fe:Ne, 1", S, False, "full"),
    ("fe:Ne, nPg, n", S, False, None),
    ("n", V, True, "full"),
    ("Ne, n", V, True, "e"),
    ("Ne, nPg, n", V, True, "full"),
    ("fe:Ne, nPg, n", V, True, "full"),
    ("", V, False, None),
    ("Ne, nPg", V, False, None),
    ("nPg, n", V, False, "pg"),
    ("n + 1", V, False, None),
    ("n, n", M, True, "full"),
    ("Ne, n, n", M, True, "e"),
    ("Ne, nPg, n, n", M, True, "full"),
    ("fe:Ne, nPg, n, n", M, True, "full"),
    ("Ne", M, False, None),
    ("n", M, False, None),
    ("nPg, n, n", M, False, "pg"),
    ("Ne, 1, n, n", M, False, "full"),
    ("Ne, n, n + 1", M, False, None),
    ("fe:1, 1, n, n", M, False, "full"),
]

SIZES = {
    "regular": (5, 4, 3),
    "Ne==nPg": (4, 4, 3),
    "nPg==n": (5, 3, 3),
    "Ne==nPg==n": (3, 3, 3),
    "Ne==1": (1, 4, 3),
    "nPg==1": (5, 1, 3),
}


def _shape(spec: str, Ne: int, nPg: int, n: int) -> tuple:
    return eval(f"({spec},)", {}, dict(Ne=Ne, nPg=nPg, n=n)) if spec else ()


def test_FeArray_broadcast_is_right_or_raises(subtests):
    """A rejected value may become valid where sizes coincide; it must then be right."""
    for spec, tensor, accepted, meaning in CASES:
        for sizeName, sizes in SIZES.items():
            with subtests.test(value=f"{spec}|{tensor}", sizes=sizeName):
                Ne, nPg, n = sizes
                if meaning == "pg" and Ne == nPg:
                    pytest.skip(
                        "(nPg, ...) at Ne == nPg reads per element: the accepted hole"
                    )
                shape = _shape(spec.removeprefix("fe:"), Ne, nPg, n)
                # distinct values, so a misread axis cannot pass for the right one
                value = (
                    np.arange(1.0, np.prod(shape) + 1).reshape(shape) if shape else 2.0
                )
                if spec.startswith("fe:"):
                    value = FeArray.asfearray(value)
                t = _shape(tensor, Ne, nPg, n)

                try:
                    res = FeArray.broadcast(value, Ne, nPg, tensor_shape=t)
                except ValueError:
                    assert not accepted, f"{spec} raised at {sizes}"
                    continue
                assert accepted or sizes != SIZES["regular"], f"{spec} accepted"

                assert isinstance(res, FeArray) or (
                    isinstance(res, float) and shape == t == ()
                )
                assert np.shape(res) in [(), (Ne, nPg, *t)]
                if meaning is not None:
                    v = np.asarray(value)
                    v = (
                        v[:, None]
                        if meaning == "e"
                        else v[None] if meaning == "pg" else v
                    )
                    expected = np.broadcast_to(v, (Ne, nPg, *t))
                    assert np.array_equal(
                        np.broadcast_to(np.asarray(res), (Ne, nPg, *t)), expected
                    )


def test_FeArray_broadcast_holds_a_constant_as_a_read_only_view():
    C = np.eye(3)
    C_e_pg = FeArray.broadcast(C, 5, 4, tensor_shape=(3, 3))
    assert not C_e_pg.flags.writeable
    assert np.shares_memory(C_e_pg, C)


def test_FeArray_broadcast_requires_the_tensor_shape():
    with pytest.raises(TypeError):
        FeArray.broadcast(np.ones(5), 5, 4)  # type: ignore [call-arg]


def test_FeArray_zeros_and_ones_take_the_shape_either_way(subtests):
    for make in [FeArray.zeros, FeArray.ones]:
        with subtests.test(make=make.__name__):
            a, b = make(2, 3, 4), make((2, 3, 4))

            assert isinstance(a, FeArray) and a.shape == b.shape == (2, 3, 4)
            assert make(2, 3, dtype=int).dtype.kind == "i"


def test_FeArray_zeros_is_zero():
    assert not np.any(FeArray.zeros(2, 3))


def test_FeArray_ones_is_one():
    assert np.all(FeArray.ones(2, 3) == 1)


# ----------------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------------

MATRICES = {
    "1x1": np.ones((1, 1)) * 3,
    "eye": np.eye(3),
    "(10, 3, 3)": np.eye(3)[np.newaxis].repeat(10, axis=0),
    "fe 1x1": FeArray.asfearray(RNG.random((10, 4, 1, 1)) + 1),
    "fe 2x2": FeArray.asfearray(RNG.random((10, 4, 2, 2)) + 2 * np.eye(2)),
    "fe 3x3": FeArray.asfearray(RNG.random((10, 4, 3, 3)) + 3 * np.eye(3)),
    "fe 5x5": FeArray.asfearray(RNG.random((10, 4, 3, 5, 5)) + 5 * np.eye(5)),
}


def test_Transpose_swaps_the_last_two_axes(subtests):
    for name, mat in MATRICES.items():
        with subtests.test(mat=name):
            res = Transpose(mat)

            assert np.array_equal(res, np.swapaxes(mat, -1, -2))
            assert isinstance(res, FeArray) == isinstance(mat, FeArray)


def test_Trace_sums_the_diagonal(subtests):
    for name, mat in MATRICES.items():
        with subtests.test(mat=name):
            res = Trace(mat)

            assert rel_error(res, np.trace(mat, axis1=-2, axis2=-1)) < 1e-12
            assert isinstance(res, FeArray) == isinstance(mat, FeArray)


def test_Det_matches_numpy(subtests):
    for name, mat in MATRICES.items():
        with subtests.test(mat=name):
            res = Det(mat)

            assert rel_error(res, np.linalg.det(mat)) < 1e-12
            assert isinstance(res, FeArray) == isinstance(mat, FeArray)


def test_Inv_times_the_matrix_is_the_identity(subtests):
    for name, mat in MATRICES.items():
        with subtests.test(mat=name):
            res = Inv(mat)

            identity = np.broadcast_to(np.eye(mat.shape[-1]), mat.shape)
            assert rel_error(np.asarray(mat) @ np.asarray(res), identity) < 1e-12
            assert isinstance(res, FeArray) == isinstance(mat, FeArray)


def test_Inv_rejects_a_non_square_matrix():
    with pytest.raises(AssertionError):
        Inv(np.ones((2, 3)))


def test_TensorProd_of_vectors_is_the_outer_product():
    a, b = _field("vector"), _field("vector")

    res = TensorProd(a, b)

    assert isinstance(res, FeArray)
    assert rel_error(np.asarray(res), np.einsum("...i,...j->...ij", a, b)) < 1e-12
    assert np.array_equal(
        TensorProd(np.ones(3), np.arange(3.0)), np.ones((3, 1)) * np.arange(3.0)
    )


def test_TensorProd_of_matrices_is_the_fourth_order_product():
    A, B = _field("matrix"), _field("matrix")

    res = TensorProd(A, B)

    assert rel_error(np.asarray(res), np.einsum("...ij,...kl->...ijkl", A, B)) < 1e-12


def test_TensorProd_symmetric_of_identities_maps_a_symmetric_matrix_to_itself():
    """(I ⊗ₛ I) : S = S for S symmetric."""
    eye = np.eye(3)
    S = RNG.random((3, 3))
    S = S + S.T

    identity = TensorProd(eye, eye, symmetric=True)

    assert rel_error(np.einsum("ijkl,kl->ij", identity, S), S) < 1e-12


def test_TensorProd_with_a_plain_operand(subtests):
    for nPg in [4, 3]:
        with subtests.test(nPg=nPg):
            rng = np.random.default_rng(0)
            a = rng.random(3)
            u = FeArray.asfearray(rng.random((5, nPg, 3)))

            au = TensorProd(a, u)
            assert isinstance(au, FeArray) and au.shape == (5, nPg, 3, 3)
            assert (
                rel_error(np.asarray(au), np.einsum("i,epj->epij", a, np.asarray(u)))
                < 1e-12
            )
            assert (
                rel_error(
                    np.asarray(TensorProd(u, a)),
                    np.einsum("epi,j->epij", np.asarray(u), a),
                )
                < 1e-12
            )


def test_TensorProd_rejects_mismatched_operands():
    with pytest.raises(AssertionError):
        TensorProd(_field("vector"), _field("matrix"))
    with pytest.raises(AssertionError):
        TensorProd(np.ones(3), np.ones(4))
    with pytest.raises(AssertionError):
        TensorProd(_field("tensor"), _field("tensor"))


def test_Norm_reduces_over_tensor_axes_only():
    rng = np.random.default_rng(0)
    for t in [(), (3,), (3, 3), (2, 2, 2, 2)]:
        x = FeArray.asfearray(rng.random((5, 4, *t)))
        res = Norm(x)
        assert isinstance(res, FeArray) and res.shape == (5, 4), t
        expected = np.sqrt((np.asarray(x) ** 2).reshape(5, 4, -1).sum(-1))
        assert rel_error(np.asarray(res), expected) < 1e-12
    scalar = FeArray.asfearray(-rng.random((5, 4)))
    assert (
        rel_error(np.asarray(Norm(scalar, ord=2)), np.abs(np.asarray(scalar))) < 1e-12
    )


def test_Norm_passes_ord_and_axis_to_numpy():
    vector, matrix = _field("vector"), _field("matrix")

    assert (
        rel_error(np.asarray(Norm(vector, ord=1)), np.abs(np.asarray(vector)).sum(-1))
        < 1e-12
    )
    assert (
        rel_error(
            np.asarray(Norm(matrix, ord=np.inf)),
            np.linalg.norm(np.asarray(matrix), ord=np.inf, axis=(-2, -1)),
        )
        < 1e-12
    )
    assert (
        rel_error(
            Norm(np.asarray(vector), axis=-1),
            np.linalg.norm(np.asarray(vector), axis=-1),
        )
        < 1e-12
    )


def test_Normalize_gives_unit_slices_and_leaves_zeros():
    vector = _field("vector")
    vector[0, 0] = 0.0

    res = Normalize(vector)

    assert isinstance(res, FeArray)
    norms = np.linalg.norm(np.asarray(res), axis=-1)
    assert np.array_equal(np.asarray(res)[0, 0], np.zeros(3))
    assert rel_error(norms.ravel()[1:], 1.0) < 1e-12
    assert (
        rel_error(Normalize(np.array([[3.0, 4.0]]), axis=-1), np.array([[0.6, 0.8]]))
        < 1e-12
    )
