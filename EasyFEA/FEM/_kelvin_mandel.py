# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Kelvin–Mandel notation: ``[xx, yy, zz, √2 yz, √2 xz, √2 xy]``, the only internal one; Voigt at input only."""

import numpy as np

from ._linalg import FeArray
from ..Utilities import _types

ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
"""Component names, in Kelvin–Mandel order."""

INDEX = {name: i for i, name in enumerate(ORDER)}
"""Component name → index in the (6,) Kelvin vector."""

IDX = {1: np.array([0]), 2: np.array([0, 1, 5]), 3: np.arange(6)}
"""Indices of the (6,) Kelvin vector kept in each dimension."""

VTK_ORDER = np.array([0, 1, 2, 5, 3, 4])
"""Reorders a (6,) Kelvin vector to VTK's ``[xx, yy, zz, xy, yz, xz]``."""

R2 = np.sqrt(2)
"""Kelvin–Mandel weight of a shear component."""

BASIS = np.array(
    [
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 1 / R2], [0, 1 / R2, 0]],
        [[0, 0, 1 / R2], [0, 0, 0], [1 / R2, 0, 0]],
        [[0, 1 / R2, 0], [1 / R2, 0, 0], [0, 0, 0]],
    ]
)
"""The six Kelvin basis tensors, (6, 3, 3), ordered as ``ORDER``."""


def Weights(dim: int) -> _types.FloatArray:
    """Kelvin weights of the ``dim`` components, ``[1, 1, √2]`` or ``[1, 1, 1, √2, √2, √2]``."""
    return np.where(IDX[dim] < 3, 1.0, R2)


def From_Voigt(C: _types.FloatArray) -> _types.FloatArray:
    """Voigt stiffness (…, 3, 3) or (…, 6, 6) → Kelvin–Mandel."""
    w = Weights(2 if C.shape[-1] == 3 else 3)
    return C * np.outer(w, w)


def Reduce(x: _types.AnyArray, dim: int, rank: int = 2) -> _types.AnyArray:
    """The ``dim`` components of a (…, 6) Kelvin vector (``rank=1``) or (…, 6, 6) matrix (``rank=2``)."""
    idx = IDX[dim]
    if rank == 1:
        return x[..., idx]
    return x[..., idx, :][..., idx]


def Vector_to_Matrix(vector: _types.FloatArray, coef=R2) -> FeArray.FeArrayALike:
    """Kelvin (…, 3) or (…, 6) vector → symmetric (…, 2, 2) or (…, 3, 3) matrix; the shear entries are divided by ``coef``."""
    vectDim = vector.shape[-1]

    assert vectDim in [3, 6], "vector must be a either (...,3) or (...,6) array."

    dim = 2 if vectDim == 3 else 3

    if isinstance(vector, FeArray):
        matrix = FeArray.zeros(*vector.shape[:2], dim, dim)
    elif isinstance(vector, np.ndarray):
        matrix = np.zeros((*vector.shape[:2], dim, dim), dtype=float)
    else:
        raise ValueError("vector must be either a FeArray or a np.ndarray.")

    for d in range(dim):
        matrix[..., d, d] = vector[..., d]

    if dim == 2:
        # [x, y, xy]
        # xy
        matrix[..., 0, 1] = matrix[..., 1, 0] = vector[..., 2] / coef

    else:
        # [x, y, z, yz, xz, xy]
        # yz
        matrix[..., 1, 2] = matrix[..., 2, 1] = vector[..., 3] / coef
        # xz
        matrix[..., 0, 2] = matrix[..., 2, 0] = vector[..., 4] / coef
        # xy
        matrix[..., 0, 1] = matrix[..., 1, 0] = vector[..., 5] / coef

    return matrix


def Matrix_to_Vector(matrix: _types.FloatArray, coef=R2) -> FeArray.FeArrayALike:
    """Symmetric (…, 2, 2) or (…, 3, 3) matrix → Kelvin (…, 3) or (…, 6) vector; the shear entries are multiplied by ``coef``."""
    matrixDim = matrix.shape[-1]

    assert matrixDim in [2, 3], "matrix must be a either (...,2,2) or (...,3,3) array."

    dim = 3 if matrixDim == 2 else 6

    if isinstance(matrix, FeArray):
        vector = FeArray.zeros(*matrix.shape[:2], dim)
    elif isinstance(matrix, np.ndarray):
        vector = np.zeros((*matrix.shape[:2], dim))
    else:
        raise ValueError("matrix must be either a FeArray or a np.ndarray.")

    if dim == 3:
        # [x, y, xy]
        vector[..., 0] = matrix[..., 0, 0]  # x
        vector[..., 1] = matrix[..., 1, 1]  # y
        vector[..., 2] = matrix[..., 0, 1] * coef  # xy

    else:
        # [x, y, z, yz, xz, xy]
        vector[..., 0] = matrix[..., 0, 0]  # x
        vector[..., 1] = matrix[..., 1, 1]  # y
        vector[..., 2] = matrix[..., 2, 2]  # z
        vector[..., 3] = matrix[..., 1, 2] * coef  # yz
        vector[..., 4] = matrix[..., 0, 2] * coef  # xz
        vector[..., 5] = matrix[..., 0, 1] * coef  # xy

    return vector


def Tensor_to_Kelvin(
    A: _types.FloatArray, orderA: int | None = None
) -> _types.FloatArray:
    """Order-2 (…, 3, 3) or order-4 (…, 3, 3, 3, 3) tensor → Kelvin–Mandel (…, 6) or (…, 6, 6); ``orderA`` inferred when None."""

    shapeA = A.shape

    if orderA is None:
        tensorShape = shapeA[2:] if isinstance(A, FeArray) else shapeA
        assert (
            np.std(tensorShape) == 0
        ), "Must have the same number of indices in all dimensions."
        orderA = len(tensorShape)

    # for xx, yy, zz, yz, xz, zy
    e = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])

    def kron(a, b):
        return 1 if a == b else 0

    if orderA == 2:
        # Aij -> AI
        assert shapeA[-2:] == (3, 3), "Must be a (3,3) array"

        A_I = np.zeros((*shapeA[:-2], 6))

        def add(i: int, j: int) -> None:  # type: ignore
            A_I[..., e[i, j]] = np.sqrt(2 - kron(i, j)) * A[..., i, j]

        [add(i, j) for i in range(3) for j in range(3)]  # type: ignore [func-returns-value]

        res = A_I

    elif orderA == 4:
        # Aijkl -> AIJ
        assert shapeA[-4:] == (3, 3, 3, 3), "Must be a (3,3,3,3) array"

        A_IJ = np.zeros((*shapeA[:-4], 6, 6))

        def add(i: int, j: int, k: int, l: int) -> None:  # type: ignore [misc]
            A_IJ[..., e[i, j], e[k, l]] = (
                np.sqrt((2 - kron(i, j)) * (2 - kron(k, l))) * A[..., i, j, k, l]
            )

        [
            add(i, j, k, l)  # type: ignore [call-arg, func-returns-value]
            for i in range(3)
            for j in range(3)
            for k in range(3)
            for l in range(3)
        ]

        res = A_IJ

    else:
        raise Exception("Not implemented.")

    if isinstance(A, FeArray):
        res = FeArray.asfearray(res)

    return res


def Normalise_axes(
    axis_1: _types.FloatArray, axis_2: _types.FloatArray
) -> tuple[_types.FloatArray, _types.FloatArray]:
    """Unit ``axis_1, axis_2``; ``ValueError`` unless same shape and perpendicular at every point."""
    axis_1 = np.asarray(axis_1, dtype=float)
    axis_2 = np.asarray(axis_2, dtype=float)
    if axis_1.shape != axis_2.shape:
        raise ValueError(
            f"Both axes must have the same shape, not {axis_1.shape} and {axis_2.shape}."
        )
    if axis_1.shape[-1] not in (2, 3) or axis_1.ndim > 3:
        raise ValueError("An axis must be a (dim,), (Ne, dim) or (Ne, nPg, dim) array.")
    axis_1 = axis_1 / np.linalg.norm(axis_1, axis=-1, keepdims=True)
    axis_2 = axis_2 / np.linalg.norm(axis_2, axis=-1, keepdims=True)
    if np.abs(np.sum(axis_1 * axis_2, axis=-1)).max() > 1e-12:
        raise ValueError("The axes must be perpendicular.")
    return axis_1, axis_2


def Get_Pmat(axis_1: _types.FloatArray, axis_2: _types.FloatArray, useMandel=True):
    """Rotation from the material frame to the global one (Chevalier 1988), (…, 3, 3) in 2D or (…, 6, 6) in 3D.

    Kelvin–Mandel ``Pm``: ``C_global = Pm C_material Pmᵀ``, ``σ_global = Pm σ_material``, ``Pm⁻¹ = Pmᵀ``. Voigt (``useMandel=False``) returns ``Ps, Pe``: ``C_global = Ps C_material Psᵀ``, ``S_global = Pe S_material Peᵀ``, ``Ps⁻¹ = Peᵀ``.
    """
    axis_1, axis_2 = Normalise_axes(axis_1, axis_2)
    frame = [axis_1, axis_2]
    if axis_1.shape[-1] == 3:
        frame.append(np.cross(axis_1, axis_2))

    dim = axis_1.shape[-1]
    # (k, …, i) -> (k, i, …)
    axes = np.moveaxis(np.stack(frame), -1, 1)
    transposeP = [*range(2, axes.ndim), 0, 1]  # (dim, dim, …) -> (…, dim, dim)

    axis_1, axis_2 = axes[0], axes[1]
    if dim == 2:
        p11, p12 = axis_1
        p21, p22 = axis_2
    elif dim == 3:
        axis_3 = axes[2]
        p11, p12, p13 = axis_1
        p21, p22, p23 = axis_2
        p31, p32, p33 = axis_3
    else:
        raise TypeError("dim error")

    # p[i, k] = axis_k[i]
    p = np.swapaxes(axes, 0, 1)

    D1 = p**2

    if dim == 2:
        A = np.array([[p11 * p21], [p12 * p22]])

        B = np.array([[p11 * p12, p21 * p22]])

        D2 = np.array([[p11 * p22 + p21 * p12]])

    elif dim == 3:
        A = np.array(
            [
                [p21 * p31, p11 * p31, p11 * p21],
                [p22 * p32, p12 * p32, p12 * p22],
                [p23 * p33, p13 * p33, p13 * p23],  # type: ignore
            ]
        )

        B = np.array(
            [
                [p12 * p13, p22 * p23, p32 * p33],  # type: ignore
                [p11 * p13, p21 * p23, p31 * p33],  # type: ignore
                [p11 * p12, p21 * p22, p31 * p32],  # type: ignore
            ]
        )

        D2 = np.array(
            [
                [p22 * p33 + p32 * p23, p12 * p33 + p32 * p13, p12 * p23 + p22 * p13],  # type: ignore
                [p21 * p33 + p31 * p23, p11 * p33 + p31 * p13, p11 * p23 + p21 * p13],  # type: ignore
                [p21 * p32 + p31 * p22, p11 * p32 + p31 * p12, p11 * p22 + p21 * p12],
            ]
        )

    if useMandel:
        cM = np.sqrt(2)
        Pmat = np.concatenate(
            (
                np.concatenate((D1, cM * A), axis=1),
                np.concatenate((cM * B, D2), axis=1),
            ),
            axis=0,
        )

        Pmat = np.transpose(Pmat, transposeP)
        return Pmat
    else:
        Ps = np.concatenate(
            (np.concatenate((D1, 2 * A), axis=1), np.concatenate((B, D2), axis=1)),
            axis=0,
        ).transpose(transposeP)

        Pe = np.concatenate(
            (np.concatenate((D1, A), axis=1), np.concatenate((2 * B, D2), axis=1)),
            axis=0,
        ).transpose(transposeP)

        return Ps, Pe


def Apply_Pmat(
    P: _types.FloatArray, M: _types.FloatArray, toGlobal=True
) -> _types.FloatArray:
    """``P M Pᵀ`` (material → global) or ``Pᵀ M P`` (``toGlobal=False``), P a Kelvin–Mandel ``Get_Pmat``; leading axes of P and M may differ (``()``, ``(Ne,)``, ``(Ne, nPg)``)."""
    assert isinstance(M, np.ndarray), "Matrix must be an array"
    assert (
        M.shape[-2:] == P.shape[-2:]
    ), "Must give an matrix of shape (e,dim,dim) or (e,p,dim,dim) or (dim,dim)"

    # Get P indices
    pDim = P.ndim
    if pDim == 2:
        pi = ""
    elif pDim == 3:
        pi = "e"
    elif pDim == 4:
        pi = "ep"

    # Get P last indices
    if toGlobal:
        i1 = "ij"
        id2 = "lk"
    else:
        i1 = "ji"
        id2 = "kl"

    # Get matrix indices
    matDim = M.ndim
    if matDim == 2:
        mi = ""
    elif matDim == 3:
        mi = "e"
    elif matDim == 4:
        mi = "ep"
    else:
        raise Exception("The matrix must be of dimension (ij) or (eij) or (epij).")

    ii = mi if matDim > pDim else pi
    newM = np.einsum(f"{pi}{i1},{mi}jk,{pi}{id2}->{ii}il", P, M, P, optimize="optimal")

    return newM
