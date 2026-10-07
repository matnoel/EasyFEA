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
    """Projects the tensor A in Kelvin Mandel notation.

    Parameters
    ----------
    A : _types.FloatArray
        tensor A (2 or 4 order tensor)
    orderA : int, optional
        tensor order, by default None

    Returns
    -------
    _types.FloatArray
        Projected tensor
    """

    shapeA = A.shape

    if orderA is None:
        if isinstance(A, FeArray):
            shapeA = shapeA[2:]
        assert (
            np.std(shapeA) == 0
        ), "Must have the same number of indices in all dimensions."
        orderA = len(shapeA)

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


def Get_Pmat(axis_1: _types.FloatArray, axis_2: _types.FloatArray, useMandel=True):
    """Constructs Pmat to pass from the material coordinates (x,y,z) to the global coordinate (X,Y,Z) such that:\n

    if useMandel:\n
        return [Pm]\n
    else:\n
        return [Ps], [Pe]\n

    In Kelvin Mandel notation:
    --------------------------

        Sig & Eps en [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]\n
        [C_global] = [Pm] * [C_material] * [Pm]^T & [C_material] = [Pm]^T * [C_global] * [Pm]\n
        [S_global] = [Pm] * [S_material] * [Pm]^T & [S_material] = [Pm]^T * [S_global] * [Pm]\n
        Sig_global = [Pm] * Sig_material & Sig_material = [Pm]^T * Sig_global\n
        Eps_global = [Pm] * Eps_material & Eps_material = [Pm]^T * Eps_global\n

    In Voigt's notation:
    --------------------

        Sig [S11, S22, S33, S23, S13, S12]\n
        Eps [E11, E22, E33, 2*E23, 2*E13, 2*E12]\n
        [C_global] = [Ps] * [C_material] * [Ps]^T & [C_material] = [Pe]^T * [C_global] * [Pe]\n
        S_global = [Pe] * [S_material] * [Pe]^T & [S_material] = [Ps]^T * S_global * [Ps]\n
        Sig_global = [Ps] * Sig_material & Sig_material = [Pe]^T * Sig_global\n
        Eps_global = [Pe] * Eps_material & Eps_material = [Ps]^T * Eps_global \n

    P matrices are orhogonal such that: inv([P]) = [P]^T\n

    Here we use "Chevalier 1988 : Comportements élastique et viscoélastique des composites"
    """

    axis_1 = np.asarray(axis_1)
    axis_2 = np.asarray(axis_2)

    dim = axis_1.shape[-1]
    assert dim in [2, 3], "Must be a 2d or 3d vector"
    shape1 = axis_1.shape
    shape2 = axis_2.shape
    assert len(shape1) <= 3, "Must be a numpy array of shape (i), (e,i) or (e,p,i)"
    assert len(shape2) <= 3, "Must be a numpy array of shape (i), (e,i) or (e,p,i)"
    assert shape1 == shape2, "axis_1 and axis_2 must be the same size"

    # get the indices and transpose
    if len(shape1) == 1:
        id = ""
        transposeP = [0, 1]  # (dim*2,dim*2) -> (dim*2,dim*2)
    elif len(shape1) == 2:
        id = "e"
        axis_1 = axis_1.transpose((1, 0))  # (e,dim) -> (dim,e)
        axis_2 = axis_2.transpose((1, 0))
        transposeP = [2, 0, 1]  # (dim,dim,e) -> (e,dim,dim)
    elif len(shape1) == 3:
        id = "ep"
        axis_1 = axis_1.transpose((2, 0, 1))  # (e,p,dim) -> (dim,e,p)
        axis_2 = axis_2.transpose((2, 0, 1))
        transposeP = [2, 3, 0, 1]  # (dim,dim,e,p) -> (e,p,dim,dim)
    else:
        raise TypeError("shape error")

    # normalize thoses vectors
    axis_1 = np.einsum(
        f"i{id},{id}->i{id}", axis_1, np.linalg.norm(axis_1, axis=0), optimize="optimal"
    )
    axis_2 = np.einsum(
        f"i{id},{id}->i{id}", axis_2, np.linalg.norm(axis_2, axis=0), optimize="optimal"
    )

    # Checks whether the two vectors are perpendicular
    dotProd = np.einsum(f"i{id},i{id}->{id}", axis_1, axis_2, optimize="optimal")  # type: ignore
    assert np.linalg.norm(dotProd) <= 1e-12, "Must give perpendicular axes"

    if dim == 2:
        p11, p12 = axis_1
        p21, p22 = axis_2
    elif dim == 3:
        # constructs z-axis
        axis_3 = np.cross(axis_1, axis_2, axis=0)
        p11, p12, p13 = axis_1
        p21, p22, p23 = axis_2
        p31, p32, p33 = axis_3
    else:
        raise TypeError("dim error")

    if len(shape1) == 1:
        p = np.zeros((dim, dim))
    elif len(shape1) == 2:
        p = np.zeros((dim, dim, shape1[0]))
    elif len(shape1) == 3:
        p = np.zeros((dim, dim, shape1[0], shape1[1]))
    else:
        raise TypeError("shape error")

    # apply vectors
    p[:, 0] = axis_1
    p[:, 1] = axis_2
    if dim == 3:
        p[:, 2] = axis_3  # type: ignore

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
    """Performs a basis transformation from the material's coordinate system to the (x,y,z) coordinate system to orient the material in space.\n
    Caution: P must be in Kelvin mandel notation

    Parameters
    ----------
    P : _types.FloatArray
        P in mandel notation obtained with Get_Pmat
    M : _types.FloatArray
        3x3 or 6x6 matrix
    toGlobal : bool, optional
        sets wheter you want to get matrix in global or material coordinates, by default True\n
        if toGlobal:\n
            Matrix_global = P * C_material * P'\n
        else:\n
            Matrix_material = P' * Matrix_global * P

    Returns
    -------
    _types.FloatArray
        new matrix
    """
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
