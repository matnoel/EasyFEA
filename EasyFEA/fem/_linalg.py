# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear algebra functions."""

import numpy as np
from typing import Union, Optional, Iterable
from ..Utilities import _types


class FeArray(_types.AnyArray):
    """Finite Element array.\n

    FeArray is a Python class designed to optimize finite element simulations by leveraging NumPy arrays with a shape of `(Ne, nPg, ...)`. This structure enables vectorized operations, eliminating the need for slow loops over elements and integration points. By using np.einsum, it efficiently handles tensor computations, significantly improving performance and code clarity for finite element analyses.
    """

    FeArrayALike = Union["FeArray", _types.AnyArray]

    def __new__(cls, input_array, broadcastFeArrays=False):
        obj = np.asarray(input_array).view(cls)
        if broadcastFeArrays:
            obj = obj[np.newaxis, np.newaxis]
        if obj.ndim < 2:
            raise ValueError("The input array must have at least 2 dimensions.")
        return obj

    def __array_finalize__(self, obj: Optional[_types.AnyArray]):
        # This method is automatically called when new instances are created.
        # It can be used to initialize additional attributes if necessary.
        if obj is None:
            return

    @property
    def _shape(self) -> tuple:
        """finite element shape"""
        return self.shape[2:]

    @property
    def _ndim(self) -> int:
        """finite element ndim"""
        return self.ndim - 2

    @property
    def _idx(self) -> str:
        """einsum indicator (e.g "", "i", "ij") used in `np.einsum()` function.\n
        see https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        if self._ndim == 0:
            return ""
        elif self._ndim == 1:
            return "i"
        elif self._ndim == 2:
            return "ij"
        elif self._ndim == 4:
            return "ijkl"
        else:
            raise ValueError("wrong dimension")

    @property
    def _type(self) -> str:
        if self._ndim == 0:
            return "scalar"
        elif self._ndim == 1:
            return "vector"
        elif self._ndim == 2:
            return "matrix"
        elif self._ndim == 4:
            return "tensor"
        else:
            raise ValueError("wrong dimension")

    def __get_array1_array2(self, other) -> tuple[_types.AnyArray, _types.AnyArray]:
        array1 = np.asarray(self)
        ndim1 = self._ndim
        shape1 = self._shape

        array2 = np.asarray(other)
        if isinstance(other, FeArray):
            ndim2 = other._ndim
            shape2 = other._shape
        elif isinstance(other, (np.ndarray, float, int)):
            ndim2 = array2.ndim
            shape2 = array2.shape
        elif type(other).__name__ == "Field":
            other: FeArray = other()  # type: ignore [no-redef]
            array2 = np.asarray(other)
            ndim2 = other._ndim
            shape2 = other._shape
        else:
            raise TypeError("other must be a FeArray, ndarray, float, int or a Field.")

        if ndim1 == 0:
            # array1(Ne, nPg)  array2(...) => (Ne, nPg, ...)
            # or
            # array1(Ne, nPg)  array2(Ne, nPg, ...) => (Ne, nPg, ...)
            for _ in range(ndim2):
                array1 = array1[..., np.newaxis]
        elif ndim2 == 0:
            if array2.size == 1:
                # array1(Ne, nPg, ...)  array2() => (Ne, nPg, ...)
                pass
            else:
                # array1(Ne, nPg, ...)  array2(Ne, nPg) => (Ne, nPg, ...)
                for _ in range(ndim1):
                    array2 = array2[..., np.newaxis]
        elif shape1 == shape2:
            pass
        else:
            type_str = "FeArray" if isinstance(other, FeArray) else "np.array"
            raise ValueError(
                f"The {type_str} `other` with shape {shape2} must be either a {self.shape} np.array or a {shape1} FeArray."
            )

        return array1, array2

    def __add__(self, other) -> FeArrayALike:
        # Overload the + operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 + array2
        return FeArray.asfearray(result)

    def __sub__(self, other) -> FeArrayALike:  # type: ignore [override]
        # Overload the - operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 - array2
        return FeArray.asfearray(result)

    def __mul__(self, other) -> FeArrayALike:
        # Overload the * operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 * array2
        return FeArray.asfearray(result)

    def __truediv__(self, other) -> FeArrayALike:  # type: ignore [override]
        # Overload the / operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 / array2
        return FeArray.asfearray(result)

    @property
    def T(self) -> FeArrayALike:  # type: ignore [override]
        if self._ndim >= 2:
            idx = self._idx
            subscripts = f"...{idx}->...{idx[::-1]}"
            result = np.einsum(subscripts, np.asarray(self), optimize="optimal")
            return FeArray.asfearray(result)
        else:
            return self.copy()

    def __matmul__(self, other) -> FeArrayALike:
        ndim1 = self._ndim

        if isinstance(other, FeArray):
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            ndim2 = other.ndim
        elif type(other).__name__ == "Field":
            other: FeArray = other()  # type: ignore [no-redef]
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")

        if ndim1 == ndim2 == 1:
            result = self.dot(other)
        elif ndim1 == ndim2 == 2:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                result = super().__matmul__(other)
        elif ndim1 == 1 and ndim2 == 2:
            result = (self[:, :, np.newaxis, :] @ other)[:, :, 0, :]
        elif ndim1 == 2 and ndim2 == 1:
            result = (self @ other[:, :, :, np.newaxis])[:, :, :, 0]
        else:
            result = self.dot(other)

        return FeArray.asfearray(result)

    def dot(self, other) -> FeArrayALike:  # type: ignore [override]
        ndim1 = self._ndim
        if ndim1 == 0:
            raise ValueError("Must be at least a finite element vector (Ne, nPg, i).")

        idx1 = self._idx

        if isinstance(other, FeArray):
            idx2 = other._idx
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            idx2 = "".join([chr(ord(idx1[0]) + i) for i in range(other.ndim)])
            ndim2 = other.ndim
        elif type(other).__name__ == "Field":
            other: FeArray = other()  # type: ignore [no-redef]
            idx2 = other._idx
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")
        idx2 = "".join([chr(ord(val) + ndim1 - 1) for val in idx2])

        if ndim2 == 0:
            raise ValueError(
                "`other` must be at least a finite element vector (Ne, nPg, i)."
            )

        end = str(idx1 + idx2).replace(idx1[-1], "")
        subscripts = f"...{idx1},...{idx2}->...{end}"

        result = np.einsum(subscripts, self, other, optimize="optimal")

        return FeArray.asfearray(result)

    def ddot(self, other) -> FeArrayALike:
        ndim1 = self._ndim
        if ndim1 < 2:
            raise ValueError(
                "Must be at least a finite element matrix (Ne, nPg, i, j)."
            )

        idx1 = self._idx

        if isinstance(other, FeArray):
            idx2 = other._idx
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            idx2 = "".join([chr(ord(idx1[0]) + i) for i in range(other.ndim)])
            ndim2 = other.ndim
        elif type(other).__name__ == "Field":
            other: FeArray = other()  # type: ignore [no-redef]
            idx2 = other._idx
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")

        if ndim2 < 2:
            raise ValueError(
                "`other` must be at least a finite element matrix (Ne, nPg, i, j)."
            )
        idx2 = "".join([chr(ord(val) + ndim1 - 2) for val in idx2])

        end = str(idx1 + idx2).replace(idx1[-1], "")
        end = end.replace(idx1[-2], "")
        subscripts = f"...{idx1},...{idx2}->...{end}"

        result = np.einsum(subscripts, self, other, optimize="optimal")

        return FeArray.asfearray(result)

    def __set_array(self, new: "FeArray"):
        """Returns the new array in `FeArray` format if the new array has the same `(Ne, nPg)` shape."""
        if self.shape[:2] == new.shape[:2]:
            return FeArray.asfearray(new)
        else:
            return np.asarray(new)

    def sum(self, *args, **kwargs) -> FeArrayALike:  # type: ignore [override]
        """`np.sum()` wrapper."""
        return self.__set_array(super().sum(*args, **kwargs))

    def max(self, *args, **kwargs) -> FeArrayALike:  # type: ignore [override]
        """`np.max()` wrapper."""
        return self.__set_array(super().max(*args, **kwargs))

    def min(self, *args, **kwargs) -> FeArrayALike:  # type: ignore [override]
        """`np.min()` wrapper."""
        return self.__set_array(super().min(*args, **kwargs))

    def _get_idx(self, *arrays) -> list[_types.AnyArray]:
        ndim = len(arrays) + 2

        Ne, nPg = self.shape[:2]

        def get_shape(i: int, array: _types.AnyArray):
            shape = np.ones(ndim, dtype=int)
            shape[i] = array.size
            return np.reshape(array, shape)

        idx = [
            get_shape(i, val)
            for i, val in enumerate([np.arange(Ne), np.arange(nPg), *arrays])
        ]

        return idx

    def _assemble(self, *arrays, value: FeArrayALike):
        idx = self._get_idx(*arrays)

        self[tuple(idx)] = value

    @staticmethod
    def asfearray(array, broadcastFeArrays=False) -> FeArrayALike:
        array = np.asarray(array)
        if broadcastFeArrays:
            return FeArray(array, broadcastFeArrays=broadcastFeArrays)
        elif array.ndim >= 2:
            return FeArray(array)
        else:
            return array

    def _asfearrays(
        *arrays: Iterable[FeArrayALike], broadcastFeArrays=False
    ) -> list[FeArrayALike]:
        return [
            FeArray.asfearray(array, broadcastFeArrays=broadcastFeArrays)
            for array in arrays
        ]

    @staticmethod
    def zeros(*shape, dtype=None) -> FeArrayALike:
        return FeArray.asfearray(np.zeros(shape=shape, dtype=dtype))

    @staticmethod
    def ones(*shape, dtype=None) -> FeArrayALike:
        return FeArray.asfearray(np.ones(shape=shape, dtype=dtype))


def __CheckMat(mat: FeArray.FeArrayALike) -> None:
    assert (
        isinstance(mat, np.ndarray) and mat.ndim >= 2 and mat.shape[-2] == mat.shape[-1]
    ), "must be a (..., dim, dim) array"
    dim = mat.shape[-1]
    assert dim > 0


def Transpose(mat: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Computes transpose(mat)"""
    assert isinstance(mat, np.ndarray) and mat.ndim >= 2
    res: FeArray.FeArrayALike = np.einsum("...ij->...ji", mat, optimize="optimal")

    if isinstance(mat, FeArray):
        res = FeArray.asfearray(res)

    return res


def Trace(mat: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Computes trace(mat)"""
    __CheckMat(mat)
    # same as np.trace(A, axis1=-2, axis2=-1)
    res: FeArray.FeArrayALike = np.einsum("...ii->...", mat, optimize="optimal")

    if isinstance(mat, FeArray):
        res = FeArray.asfearray(res)

    return res


def Det(mat: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Computes det(mat)"""
    __CheckMat(mat)

    dim = mat.shape[-1]

    if dim == 1:
        det = mat[..., 0, 0]

    elif dim == 2:
        a = mat[..., 0, 0]
        b = mat[..., 0, 1]
        c = mat[..., 1, 0]
        d = mat[..., 1, 1]

        det = (a * d) - (c * b)

    elif dim == 3:
        a11 = mat[..., 0, 0]
        a12 = mat[..., 0, 1]
        a13 = mat[..., 0, 2]
        a21 = mat[..., 1, 0]
        a22 = mat[..., 1, 1]
        a23 = mat[..., 1, 2]
        a31 = mat[..., 2, 0]
        a32 = mat[..., 2, 1]
        a33 = mat[..., 2, 2]

        det = (
            a11 * ((a22 * a33) - (a32 * a23))
            - a12 * ((a21 * a33) - (a31 * a23))
            + a13 * ((a21 * a32) - (a31 * a22))
        )

    else:
        det = np.linalg.det(mat)

    if isinstance(mat, FeArray):
        det = FeArray.asfearray(det)

    return det


def Inv(mat: FeArray.FeArrayALike):
    """Computes inv(mat)"""
    __CheckMat(mat)

    dim = mat.shape[-1]

    if dim == 1:
        inv = 1 / mat

    elif dim == 2:
        # mat = [alpha, beta          inv(mat) = 1/det * [b, -beta
        #        a    , b   ]                            -a,  alpha]

        inv = np.zeros_like(mat, dtype=float)

        det = Det(mat)

        alpha = mat[..., 0, 0]
        beta = mat[..., 0, 1]
        a = mat[..., 1, 0]
        b = mat[..., 1, 1]

        adj = np.zeros_like(mat)

        adj[..., 0, 0] = b
        adj[..., 0, 1] = -beta
        adj[..., 1, 0] = -a
        adj[..., 1, 1] = alpha

        inv = np.einsum("...,...ij->...ij", 1 / det, adj, optimize="optimal")

    elif dim == 3:
        # optimized such that invmat = 1/det * Adj(mat)
        # https://fr.wikihow.com/calculer-l'inverse-d'une-matrice-3x3

        det = Det(mat)

        matT = Transpose(mat)

        a00 = matT[..., 0, 0]
        a01 = matT[..., 0, 1]
        a02 = matT[..., 0, 2]
        a10 = matT[..., 1, 0]
        a11 = matT[..., 1, 1]
        a12 = matT[..., 1, 2]
        a20 = matT[..., 2, 0]
        a21 = matT[..., 2, 1]
        a22 = matT[..., 2, 2]

        det00 = (a11 * a22) - (a21 * a12)
        det01 = (a10 * a22) - (a20 * a12)
        det02 = (a10 * a21) - (a20 * a11)
        det10 = (a01 * a22) - (a21 * a02)
        det11 = (a00 * a22) - (a20 * a02)
        det12 = (a00 * a21) - (a20 * a01)
        det20 = (a01 * a12) - (a11 * a02)
        det21 = (a00 * a12) - (a10 * a02)
        det22 = (a00 * a11) - (a10 * a01)

        adj = np.zeros_like(mat)

        # Don't forget the - or + !!!
        adj[..., 0, 0] = det00
        adj[..., 0, 1] = -det01
        adj[..., 0, 2] = det02
        adj[..., 1, 0] = -det10
        adj[..., 1, 1] = det11
        adj[..., 1, 2] = -det12
        adj[..., 2, 0] = det20
        adj[..., 2, 1] = -det21
        adj[..., 2, 2] = det22

        inv = np.einsum("...,...ij->...ij", 1 / det, adj, optimize="optimal")

    else:
        inv = np.linalg.inv(mat)

    if isinstance(mat, FeArray):
        inv = FeArray.asfearray(inv)

    return inv


def TensorProd(
    A: FeArray.FeArrayALike,
    B: FeArray.FeArrayALike,
    symmetric=False,
    ndim: Optional[int] = None,
) -> FeArray.FeArrayALike:
    """Computes tensor product.

    Parameters
    ----------
    A : FeArray.FeArrayALike
        array A
    B : FeArray.FeArrayALike
        array B
    symmetric : bool, optional
        do symmetric product, by default False
    ndim : int, optional
        ndim=1 -> vect or ndim=2 -> matrix, by default None

    Returns
    -------
    FeArray.FeArrayALike:
        the calculated tensor product
    """

    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)

    useFeArray = isinstance(A, FeArray) or isinstance(B, FeArray)

    if ndim is None:
        ndim = A._ndim if useFeArray else A.ndim

    assert ndim in [1, 2], "A and B must be vectors (i) or matrices (ij)"

    error = "A and B must have the same dimensions"
    if useFeArray:
        ndim1 = A._ndim if useFeArray else A.ndim
        ndim2 = B._ndim if useFeArray else B.ndim
        assert ndim1 == ndim2, error
    else:
        assert A.size == B.size, error

    if ndim == 1:
        # vectors
        # Ai Bj
        res = np.einsum("...i,...j->...ij", A, B, optimize="optimal")

    elif ndim == 2:
        # matrices
        if symmetric:
            # 1/2 * (Aik Bjl + Ail Bjk) = 1/2 (p1 + p2)
            p1 = np.einsum("...ik,...jl->...ijkl", A, B, optimize="optimal")
            p2 = np.einsum("...il,...jk->...ijkl", A, B, optimize="optimal")
            res = 1 / 2 * (p1 + p2)
        else:
            # Aij Bkl
            res = np.einsum("...ij,...kl->...ijkl", A, B, optimize="optimal")

    else:
        raise Exception("Not implemented")

    if useFeArray:
        res = FeArray.asfearray(res)

    return res


def Norm(array: FeArray.FeArrayALike, **kwargs) -> FeArray.FeArrayALike:
    """`np.linalg.norm()` wrapper.\n
    see https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html"""

    res: FeArray.FeArrayALike = np.linalg.norm(array, **kwargs)

    if isinstance(array, FeArray):
        res = FeArray.asfearray(res)

    return res
