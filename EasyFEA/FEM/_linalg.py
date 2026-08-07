# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear algebra functions."""

import numpy as np
from functools import lru_cache
from typing import Union, Optional, Iterable
from ..Utilities import _types


def _Evaluate(operand):
    """A Field evaluates to its finite element array; anything else is passed through."""
    return operand() if getattr(operand, "_isFeField", False) else operand


def _Base(operand):
    """Drops the FeArray view, so numpy's own machinery handles the operand."""
    if isinstance(operand, FeArray):
        return operand.view(np.ndarray)
    elif isinstance(operand, (list, tuple)):
        return type(operand)(_Base(item) for item in operand)
    else:
        return operand


def _KeepsFeAxes(axis, ndim: int) -> bool:
    """True when a reduction consumes tensor axes only, so the (Ne, nPg) axes survive."""
    if axis is None:
        return False
    axes = axis if isinstance(axis, tuple) else (axis,)
    return all(a >= 2 if a >= 0 else a >= 2 - ndim for a in axes)


_REDUCERS = frozenset(
    {
        np.sum,
        np.prod,
        np.mean,
        np.std,
        np.var,
        np.median,
        np.average,
        np.max,
        np.min,
        np.amax,
        np.amin,
        np.all,
        np.any,
        np.argmax,
        np.argmin,
    }
)


def _FeShape(operands) -> tuple:
    """The (Ne, nPg) an operation runs at: the broadcast of its FeArray operands'."""
    shapes = set()
    stack = list(operands)
    while stack:
        operand = stack.pop()
        if isinstance(operand, FeArray):
            shapes.add(operand.shape[:2])
        elif isinstance(operand, (list, tuple)):
            stack.extend(operand)
    if len(shapes) == 1:
        return shapes.pop()
    return np.broadcast_shapes(*shapes) if shapes else ()


class FeArray(np.ndarray):
    """Finite Element array.\n

    FeArray is a Python class designed to optimize finite element simulations by leveraging NumPy arrays with a shape of `(Ne, nPg, ...)`. This structure enables vectorized operations, eliminating the need for slow loops over elements and integration points. By using np.einsum, it efficiently handles tensor computations, significantly improving performance and code clarity for finite element analyses.

    Two rules govern how it mixes with other arrays:

    - **Rank.** A FeArray's tensor rank is ``ndim - 2`` and a plain array's is its own ``ndim``
      -- always, with no exception and nothing inferred from a shape coincidence. So a
      ``(Ne, nPg)`` FeArray is a scalar field even where ``Ne`` and ``nPg`` match a tensor's
      dimensions, and a plain array is a constant tensor held at every Gauss point. Fields are
      padded to the widest rank, then broadcast once. A plain array that is really a field must
      say so with :meth:`asfearray`, or it multiplies out as a constant tensor.
    - **Type.** An operation stays a FeArray exactly when the ``(Ne, nPg)`` axes come out
      unchanged. ``np.einsum``, ``np.where`` and ``np.linalg.solve`` keep them; ``reshape``,
      ``ravel`` and a sum over elements do not.
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

    def __check_fe_dims(self) -> None:
        """A FeArray must keep its leading (Ne, nPg) axes.

        ``__new__`` enforces that, but indexing and reshaping do not go through it: ``fe[0]``
        returns a FeArray of one dimension, whose finite element rank would be -1. Nothing
        downstream expects that, so it is caught here rather than surfacing later as a
        confusing shape error.
        """
        if self.ndim < 2:
            raise ValueError(
                f"this FeArray has shape {self.shape}, which has lost the leading "
                "(Ne, nPg) axes -- indexing or reshaping dropped them. Use np.asarray(...) "
                "if plain array semantics are what is wanted here."
            )

    @property
    def _shape(self) -> tuple:
        """finite element shape"""
        self.__check_fe_dims()
        return self.shape[2:]

    @property
    def _ndim(self) -> int:
        """finite element ndim"""
        self.__check_fe_dims()
        return self.ndim - 2

    @staticmethod
    def _align(operands: tuple) -> tuple:
        """Pads each field's tensor rank up to the widest, so one numpy broadcast is correct.

        The finite element axes then line up on the left and the tensor axes on the right,
        which is numpy's own rule.
        """
        # by far the commonest case: all fields of the same shape, nothing to line up
        shape = operands[0].shape if isinstance(operands[0], FeArray) else None
        for op in operands:
            if not isinstance(op, FeArray) or op.shape != shape:
                break
        else:
            return operands

        operands = tuple(_Evaluate(op) for op in operands)
        ranks = [
            op.ndim - 2 if isinstance(op, FeArray) else np.ndim(op) for op in operands
        ]
        nt = max(ranks)
        return tuple(
            (
                op[(slice(None), slice(None)) + (None,) * (nt - rank)]
                if isinstance(op, FeArray) and rank < nt
                else op
            )
            for op, rank in zip(operands, ranks)
        )

    @staticmethod
    def __wrap(res, feShape: tuple):
        """A result is a field exactly when it came out on the operation's (Ne, nPg) axes."""
        if not isinstance(res, np.ndarray):
            return res
        elif res.ndim >= 2 and res.shape[:2] == feShape:
            return res.view(FeArray)
        else:
            return np.asarray(res)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # numpy asks that a subclass put all its override logic here rather than also defining
        # __add__ and friends, so that the type hierarchy is decided in one place
        elementwise = method == "__call__" and ufunc.signature is None

        # two fields of the same shape need no alignment and no rewrapping decision; this is
        # the overwhelming majority of calls, and it is what keeps small arrays cheap
        if elementwise and not kwargs and len(inputs) == 2:
            left, right = inputs
            if (
                type(left) is FeArray
                and type(right) is FeArray
                and left.shape == right.shape
            ):
                return ufunc(left.view(np.ndarray), right.view(np.ndarray)).view(
                    FeArray
                )

        if elementwise:
            inputs = FeArray._align(inputs)

        # ndarray refuses to run a ufunc on a subclass that overrides __array_ufunc__, so hand
        # it plain views -- of the `out` and `where` operands too, or the call comes straight
        # back here -- and put the type back afterwards
        out = kwargs.pop("out", None) if kwargs else None
        if kwargs:
            kwargs = {key: _Base(value) for key, value in kwargs.items()}
        if out is not None:
            kwargs["out"] = _Base(out)

        args = [_Base(array) for array in inputs]
        res = (
            ufunc(*args, **kwargs)
            if elementwise
            else getattr(ufunc, method)(*args, **kwargs)
        )

        if out is not None:
            return out[0] if len(out) == 1 else out
        elif elementwise and type(res) is np.ndarray:
            # broadcasting against a FeArray always keeps the (Ne, nPg) axes
            return res.view(FeArray)
        feShape = _FeShape(inputs)
        if isinstance(res, tuple):
            return tuple(FeArray.__wrap(array, feShape) for array in res)
        return FeArray.__wrap(res, feShape)

    def __array_function__(self, func, types, args, kwargs):
        # numpy's own implementations broadcast the plain way and must keep doing so: einsum
        # with optimize= reaches for np.multiply internally, which would otherwise come back
        # through __array_ufunc__ and be aligned a second time
        feShape = _FeShape(args) or _FeShape(kwargs.values())
        # numpy calls a dispatched reduction on the stripped array, so the method wrapper never
        # sees it and the axis has to be read here instead
        if func in _REDUCERS:
            axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
            if not _KeepsFeAxes(axis, np.ndim(args[0])):
                feShape = ()
        args = tuple(_Base(arg) for arg in args)
        kwargs = {key: _Base(value) for key, value in kwargs.items()}
        res = super().__array_function__(func, types, args, kwargs)
        return FeArray.__wrap(res, feShape)

    @property
    def T(self) -> FeArrayALike:  # type: ignore [override]
        if self._ndim == 2:
            # swapaxes returns a view — no allocation
            return FeArray.asfearray(np.swapaxes(np.asarray(self), -1, -2))
        elif self._ndim > 2:
            # np.transpose returns a view — no allocation
            n = self.ndim
            axes = tuple(range(2)) + tuple(range(n - 1, 1, -1))
            return FeArray.asfearray(np.asarray(self).transpose(axes))
        else:
            return FeArray.asfearray(self)

    def __matmul__(self, other) -> FeArrayALike:
        ndim1 = self._ndim

        if isinstance(other, FeArray):
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            ndim2 = other.ndim
        elif getattr(other, "_isFeField", False):
            other: FeArray = other()  # type: ignore [no-redef]
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")

        if ndim1 == ndim2 == 1:
            return self.dot(other)
        elif ndim1 == ndim2 == 2:
            return super().__matmul__(other)
        elif ndim1 == 1 and ndim2 == 2:
            return FeArray.asfearray(np.einsum("...i,...ij->...j", self, other))
        elif ndim1 == 2 and ndim2 == 1:
            return FeArray.asfearray(np.einsum("...ij,...j->...i", self, other))
        else:
            return self.dot(other)

    @staticmethod
    @lru_cache(maxsize=16)
    def _dot_subscript(ndim1: int, ndim2: int) -> str:
        """Build and cache the einsum subscript for dot(ndim1, ndim2)."""
        _idx = {0: "", 1: "i", 2: "ij", 4: "ijkl"}
        idx1 = _idx[ndim1]
        idx2 = "".join(chr(ord(v) + ndim1 - 1) for v in _idx[ndim2])
        end = (idx1 + idx2).replace(idx1[-1], "")
        return f"...{idx1},...{idx2}->...{end}"

    @staticmethod
    @lru_cache(maxsize=16)
    def _ddot_subscript(ndim1: int, ndim2: int) -> str:
        """Build and cache the einsum subscript for ddot(ndim1, ndim2)."""
        _idx = {0: "", 1: "i", 2: "ij", 4: "ijkl"}
        idx1 = _idx[ndim1]
        idx2 = "".join(chr(ord(v) + ndim1 - 2) for v in _idx[ndim2])
        end = (idx1 + idx2).replace(idx1[-1], "").replace(idx1[-2], "")
        return f"...{idx1},...{idx2}->...{end}"

    def dot(self, other) -> FeArrayALike:  # type: ignore [override]
        ndim1 = self._ndim
        if ndim1 == 0:
            raise ValueError("Must be at least a finite element vector (Ne, nPg, i).")

        if isinstance(other, FeArray):
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            ndim2 = other.ndim
        elif getattr(other, "_isFeField", False):
            other: FeArray = other()  # type: ignore [no-redef]
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")

        if ndim2 == 0:
            raise ValueError(
                "`other` must be at least a finite element vector (Ne, nPg, i)."
            )

        result = np.einsum(self._dot_subscript(ndim1, ndim2), self, other)

        return FeArray.asfearray(result)

    def ddot(self, other) -> FeArrayALike:
        ndim1 = self._ndim
        if ndim1 < 2:
            raise ValueError(
                "Must be at least a finite element matrix (Ne, nPg, i, j)."
            )

        if isinstance(other, FeArray):
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            ndim2 = other.ndim
        elif getattr(other, "_isFeField", False):
            other: FeArray = other()  # type: ignore [no-redef]
            ndim2 = other._ndim
        else:
            raise TypeError("`other` must be either a FeArray, NDArray or a Field.")

        if ndim2 < 2:
            raise ValueError(
                "`other` must be at least a finite element matrix (Ne, nPg, i, j)."
            )

        result = np.einsum(self._ddot_subscript(ndim1, ndim2), self, other)

        return result.view(FeArray)

    # A reduction over a tensor axis is still a field; one over elements or Gauss points is
    # not. Which axes were consumed is read from `axis`, never guessed from the result shape.
    def _make_reducer(_name: str):
        _parent = getattr(np.ndarray, _name)

        def _reducer(self, *args, **kwargs):
            res = _parent(self, *args, **kwargs)
            axis = kwargs.get("axis", args[0] if args else None)
            if _KeepsFeAxes(axis, self.ndim) and getattr(res, "ndim", 0) >= 2:
                return res.view(FeArray)
            return np.asarray(res)

        _reducer.__name__ = _name
        _reducer.__qualname__ = f"FeArray.{_name}"
        _reducer.__doc__ = (
            f"``np.{_name}()`` wrapper — ``ndarray`` unless (Ne, nPg) survives."
        )
        return _reducer

    for _name in (
        "sum",
        "prod",
        "mean",
        "std",
        "var",
        "max",
        "min",
        "argmax",
        "argmin",
        "all",
        "any",
        "ravel",
    ):
        locals()[_name] = _make_reducer(_name)
    del _name, _make_reducer

    def reshape(self, *args, **kwargs):
        new = super().reshape(*args, **kwargs)
        if self.ndim >= 2 and new.shape[:2] == self.shape[:2]:
            return new
        return np.asarray(new)

    def integrate(self) -> np.ndarray:
        """Integrate over the Gauss-point axis (axis 1). Returns ``(Ne, ...)`` ndarray."""
        return np.asarray(super().sum(axis=1))

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
    def asfearray(array, broadcastFeArrays=False) -> "FeArray":
        """Views ``array`` as a FeArray. Refuses anything without the (Ne, nPg) axes."""
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if broadcastFeArrays:
            return FeArray(array, broadcastFeArrays=broadcastFeArrays)
        elif array.ndim < 2:
            raise ValueError(
                f"cannot view a {array.shape} array as a FeArray: it has no (Ne, nPg) axes. "
                "Pass broadcastFeArrays=True to hold it at every Gauss point, or keep it a "
                "plain array."
            )
        return array.view(FeArray)

    @staticmethod
    def broadcast(
        value, Ne: int, nPg: int, tensor_ndim: int = 0
    ) -> "FeArray.FeArrayALike":
        """Broadcast a scalar or array coefficient to a shape compatible with multiplication against an ``(Ne, nPg, ...)`` FeArray.

        Returns a stride-tricked read-only view for non-scalar inputs (no data duplication). Callers must not mutate the result in place;
        the expected use is consumption inside expressions such as ``coef * wJ_e_pg * dN_e_pg.T @ dN_e_pg``, which create new arrays.

        ``tensor_ndim`` declares how many trailing axes of ``value`` are tensor dims (e.g. ``2`` for a Hooke tensor ``(..., nstrain, nstrain)``).
        With it set, the leading axes are checked against ``()`` / ``(Ne,)`` / ``(Ne, nPg)`` strictly — this disambiguates shapes like
        ``(Ne, n, n)`` from ``(Ne, nPg, n)`` when ``nPg == n`` (e.g. TRI6 in 2D, where ``nPg == nstrain == 3``).

        Accepted shapes (with default ``tensor_ndim=0``)
        ------------------------------------------------
        - scalar (int / float / numpy scalar) → returned as ``float``.
        - ``(Ne, nPg, ...)`` ndarray / FeArray → wrapped as FeArray.
        - 1-D ``(Ne,)`` or ``(nPg,)`` → tiled to ``(Ne, nPg)``.
        - Any other shape broadcastable to ``(Ne, nPg, ...)`` → tiled with leading ``(Ne, nPg)`` dims.
        """
        if isinstance(value, (int, float, np.floating, np.integer)):
            return float(value)
        arr = np.asarray(value)

        if tensor_ndim > 0:
            tail = arr.shape[-tensor_ndim:] if tensor_ndim else ()
            lead = arr.shape[:-tensor_ndim] if tensor_ndim else arr.shape
            if lead == (Ne, nPg):
                return FeArray.asfearray(arr)
            if lead == (Ne,):
                return FeArray.asfearray(
                    np.broadcast_to(arr[:, None], (Ne, nPg) + tail)
                )
            if lead == ():
                return FeArray.asfearray(
                    np.broadcast_to(arr[None, None], (Ne, nPg) + tail)
                )
            raise ValueError(
                f"With tensor_ndim={tensor_ndim}, leading axes must be (), (Ne,), or (Ne, nPg); got {lead}."
            )

        if arr.shape[:2] == (Ne, nPg):
            return FeArray.asfearray(arr)
        if arr.ndim == 1:
            if arr.shape[0] == Ne:
                return FeArray.asfearray(np.broadcast_to(arr[:, None], (Ne, nPg)))
            if arr.shape[0] == nPg:
                return FeArray.asfearray(np.broadcast_to(arr[None, :], (Ne, nPg)))
        return FeArray.asfearray(
            np.broadcast_to(arr[None, None], (Ne, nPg) + arr.shape)
        )

    def _asfearrays(
        *arrays: Iterable[FeArrayALike], broadcastFeArrays=False
    ) -> list[FeArrayALike]:
        return [
            FeArray.asfearray(array, broadcastFeArrays=broadcastFeArrays)
            for array in arrays
        ]

    @staticmethod
    def __shape(shape: tuple) -> tuple:
        """Accepts both ``zeros(Ne, nPg, 6)`` and ``zeros((Ne, nPg, 6))``."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(shape[0])
        return shape

    @staticmethod
    def zeros(*shape, dtype=None) -> FeArrayALike:
        return FeArray.asfearray(np.zeros(FeArray.__shape(shape), dtype=dtype))

    @staticmethod
    def ones(*shape, dtype=None) -> FeArrayALike:
        return FeArray.asfearray(np.ones(FeArray.__shape(shape), dtype=dtype))


def __CheckMat(mat: FeArray.FeArrayALike) -> None:
    assert (
        isinstance(mat, np.ndarray) and mat.ndim >= 2 and mat.shape[-2] == mat.shape[-1]
    ), "must be a (..., dim, dim) array"
    dim = mat.shape[-1]
    assert dim > 0


def Transpose(mat: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Computes transpose(mat)"""
    assert isinstance(mat, np.ndarray) and mat.ndim >= 2
    res: FeArray.FeArrayALike = np.swapaxes(mat, -1, -2)

    if isinstance(mat, FeArray):
        res = FeArray.asfearray(res)

    return res


def Trace(mat: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Computes trace(mat)"""
    __CheckMat(mat)
    # same as np.trace(A, axis1=-2, axis2=-1)
    res: FeArray.FeArrayALike = np.einsum("...ii->...", mat)

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

        inv = np.einsum("...,...ij->...ij", 1 / det, adj)

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

        inv = np.einsum("...,...ij->...ij", 1 / det, adj)

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
        res = np.einsum("...i,...j->...ij", A, B)

    elif ndim == 2:
        # matrices
        if symmetric:
            # 1/2 * (Aik Bjl + Ail Bjk) = 1/2 (p1 + p2)
            p1 = np.einsum("...ik,...jl->...ijkl", A, B)
            p2 = np.einsum("...il,...jk->...ijkl", A, B)
            res = 1 / 2 * (p1 + p2)
        else:
            # Aij Bkl
            res = np.einsum("...ij,...kl->...ijkl", A, B)

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


def Normalize(array: FeArray.FeArrayALike, axis: int = -1) -> FeArray.FeArrayALike:
    """Unit-normalize ``array`` along ``axis``.

    Each slice is divided by its Euclidean length. Zero-norm slices are left
    unchanged (no division by zero), so a zero vector stays zero. Returns a
    ``FeArray`` when given one.

    Parameters
    ----------
    array : FeArray.FeArrayALike
        Values to normalize, e.g. a direction field ``(Ne, nPg, 3)``.
    axis : int, optional
        Axis along which the norm is taken, by default ``-1``.
    """
    arr = np.asarray(array)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    # leave zero-norm slices untouched (avoid 0/0): divide by 1 there
    norm = np.where(norm == 0.0, 1.0, norm)
    res: FeArray.FeArrayALike = arr / norm

    if isinstance(array, FeArray):
        res = FeArray.asfearray(res)

    return res
