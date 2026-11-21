# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear system `A x = b` of one problem."""

import numpy as np
from scipy import sparse

from ..Utilities import _types
from ..Utilities._mpi import (
    MPI_SIZE,
    Exchange,
    Owned,
    Sum_owned,
    Build_exchange,
    Scatter_forward,
    Exscan_sum,
    Reduce_max,
    Reduce_sum,
)
from ._backends import _Backend, _PETScSolver
from ._constraints import Constraints, LagrangeConstraint, ResolType


class System:
    """`A x = b` over a vector's rows: numbering, dof exchange and kept PETSc solver, A passed in; `comm` is passed in, never stored."""

    def __init__(
        self,
        owners: _types.IntArray | None = None,
        rank: int = 0,
        globalIds: _types.IntArray | None = None,
    ) -> None:
        """`owners` and `globalIds` of the vector's rows, None when not partitioned."""
        self.__owners = owners
        self.__rank = rank
        self.__globalIds = globalIds
        self.__exchange: Exchange | None = None
        self.__known: _types.BoolArray | None = None
        self.rows: _types.IntArray = np.empty(0, dtype=int)
        """the owned unknowns, in PETSc order"""
        self.ids: _types.IntArray = np.empty(0, dtype=np.int64)
        """PETSc id of each dof, -1 for a known one"""
        self.rowIds: _types.IntArray = np.empty(0, dtype=np.int64)
        """`ids`, -1 at the ghost rows"""
        self._petsc = _PETScSolver()
        """PETSc's KSP and Mat, kept for the PETSc backend"""

    def __getstate__(self) -> dict:
        # PETSc objects and graph communicators do not pickle
        return {
            **self.__dict__,
            "_System__exchange": None,
            "_petsc": _PETScSolver(),
        }

    def Get_exchange(self, comm) -> Exchange | None:
        """Exchange of the vector's rows, None unless shared; built on first use. Collective on `comm`."""
        if self.__exchange is None and self.__globalIds is not None:
            assert self.__owners is not None
            self.__exchange = Build_exchange(self.__globalIds, self.__owners, comm)
        return self.__exchange

    def Scatter_forward(self, values: np.ndarray, comm) -> None:
        """Copies onto the ghost rows of a vector of the problem their owners' values. Collective on `comm`."""
        Scatter_forward(values, self.Get_exchange(comm))

    def Set_known(self, known: _types.BoolArray, comm) -> None:
        """Numbers the owned unknowns rank after rank, the multipliers last; kept while `known` holds. Collective on `comm`."""
        changed = self.__known is None or not np.array_equal(known, self.__known)
        if not Reduce_max(changed, comm):
            return
        owned = np.zeros(known.size, dtype=bool)
        size = known.size if self.__owners is None else self.__owners.size
        owned[:size][Owned(self.__owners, self.__rank)] = True
        # the multipliers on the last rank
        owned[size:] = comm is None or self.__rank == comm.Get_size() - 1
        self.__known = known
        self.rows = np.flatnonzero(owned & ~known)
        ids = np.full(known.size, -1, dtype=np.int64)
        ids[self.rows] = Exscan_sum(self.rows.size, comm) + np.arange(self.rows.size)
        self.Scatter_forward(ids, comm)
        self.ids = ids
        self.rowIds = np.where(owned, ids, -1)

    @property
    def known(self) -> _types.BoolArray:
        """The known dofs `Set_known` numbered around."""
        assert self.__known is not None
        return self.__known

    def Norm(self, values: _types.FloatArray, comm) -> float:
        """2-norm of the owned rows of a vector of the problem, the same on every rank. Collective on `comm`."""
        return float(np.sqrt(Sum_owned(values**2, self.__owners, self.__rank, comm)))

    def Free(self) -> None:
        """Destroys the kept PETSc solver. Collective."""
        self._petsc.Free()

    def Solve(
        self,
        A: sparse.csr_matrix,
        b: _types.FloatArray,
        x: _types.FloatArray,
        constraints: Constraints,
        backend: _Backend,
        comm,
    ) -> tuple[_types.FloatArray, float]:
        """dx of `A dx = b` at x, `x + dx` meeting the constraints, A and b grown by one row per Lagrange constraint; b's residual norm over the owned dof rows. Collective on `comm`."""
        list_lagrangeConstraint = constraints.list_lagrangeConstraint
        size = x.size
        assert b.size == size
        if list_lagrangeConstraint:
            shifted = [
                lagrangeConstraint._replace(
                    value=float(
                        lagrangeConstraint.value
                        - lagrangeConstraint.coefs @ x[lagrangeConstraint.dofs]
                    )
                )
                for lagrangeConstraint in list_lagrangeConstraint
            ]
            A, b = _Border(A, b, shifted)
        knownDofs = constraints.knownDofs
        known = np.zeros(b.size, dtype=bool)
        known[knownDofs] = True
        dx = np.zeros(b.size)
        dx[knownDofs] = constraints.knownValues - x[knownDofs]
        # the known values of the ghosts from their owner
        self.Scatter_forward(dx, comm)
        if constraints.resolution == ResolType.r3:
            A, b, known = _Penalty(A, b, dx, known)
        self.Set_known(known, comm)

        backend.Check_saddle_point(
            lambda: bool(list_lagrangeConstraint) or self.__Has_zero_diagonal(A, comm)
        )

        # no known dof moved: no lift
        residual = b - A @ dx if dx.any() else b.copy()
        # the multipliers past the dofs
        rhsNorm = _Residual_norm(residual[self.rows[self.rows < size]], comm)

        dx += backend.Solve(A, residual, self, comm)
        self.Scatter_forward(dx, comm)
        return dx[:size], rhsNorm

    def __Has_zero_diagonal(self, A: sparse.csr_matrix, comm) -> bool:
        """Whether a numbered row has no diagonal entry, on any rank (a saddle point); only owned rows are complete under MPI."""
        diagonal = A.diagonal()[self.rows]
        return Reduce_sum(int(np.count_nonzero(diagonal == 0)), comm) > 0


def _Border(
    A: sparse.csr_matrix,
    b: _types.FloatArray,
    list_lagrangeConstraint: list[LagrangeConstraint],
) -> tuple[sparse.csr_matrix, _types.FloatArray]:
    """A and b grown by one row per Lagrange constraint, A bordered, both scaled by `max|A|`."""
    if MPI_SIZE > 1:
        raise NotImplementedError(
            "Simulations using Lagrangian boundary conditions are not yet supported in MPI environments."
        )
    alpha = float(np.abs(A.data).max(initial=0.0))
    size = b.size
    sizeBordered = size + len(list_lagrangeConstraint)
    b = np.concatenate([b, np.zeros(len(list_lagrangeConstraint))])
    rows, cols, values = [], [], []
    for i, lagrangeConstraint in enumerate(list_lagrangeConstraint, size):
        dofs = lagrangeConstraint.dofs
        line = np.full(dofs.size, i)
        rows += [line, dofs]
        cols += [dofs, line]
        values += [alpha * lagrangeConstraint.coefs] * 2
        b[i] = alpha * lagrangeConstraint.value
    # A's entries kept, explicit zeros included: the same pattern, the same ordering
    coo = A.tocoo()
    bordered = sparse.csr_matrix(
        (
            np.concatenate([coo.data, *values]),
            (np.concatenate([coo.row, *rows]), np.concatenate([coo.col, *cols])),
        ),
        shape=(sizeBordered, sizeBordered),
    )
    return bordered, b


def _Penalty(
    A: sparse.csr_matrix,
    b: _types.FloatArray,
    x: _types.FloatArray,
    known: _types.BoolArray,
) -> tuple[sparse.csr_matrix, _types.FloatArray, _types.BoolArray]:
    """A's known rows dropped, 1.0 on their diagonal, b's set to x's: every dof unknown."""
    weights = known.astype(float)
    A = (sparse.diags(1.0 - weights) @ A + sparse.diags(weights)).tocsr()
    b = np.where(known, x, b)
    return A, b, np.zeros_like(known)


def _Residual_norm(residual: _types.FloatArray, comm) -> float:
    """2-norm of the ranks' rows of `residual`; raises on NaN/Inf. Collective on `comm`."""
    rhsNorm = float(np.sqrt(Reduce_sum(np.linalg.norm(residual) ** 2, comm)))
    if not np.isfinite(rhsNorm):
        raise FloatingPointError("The residual has NaN or Inf entries.")
    return rhsNorm
