# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear system `A x = b` of one problem."""

from typing import Callable, NamedTuple

import numpy as np
from scipy import sparse

from ..Utilities import _types
from ..Utilities._mpi import (
    Exchange,
    Owned,
    Build_exchange,
    Scatter_forward,
    Exscan_sum,
    Reduce_max,
    Reduce_sum,
)
from ._terms import Same_provenance


class _Slot(NamedTuple):
    """One assembled K, C, M or F: its csr map key, what it was folded from, the matrix."""

    key: tuple
    provenance: list
    matrix: sparse.csr_matrix


class _LinearSystem:
    """`A x = b` of one problem key, dropped with the mesh: K, C, M, F, A, PETSc numbering, dof exchange and kept solver; `comm` is passed in, never stored."""

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
        self.__slots: dict[str, _Slot] = {}
        self.__valuesDirty = True
        self.__patternDirty = True
        self.__summed: list[tuple[float, str]] = []
        """`(coef, slot)` A was summed from"""
        self.__pattern: tuple = ()
        self.__A: sparse.csr_matrix | None = None
        self.__stamp = 0
        self.__patternStamp = 0
        self.__known: _types.BoolArray | None = None
        self.rows: _types.IntArray = np.empty(0, dtype=int)
        """the owned unknowns, in PETSc order"""
        self.ids: _types.IntArray = np.empty(0, dtype=np.int64)
        """PETSc id of each dof, -1 for a known one"""
        self.rowIds: _types.IntArray = np.empty(0, dtype=np.int64)
        """`ids`, -1 at the ghost rows"""
        self.__numberingStamp = 0
        self.__solver: tuple | None = None
        """`(ksp, matrix)`"""
        self.__solverStamps: tuple[int, int, int] | None = None

    def __getstate__(self) -> dict:
        # PETSc objects and graph communicators do not pickle; a copy assembles into its own slots
        return {
            **self.__dict__,
            "_LinearSystem__slots": dict(self.__slots),
            "_LinearSystem__exchange": None,
            "_LinearSystem__solver": None,
            "_LinearSystem__solverStamps": None,
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
        """Numbers the owned unknowns rank after rank, kept until `known` changes on a rank. Collective on `comm`."""
        changed = self.__known is None or not np.array_equal(known, self.__known)
        if not Reduce_max(changed, comm):
            return
        owned = np.zeros(known.size, dtype=bool)
        owned[Owned(self.__owners, self.__rank)] = True
        self.__known = known
        self.rows = np.flatnonzero(owned & ~known)
        ids = np.full(known.size, -1, dtype=np.int64)
        ids[self.rows] = Exscan_sum(self.rows.size, comm) + np.arange(self.rows.size)
        self.Scatter_forward(ids, comm)
        self.ids = ids
        self.rowIds = np.where(owned, ids, -1)
        self.__numberingStamp += 1

    @property
    def solver(self) -> tuple | None:
        """The kept PETSc `(ksp, matrix)`."""
        return self.__solver

    def Solver_update(self, comm) -> tuple[bool, bool]:
        """Whether the kept solver must be laid out again (new pattern of A or numbering) and refilled (new A). Collective on `comm`."""
        stamps = (self.__patternStamp, self.__numberingStamp, self.__stamp)
        kept = self.__solverStamps
        relayout = self.__solver is None or kept is None or stamps[:2] != kept[:2]
        refill = relayout or kept is None or stamps[2] != kept[2]
        relayout, refill = Reduce_sum(np.array([relayout, refill], dtype=int), comm)
        return bool(relayout), bool(refill)

    def Set_solver(self, solver: tuple) -> None:
        """Keeps `(ksp, matrix)`, laid out on the current pattern and numbering, filled with the current A."""
        self.__solver = solver
        self.__solverStamps = (self.__patternStamp, self.__numberingStamp, self.__stamp)

    def Take_solver(self) -> tuple | None:
        """The kept solver, no longer kept."""
        solver = self.__solver
        self.__solver = None
        self.__solverStamps = None
        return solver

    def Assembled(
        self,
        slot: str,
        key: tuple,
        provenance: list,
        assemble: Callable[[], sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        """The slot's matrix, assembled again unless folded from the same cached arrays on the same map; a new K, C or M dirties A, its pattern too on a new map."""
        old = self.__slots.get(slot)
        if (
            old is not None
            and old.key == key
            and Same_provenance(old.provenance, provenance)
        ):
            return old.matrix
        matrix = assemble()
        if slot != "F":
            self.__valuesDirty = True
            if old is None or old.key != key:
                self.__patternDirty = True
        self.__slots[slot] = _Slot(key, provenance, matrix)
        return matrix

    def Get_K_C_M_F(
        self,
    ) -> tuple[
        sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix
    ]:
        """The last assembled K, C, M, F, read-only."""
        self.__Check_assembled()
        K, C, M, F = (self.__slots[slot].matrix for slot in "KCMF")
        return K, C, M, F

    def __Check_assembled(self) -> None:
        if "K" not in self.__slots:
            raise RuntimeError(
                "Nothing assembled for this problem: a `Get_K_C_M_F` override must assemble through `Assembly`."
            )

    def Assembled_rows(self) -> _types.BoolArray:
        """Rows K, C or M hold an entry in, ghost elements included."""
        self.__Check_assembled()
        rows = sum(np.diff(self.__slots[slot].matrix.indptr) for slot in "KCM")
        return rows > 0

    def Get_A(self, coefs: tuple[float, float, float]) -> sparse.csr_matrix:
        """`coefK·K + coefC·C + coefM·M` (K alone when the others weigh nothing); the same read-only object until a slot or a coefficient changes."""
        self.__Check_assembled()
        slots = self.__slots
        summed = [
            (coef, slot)
            for coef, slot in zip(coefs, "KCM")
            if slot == "K" or (coef != 0 and slots[slot].matrix.nnz)
        ]
        if self.__A is not None and not self.__valuesDirty and summed == self.__summed:
            return self.__A

        names = [slot for _, slot in summed]
        if self.__patternDirty or names != [slot for _, slot in self.__summed]:
            self.__pattern = self.__Pattern(names)
            self.__patternDirty = False
            self.__patternStamp += 1

        K = slots["K"].matrix
        indices, indptr, positions = self.__pattern
        if [coef for coef, _ in summed] == [1.0]:
            A = K
        else:
            data = np.zeros(indices.size)
            for (coef, slot), position in zip(summed, positions):
                data[position] += coef * slots[slot].matrix.data
            A = sparse.csr_matrix((data, indices, indptr), shape=K.shape)
        A.has_canonical_format = True
        A.data.flags.writeable = False

        self.__A = A
        self.__summed = summed
        self.__valuesDirty = False
        self.__stamp += 1
        return A

    def __Pattern(
        self, names: list[str]
    ) -> tuple[_types.IntArray, _types.IntArray, list[_types.IntArray | slice]]:
        """Union of the slots' patterns: `(indices, indptr)`, each slot's positions in it; exact zeros stay, so A's pattern never follows its values."""
        slots = [self.__slots[name] for name in names]
        K = slots[0].matrix
        if all(slot.key == slots[0].key for slot in slots):
            return K.indices, K.indptr, [slice(None)] * len(slots)

        # entries as row·n + col, sorted: the canonical CSR order
        n = K.shape[0]
        keys = [
            np.repeat(np.arange(n, dtype=np.int64), np.diff(X.indptr)) * n + X.indices
            for X in (slot.matrix for slot in slots)
        ]
        canon = np.unique(np.concatenate(keys))
        union = sparse.csr_matrix((np.ones(canon.size), divmod(canon, n)), shape=(n, n))
        positions = [np.searchsorted(canon, key) for key in keys]
        return union.indices, union.indptr, positions
