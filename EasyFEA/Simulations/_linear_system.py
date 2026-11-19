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
from ._terms import Same_contributions, _Contribution


class _Slot(NamedTuple):
    """One assembled K, C, M or F: its csr map key, its contributions, the matrix."""

    key: tuple
    contributions: list[_Contribution]
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
        self.__fold: list[_Contribution] = []
        self.__folded = False
        self.__A: sparse.csr_matrix | None = None
        self.__AKey: tuple = ()
        """`(layout, Ndof, contributing)` of A's map"""
        self.__ACoefs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.__AFold: list[_Contribution] = []
        """the K, C and M contributions A was summed from"""
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
        # PETSc objects and graph communicators do not pickle, nor a term's bound method; a copy assembles into its own slots
        return {
            **self.__dict__,
            "_LinearSystem__slots": {
                slot: kept._replace(contributions=[])
                for slot, kept in self.__slots.items()
            },
            "_LinearSystem__fold": [],
            "_LinearSystem__folded": False,
            "_LinearSystem__A": None,
            "_LinearSystem__AFold": [],
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
        contributions: list[_Contribution],
        assemble: Callable[[], sparse.csr_matrix],
    ) -> sparse.csr_matrix:
        """The slot's matrix, assembled again unless a K, C or M of the same cached arrays on the same map."""
        old = self.__slots.get(slot)
        if (
            slot != "F"
            and old is not None
            and old.key == key
            and Same_contributions(old.contributions, contributions)
        ):
            return old.matrix
        matrix = assemble()
        self.__slots[slot] = _Slot(key, contributions, matrix)
        return matrix

    def Set_fold(self, fold: list[_Contribution]) -> None:
        """Keeps the last fold, A's source."""
        self.__fold = fold
        self.__folded = True

    @property
    def folded(self) -> bool:
        """Whether the system holds a fold: a pickled copy does not."""
        return self.__folded

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

    def Get_A(
        self,
        coefs: tuple[float, float, float],
        layout: tuple,
        Ndof: int,
        Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    ) -> sparse.csr_matrix:
        """`coefK·K + coefC·C + coefM·M` from the last fold, on the map of every group it fills; the same read-only object while its K, C, M are the same cached arrays and `coefs` hold."""
        self.__Check_assembled()
        fold = [c for c in self.__fold if c.slot in "KCM"]
        sizes: dict = {}
        for c in fold:
            sizes.setdefault(c.block, {}).setdefault(c.group, c.array.size)
        contributing = tuple((block, tuple(groups)) for block, groups in sizes.items())
        key = (layout, Ndof, contributing)
        if (
            self.__A is not None
            and key == self.__AKey
            and coefs == self.__ACoefs
            and Same_contributions(self.__AFold, fold)
        ):
            return self.__A

        if key != self.__AKey:
            self.__AKey = key
            self.__patternStamp += 1

        # each (block, group)'s element entries, end to end in the map's order
        spans = {}
        size = 0
        for block, groups in sizes.items():
            for group, n in groups.items():
                spans[block, group] = size
                size += n
        entries = np.zeros(size)
        for c in fold:
            weight = coefs["KCM".index(c.slot)]
            if weight == 0:
                continue
            if np.iscomplexobj(c.array):
                raise TypeError("Complex systems are not supported.")
            start = spans[c.block, c.group]
            entries[start : start + c.array.size] += (weight * c.coef) * c.array.ravel()

        # an MPI rank may own no element of any contributing group
        if size == 0:
            A = sparse.csr_matrix((Ndof, Ndof))
        else:
            inv, indices, indptr, nnz = Get_map(contributing)
            assert entries.size == inv.size, "A's map does not match its contributions."
            data = np.bincount(inv, weights=entries, minlength=nnz)
            A = sparse.csr_matrix((data, indices, indptr), shape=(Ndof, Ndof))
        # canonical by construction: Solvers skips its fixup
        A.has_canonical_format = True
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__AFold = fold
        self.__stamp += 1
        return A
