# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear system `A x = b` of one problem."""

from typing import Callable

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


class _LinearSystem:
    """`A x = b` of one problem key, dropped with the mesh: last fold, A, PETSc numbering, dof exchange and kept solver; `comm` is passed in, never stored."""

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
        self.__fold: list[_Contribution] = []
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
        # PETSc objects and graph communicators do not pickle, nor a term's bound method; a copy refolds
        return {
            **self.__dict__,
            "_LinearSystem__fold": [],
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

    def Set_fold(self, fold: list[_Contribution]) -> None:
        """Keeps the last fold, A's source."""
        self.__fold = fold

    def Connected_rows(self, layout: tuple, Ndof: int) -> _types.BoolArray:
        """Rows a K, C or M contribution of the last fold fills, ghost elements included."""
        rows = np.zeros(Ndof, dtype=bool)
        for c in self.__fold:
            if c.slot in "KCM":
                rows[_Dofs_e(layout, c.block[0], c.group)] = True
        return rows

    def Get_A(
        self,
        coefs: tuple[float, float, float],
        layout: tuple,
        Ndof: int,
        Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    ) -> sparse.csr_matrix:
        """`coefK·K + coefC·C + coefM·M` from the last fold, on the map of every group it fills; the same read-only object while its K, C, M are the same cached arrays and `coefs` hold."""
        fold = [c for c in self.__fold if c.slot in "KCM"]
        key = (layout, Ndof, _Contributing(fold))
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

        A = _Sum(fold, coefs, Ndof, Get_map)
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__AFold = fold
        self.__stamp += 1
        return A

    def Get_operator(
        self,
        slot: str,
        layout: tuple,
        Ndof: int,
        Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    ) -> sparse.csr_matrix:
        """The last fold's K, C or M alone, a new matrix on A's map."""
        fold = [c for c in self.__fold if c.slot in "KCM"]
        coefK, coefC, coefM = (float(s == slot) for s in "KCM")
        return _Sum(fold, (coefK, coefC, coefM), Ndof, Get_map)

    def Get_load(self, layout: tuple, Ndof: int) -> _types.FloatArray:
        """The last fold's F contributions, summed: the load the terms declare."""
        pieces = [(c, c.coef * c.array) for c in self.__fold if c.slot == "F"]
        return _Scatter(pieces, layout, Ndof)

    @property
    def internal(self) -> bool:
        """Whether the last fold holds an internal force R."""
        return any(c.slot == "R" for c in self.__fold)

    def Get_b(
        self,
        load: _types.FloatArray,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
    ) -> _types.FloatArray:
        """`load − Σ coef·R − Σ coef·X_e·field` over the last fold, X_e of the terms declaring no R; a None field is skipped."""
        return load + _Scatter(self.__Pieces(fields, layout, True), layout, load.size)

    def Get_product(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        Ndof: int,
    ) -> _types.FloatArray:
        """`Σ coef·X_e·field` over the last fold's K, C and M, each slot on its field; a None field is skipped."""
        return _Scatter(self.__Pieces(fields, layout, False), layout, Ndof)

    def __Pieces(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        isB: bool,
    ) -> list[tuple[_Contribution, np.ndarray]]:
        """Element vectors of `coef·X_e·field`, or of b's `−coef·R` and `−coef·X_e·field` over the linear terms."""
        pieces = []
        for c in self.__fold:
            if c.slot == "R":
                if isB:
                    pieces.append((c, -c.coef * c.array))
            elif c.slot in "KCM" and not (isB and "R" in c.term.slots):
                field = fields["KCM".index(c.slot)]
                if field is None:
                    continue
                x_e = field[_Dofs_e(layout, c.block[1], c.group)]
                coef = -c.coef if isB else c.coef
                pieces.append((c, coef * np.matmul(c.array, x_e[..., None])[..., 0]))
        return pieces


def _Contributing(fold: list[_Contribution]) -> tuple:
    """`(block, groups)` of every block `fold` fills, the order of its map."""
    sizes: dict = {}
    for c in fold:
        sizes.setdefault(c.block, {}).setdefault(c.group, c.array.size)
    return tuple((block, tuple(groups)) for block, groups in sizes.items())


def _Sum(
    fold: list[_Contribution],
    coefs: tuple[float, float, float],
    Ndof: int,
    Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
) -> sparse.csr_matrix:
    """`Σ coef_slot·coef·X_e` over `fold`'s K, C, M on their union map: a zero-coef slot in the pattern, not in the values."""
    sizes: dict = {}
    for c in fold:
        sizes.setdefault(c.block, {}).setdefault(c.group, c.array.size)
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
        inv, indices, indptr, nnz = Get_map(_Contributing(fold))
        assert entries.size == inv.size, "A's map does not match its contributions."
        data = np.bincount(inv, weights=entries, minlength=nnz)
        A = sparse.csr_matrix((data, indices, indptr), shape=(Ndof, Ndof))
    # canonical by construction: Solvers skips its fixup
    A.has_canonical_format = True
    return A


def _Dofs_e(layout: tuple, field, group) -> _types.IntArray:
    """`field`'s dofs on each element of `group`, in the problem's vector."""
    for problemType, space, dof_n, offset in layout:
        if problemType == field:
            return space.Get_assembly_e(group, dof_n) + offset
    raise KeyError(field)


def _Scatter(
    pieces: list[tuple[_Contribution, np.ndarray]], layout: tuple, Ndof: int
) -> _types.FloatArray:
    """Element vectors summed into a vector of `Ndof`, each into its contribution's rows."""
    if not pieces:
        return np.zeros(Ndof)
    rows = np.concatenate(
        [_Dofs_e(layout, c.block[0], c.group).ravel() for c, _ in pieces]
    )
    data = np.concatenate([vector.ravel() for _, vector in pieces])
    if np.iscomplexobj(data):
        raise TypeError("Complex systems are not supported.")
    return np.bincount(rows, weights=data, minlength=Ndof)
