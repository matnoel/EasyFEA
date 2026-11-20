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


class _Entries(NamedTuple):
    """Sparse entries, summed where they repeat."""

    rows: _types.IntArray
    cols: _types.IntArray
    values: _types.FloatArray

    @staticmethod
    def Empty() -> "_Entries":
        return _Entries(np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0))

    @staticmethod
    def Concatenate(pieces: list["_Entries"]) -> "_Entries":
        if not pieces:
            return _Entries.Empty()
        return _Entries(*(np.concatenate(arrays) for arrays in zip(*pieces)))


class _LinearSystem:
    """`A x = b` of one problem key, dropped with the mesh: last contributions, A, PETSc numbering, dof exchange and kept solver; `comm` is passed in, never stored."""

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
        self.__list_contribution: list[_Contribution] = []
        self.__A: sparse.csr_matrix | None = None
        self.__AKey: tuple = ()
        """`(layout, Ndof, contributing)` of A's map"""
        self.__ACoefs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.__list_contributionA: list[_Contribution] = []
        """the K, C and M contributions A was summed from"""
        self.__stamp = 0
        self.__patternStamp = 0
        self.__known: _types.BoolArray | None = None
        self.__lagrange: tuple = ()
        self.rows: _types.IntArray = np.empty(0, dtype=int)
        """the owned unknowns, in PETSc order"""
        self.ids: _types.IntArray = np.empty(0, dtype=np.int64)
        """PETSc id of each dof, -1 for a known one"""
        self.rowIds: _types.IntArray = np.empty(0, dtype=np.int64)
        """`ids`, -1 at the ghost rows"""
        self.__numberingStamp = 0
        self.__derived: tuple | None = None
        """`(key, matrix, source, inverse)`: the entries `source` of A's then the extra ones, summed at `inverse`"""
        self.__derivedValues: tuple = ()
        self.__derivedPatternStamp = 0
        self.__derivedStamp = 0
        self.__alpha: tuple[int, float] = (0, 0.0)
        self.__solver: tuple | None = None
        """`(ksp, matrix)`"""
        self.__solverStamps: tuple | None = None
        self.__pinnedFor: _types.BoolArray | None = None
        """the free dofs the pinned ones were found for"""
        self.__pinnedDofs: _types.IntArray = np.empty(0, dtype=np.int64)
        self.__pinnedWeights: dict = {}

    def __getstate__(self) -> dict:
        # PETSc objects and graph communicators do not pickle, nor a term's bound method; a copy rebuilds its contributions
        return {
            **self.__dict__,
            "_LinearSystem__list_contribution": [],
            "_LinearSystem__A": None,
            "_LinearSystem__list_contributionA": [],
            "_LinearSystem__exchange": None,
            "_LinearSystem__derived": None,
            "_LinearSystem__derivedValues": (),
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

    def Set_known(self, known: _types.BoolArray, comm, lagrange: tuple = ()) -> None:
        """Numbers the owned unknowns rank after rank, the multipliers last; kept while `known` and the Lagrange dofs hold. Collective on `comm`."""
        changed = (
            self.__known is None
            or not np.array_equal(known, self.__known)
            or len(lagrange) != len(self.__lagrange)
            or not all(map(np.array_equal, lagrange, self.__lagrange))
        )
        if not Reduce_max(changed, comm):
            return
        owned = np.zeros(known.size, dtype=bool)
        size = known.size if self.__owners is None else self.__owners.size
        owned[:size][Owned(self.__owners, self.__rank)] = True
        # the multipliers on the last rank
        owned[size:] = comm is None or self.__rank == comm.Get_size() - 1
        self.__known = known
        self.__lagrange = lagrange
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

    def __Stamps(self, derived: bool) -> tuple:
        """Pattern, numbering and values of A or of the derived matrix."""
        pattern, values = (
            (("D", self.__derivedPatternStamp), ("D", self.__derivedStamp))
            if derived
            else (("A", self.__patternStamp), ("A", self.__stamp))
        )
        return pattern, self.__numberingStamp, values

    def Solver_update(self, comm, derived: bool = False) -> tuple[bool, bool]:
        """Whether the kept solver must be laid out again (new pattern or numbering) and refilled (new values), for A or the derived matrix. Collective on `comm`."""
        stamps = self.__Stamps(derived)
        kept = self.__solverStamps
        relayout = self.__solver is None or kept is None or stamps[:2] != kept[:2]
        refill = relayout or kept is None or stamps[2] != kept[2]
        relayout, refill = Reduce_sum(np.array([relayout, refill], dtype=int), comm)
        return bool(relayout), bool(refill)

    def Set_solver(self, solver: tuple, derived: bool = False) -> None:
        """Keeps `(ksp, matrix)`, laid out and filled with A or the derived matrix."""
        self.__solver = solver
        self.__solverStamps = self.__Stamps(derived)

    def Take_solver(self) -> tuple | None:
        """The kept solver, no longer kept."""
        solver = self.__solver
        self.__solver = None
        self.__solverStamps = None
        return solver

    def Set_pinned(
        self, free: _types.BoolArray, dofs: _types.IntArray, weights: dict
    ) -> None:
        """Keeps the pinned dofs and weights found for `free`."""
        self.__pinnedFor = free.copy()
        self.__pinnedDofs = dofs
        self.__pinnedWeights = weights

    def Pinned_for(self, free: _types.BoolArray) -> bool:
        """Whether the kept pinned dofs were found for `free`."""
        return self.__pinnedFor is not None and np.array_equal(free, self.__pinnedFor)

    @property
    def pinnedDofs(self) -> _types.IntArray:
        """Dofs pinning a field left free up to a constant."""
        return self.__pinnedDofs

    @property
    def pinnedWeights(self) -> dict:
        """Zero-mean weights `∫ N_i dΩ` of each pinned field."""
        return self.__pinnedWeights

    def Set_list_contribution(self, list_contribution: list[_Contribution]) -> None:
        """Keeps the last contributions, A's source."""
        self.__list_contribution = list_contribution

    def Connected_rows(self, layout: tuple, Ndof: int) -> _types.BoolArray:
        """Rows a K, C or M contribution of the last contributions fills, ghost elements included."""
        rows = np.zeros(Ndof, dtype=bool)
        for c in self.__list_contribution:
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
        """`coefK·K + coefC·C + coefM·M` from the last contributions, on the map of every group it fills; the same read-only object while its K, C, M are the same cached arrays and `coefs` hold."""
        list_contribution = [c for c in self.__list_contribution if c.slot in "KCM"]
        key = (layout, Ndof, _Contributing(list_contribution))
        if (
            self.__A is not None
            and key == self.__AKey
            and coefs == self.__ACoefs
            and Same_contributions(self.__list_contributionA, list_contribution)
        ):
            return self.__A

        if key != self.__AKey:
            self.__AKey = key
            self.__patternStamp += 1

        A = _Sum(list_contribution, coefs, Ndof, Get_map)
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__list_contributionA = list_contribution
        self.__stamp += 1
        return A

    def Get_alpha(self) -> float:
        """`max|A.data|` of the kept A, once per A."""
        assert self.__A is not None
        if self.__alpha[0] != self.__stamp:
            self.__alpha = (self.__stamp, float(np.abs(self.__A.data).max(initial=0.0)))
        return self.__alpha[1]

    def Get_derived(
        self,
        extra: _Entries,
        eliminate: bool,
        dropped: _types.BoolArray | None = None,
    ) -> sparse.csr_matrix:
        """The kept A plus `extra`, the `dropped` rows' entries left out, reduced to the unknowns when `eliminate`; one object, refilled in place."""
        A = self.__A
        assert A is not None and self.__known is not None
        rows, cols, values = extra
        key = (
            self.__patternStamp,
            self.__numberingStamp,
            eliminate,
            rows.tobytes(),
            cols.tobytes(),
            None if dropped is None else dropped.tobytes(),
        )
        if self.__derived is None or self.__derived[0] != key:
            Ndof = A.shape[0]
            if eliminate:
                unknown = np.flatnonzero(~self.__known)
                index = np.full(Ndof, -1)
                index[unknown] = np.arange(unknown.size)
            else:
                index = np.arange(Ndof)
            kept = np.ones(Ndof, dtype=bool) if dropped is None else ~dropped
            matrix, source, inverse = _Derive(
                A,
                index,
                kept,
                rows,
                cols,
            )
            self.__derived = (key, matrix, source, inverse)
            self.__derivedValues = ()
            self.__derivedPatternStamp += 1

        _, matrix, source, inverse = self.__derived
        valuesKey = (self.__stamp, values.tobytes())
        if self.__derivedValues != valuesKey:
            entries = np.concatenate([A.data, values])[source]
            matrix.data[:] = np.bincount(inverse, entries, matrix.data.size)
            self.__derivedValues = valuesKey
            self.__derivedStamp += 1
        return matrix

    def Get_operator(
        self,
        slot: str,
        layout: tuple,
        Ndof: int,
        Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    ) -> sparse.csr_matrix:
        """The last contributions' K, C or M alone, a new matrix on A's map."""
        list_contribution = [c for c in self.__list_contribution if c.slot in "KCM"]
        coefK, coefC, coefM = (float(s == slot) for s in "KCM")
        return _Sum(list_contribution, (coefK, coefC, coefM), Ndof, Get_map)

    def Get_load(self, layout: tuple, Ndof: int) -> _types.FloatArray:
        """The last F contributions, summed: the load the terms declare."""
        pieces = [
            (c, c.coef * c.array) for c in self.__list_contribution if c.slot == "F"
        ]
        return _Scatter(pieces, layout, Ndof)

    @property
    def internal(self) -> bool:
        """Whether the last contributions hold an internal force R."""
        return any(c.slot == "R" for c in self.__list_contribution)

    def Get_b(
        self,
        load: _types.FloatArray,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
    ) -> _types.FloatArray:
        """`load − Σ coef·R − Σ coef·X_e·field` over the last contributions, X_e of the terms declaring no R; a None field is skipped."""
        return load + _Scatter(self.__Pieces(fields, layout, True), layout, load.size)

    def Get_product(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        Ndof: int,
    ) -> _types.FloatArray:
        """`Σ coef·X_e·field` over the last contributions' K, C and M, each slot on its field; a None field is skipped."""
        return _Scatter(self.__Pieces(fields, layout, False), layout, Ndof)

    def __Pieces(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        forB: bool,
    ) -> list[tuple[_Contribution, np.ndarray]]:
        """Element vectors of `coef·X_e·field`, or of b's `−coef·R` and `−coef·X_e·field` over the linear terms."""
        pieces = []
        for c in self.__list_contribution:
            if c.slot == "R":
                if forB:
                    pieces.append((c, -c.coef * c.array))
            elif c.slot in "KCM" and not (forB and "R" in c.term.slots):
                field = fields["KCM".index(c.slot)]
                if field is None:
                    continue
                x_e = field[_Dofs_e(layout, c.block[1], c.group)]
                coef = -c.coef if forB else c.coef
                pieces.append((c, coef * np.matmul(c.array, x_e[..., None])[..., 0]))
        return pieces


def _Sizes(list_contribution: list[_Contribution]) -> dict:
    """`{block: {group: entries}}` of every block `list_contribution` fills, the order of its map."""
    sizes: dict = {}
    for c in list_contribution:
        sizes.setdefault(c.block, {}).setdefault(c.group, c.array.size)
    return sizes


def _Contributing(list_contribution: list[_Contribution]) -> tuple:
    """`(block, groups)` of every block `list_contribution` fills, the order of its map."""
    return tuple(
        (block, tuple(groups)) for block, groups in _Sizes(list_contribution).items()
    )


def _Sum(
    list_contribution: list[_Contribution],
    coefs: tuple[float, float, float],
    Ndof: int,
    Get_map: Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]],
) -> sparse.csr_matrix:
    """`Σ coef_slot·coef·X_e` over `list_contribution`'s K, C, M on their union map: a zero-coef slot in the pattern, not in the values."""
    sizes = _Sizes(list_contribution)
    # each (block, group)'s element entries, end to end in the map's order
    spans = {}
    size = 0
    for block, groups in sizes.items():
        for group, n in groups.items():
            spans[block, group] = size
            size += n
    entries = np.zeros(size)
    for c in list_contribution:
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
        contributing = tuple((block, tuple(groups)) for block, groups in sizes.items())
        inv, indices, indptr, nnz = Get_map(contributing)
        assert entries.size == inv.size, "A's map does not match its contributions."
        data = np.bincount(inv, weights=entries, minlength=nnz)
        A = sparse.csr_matrix((data, indices, indptr), shape=(Ndof, Ndof))
    # canonical by construction: Solvers skips its fixup
    A.has_canonical_format = True
    return A


def _Derive(
    A: sparse.csr_matrix,
    index: _types.IntArray,
    kept: _types.BoolArray,
    rows: _types.IntArray,
    cols: _types.IntArray,
) -> tuple[sparse.csr_matrix, _types.IntArray, _types.IntArray]:
    """Unfilled pattern of A's `kept` rows plus the extra `(rows, cols)`, renumbered by `index` (-1 dropped); the entries taken and their positions."""
    rowsA = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    rowsD = np.concatenate([index[rowsA], index[rows]])
    colsD = np.concatenate([index[A.indices], index[cols]])
    isKept = (rowsD >= 0) & (colsD >= 0)
    isKept[: rowsA.size] &= kept[rowsA]
    source = np.flatnonzero(isKept)
    n = int(np.count_nonzero(index >= 0))
    positions = rowsD[source] * n + colsD[source]
    unique, inverse = np.unique(positions, return_inverse=True)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(unique // n, minlength=n), out=indptr[1:])
    matrix = sparse.csr_matrix(
        (np.zeros(unique.size), (unique % n).astype(np.int32), indptr), shape=(n, n)
    )
    matrix.has_canonical_format = True
    return matrix, source, inverse


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
