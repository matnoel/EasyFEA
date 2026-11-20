# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Assembly of one problem's contributions into A and b."""

from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from scipy import sparse

from ..FEM._function_space import _FunctionSpace
from ..Utilities import _types
from ._problem_type import ProblemType
from ._terms import Same_contributions, Term, _Contribution

if TYPE_CHECKING:
    from ..FEM import _GroupElem
    from ._simu import _Simu


class _Assembly:
    """Contributions of one problem key and what is summed from them (A, b, products), with its CSR maps and pinned dofs; dropped with the mesh."""

    def __init__(self) -> None:
        self.__list_contribution: list[_Contribution] = []
        self.__A: sparse.csr_matrix | None = None
        self.__AKey: tuple = ()
        """`(layout, Ndof, contributing)` of A's map"""
        self.__ACoefs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.__list_contributionA: list[_Contribution] = []
        """the K, C and M contributions A was summed from"""
        self.__maps: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        """`{(layout, Ndof, contributing): csr map}`"""
        self.__pinnedFor: _types.BoolArray | None = None
        """the free dofs the pinned ones were found for"""
        self.__pinnedDofs: _types.IntArray = np.empty(0, dtype=np.int64)
        self.__pinnedWeights: dict = {}
        self.__dofsLayout: tuple = ()
        self.__dict_dofs_e: dict[tuple[str, _GroupElem], _types.IntArray] = {}
        """`{(field, group): dofs_e}` on `__dofsLayout`"""
        self.__connectedKey: tuple = ()
        self.__connected: _types.BoolArray | None = None

    def __getstate__(self) -> dict:
        # a term's bound method does not pickle; a copy rebuilds its contributions
        return {
            **self.__dict__,
            "_Assembly__list_contribution": [],
            "_Assembly__A": None,
            "_Assembly__list_contributionA": [],
            "_Assembly__dict_dofs_e": {},
            "_Assembly__connected": None,
        }

    def Set_list_contribution(self, list_contribution: list[_Contribution]) -> None:
        """Keeps the last contributions, A's source."""
        self.__list_contribution = list_contribution

    def Connected_rows(self, layout: tuple, Ndof: int) -> _types.BoolArray:
        """Rows a K, C or M contribution of the last contributions fills, ghost elements included; a copy."""
        list_contribution = [c for c in self.__list_contribution if c.slot in "KCM"]
        key = (layout, Ndof, _Contributing(list_contribution))
        if self.__connected is None or key != self.__connectedKey:
            rows = np.zeros(Ndof, dtype=bool)
            for c in list_contribution:
                rows[self.__Dofs_e(layout, c.block[0], c.group)] = True
            self.__connected = rows
            self.__connectedKey = key
        return self.__connected.copy()

    def Get_A(
        self, coefs: tuple[float, float, float], layout: tuple, Ndof: int
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

        self.__AKey = key
        A = _Sum(list_contribution, coefs, Ndof, self.__Map_getter(layout, Ndof))
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__list_contributionA = list_contribution
        return A

    def __Map_getter(
        self, layout: tuple, Ndof: int
    ) -> Callable[[tuple], tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """The CSR map of the contributing blocks on `layout`, kept per key."""

        def Get_map(contributing: tuple):
            key = (layout, Ndof, contributing)
            csrMap = self.__maps.get(key)
            if csrMap is None:
                csrMap = _Csr_map(layout, Ndof, contributing)
                self.__maps[key] = csrMap
            return csrMap

        return Get_map

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

    def Get_operator(self, slot: str, layout: tuple, Ndof: int) -> sparse.csr_matrix:
        """The last contributions' K, C or M alone, a new matrix on A's map."""
        list_contribution = [c for c in self.__list_contribution if c.slot in "KCM"]
        coefK, coefC, coefM = (float(s == slot) for s in "KCM")
        return _Sum(
            list_contribution,
            (coefK, coefC, coefM),
            Ndof,
            self.__Map_getter(layout, Ndof),
        )

    def Get_load(self, layout: tuple, Ndof: int) -> _types.FloatArray:
        """The last F contributions, summed: the load the terms declare."""
        pieces = [
            (c, c.coef * c.array) for c in self.__list_contribution if c.slot == "F"
        ]
        return self.__Scatter(pieces, layout, Ndof)

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
        return load + self.__Scatter(
            self.__Pieces(fields, layout, True), layout, load.size
        )

    def Get_product(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        Ndof: int,
    ) -> _types.FloatArray:
        """`Σ coef·X_e·field` over the last contributions' K, C and M, each slot on its field; a None field is skipped."""
        return self.__Scatter(self.__Pieces(fields, layout, False), layout, Ndof)

    def __Pieces(
        self,
        fields: tuple[_types.FloatArray | None, ...],
        layout: tuple,
        forB: bool,
    ) -> list[tuple[_Contribution, np.ndarray]]:
        """Element vectors of `coef·X_e·field`, or of b's `−coef·R` and `−coef·X_e·field` over the linear terms."""
        # contributions sharing one array (a term and its scaled copies) take one matmul on their summed fields
        shared: dict[tuple, list] = {}
        pieces = []
        for c in self.__list_contribution:
            if c.slot == "R":
                if forB:
                    pieces.append((c, -c.coef * c.array))
            elif c.slot in "KCM" and not (forB and "R" in c.term.slots):
                field = fields["KCM".index(c.slot)]
                if field is None:
                    continue
                coef = -c.coef if forB else c.coef
                key = (id(c.array), c.block, c.group)
                if key in shared:
                    shared[key][1] += coef * field
                else:
                    shared[key] = [c, coef * field]
        for c, field in shared.values():
            x_e = field[self.__Dofs_e(layout, c.block[1], c.group)]
            pieces.append((c, np.matmul(c.array, x_e[..., None])[..., 0]))
        return pieces

    def __Dofs_e(
        self, layout: tuple, field: str, group: "_GroupElem"
    ) -> _types.IntArray:
        """`field`'s dofs on each element of `group`, in the problem's vector; kept while `layout` holds."""
        if layout != self.__dofsLayout:
            self.__dofsLayout = layout
            self.__dict_dofs_e = {}
        dofs_e = self.__dict_dofs_e.get((field, group))
        if dofs_e is None:
            dofs_e = _Dofs_e(layout, field, group)
            dofs_e.flags.writeable = False
            self.__dict_dofs_e[field, group] = dofs_e
        return dofs_e

    def __Scatter(
        self, pieces: list[tuple[_Contribution, np.ndarray]], layout: tuple, Ndof: int
    ) -> _types.FloatArray:
        """Element vectors summed into a vector of `Ndof`, each into its contribution's rows."""
        if not pieces:
            return np.zeros(Ndof)
        rows = np.concatenate(
            [self.__Dofs_e(layout, c.block[0], c.group).ravel() for c, _ in pieces]
        )
        data = np.concatenate([vector.ravel() for _, vector in pieces])
        if np.iscomplexobj(data):
            raise TypeError("Complex systems are not supported.")
        return np.bincount(rows, weights=data, minlength=Ndof)


def _Dofs_e(layout: tuple, field, group) -> _types.IntArray:
    """`field`'s dofs on each element of `group`, in the problem's vector."""
    for problemType, space, dof_n, offset in layout:
        if problemType == field:
            return space.Get_assembly_e(group, dof_n) + offset
    raise KeyError(field)


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
    # canonical by construction: no sort before the backends
    A.has_canonical_format = True
    return A


def _Csr_map(
    layout: tuple[tuple[ProblemType, _FunctionSpace, int, int], ...],
    Ndof: int,
    contributing: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """`(inv, indices, indptr, nnz)`: `inv[k]` the CSR-data slot element entry `k` feeds, the rest scipy's canonical pattern; `inv` int32, slots bounded by nnz < 2^31."""
    byField = {
        problemType: (functionSpace, dof_n, offset)
        for problemType, functionSpace, dof_n, offset in layout
    }

    list_rows, list_cols = [], []
    for (r, c), groups in contributing:
        space_r, dof_n_r, offset_r = byField[r]
        space_c, dof_n_c, offset_c = byField[c]
        for groupElem in groups:
            rows_e = space_r.Get_assembly_e(groupElem, dof_n_r) + offset_r
            cols_e = space_c.Get_assembly_e(groupElem, dof_n_c) + offset_c
            list_rows.append(
                _FunctionSpace._Get_rows_e(rows_e, cols_e.shape[1]).ravel()
            )
            list_cols.append(
                _FunctionSpace._Get_columns_e(cols_e, rows_e.shape[1]).ravel()
            )
    rows = np.concatenate(list_rows)
    cols = np.concatenate(list_cols)

    # scipy's own pattern, from a structure-only matrix
    matrix = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(Ndof, Ndof))
    matrix.sort_indices()

    # each element entry to the slot of its (row, col), by the sorted row-major index
    canon = np.repeat(np.arange(Ndof), np.diff(matrix.indptr)) * Ndof + matrix.indices
    inv = np.searchsorted(canon, rows.astype(np.int64) * Ndof + cols).astype(np.int32)
    return inv, matrix.indices, matrix.indptr, matrix.nnz


def _Dof_layout(
    simu: "_Simu", problemTypes: tuple[ProblemType, ...]
) -> tuple[tuple[tuple[ProblemType, _FunctionSpace, int, int], ...], int]:
    """`(problemType, space, dof_n, offset)` of each field, side by side in the given order, and Ndof, every Lagrange dof after them."""
    layout = []
    offset = 0
    for problemType in problemTypes:
        space = simu._Get_FunctionSpace(problemType)
        dof_n = simu.Get_dof_n(problemType)
        layout.append((problemType, space, dof_n, offset))
        offset += space.Nn * dof_n
    Ndof = offset + sum(
        simu._Bc_Lagrange_dim(problemType) for problemType in problemTypes
    )
    return tuple(layout), Ndof


def _Blocks(
    terms: dict[ProblemType, list[Term]],
) -> Iterator[tuple[str, ProblemType, ProblemType]]:
    """`(slot, rows, cols)` of each array the terms declare, a vector's `cols` its rows."""
    for problemType, fieldTerms in terms.items():
        for term in fieldTerms:
            for slot, (rows, cols) in zip(term.slots, term._Get_blocks(problemType)):
                yield slot, rows, cols
