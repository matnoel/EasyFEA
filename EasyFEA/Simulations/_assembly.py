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


class _Assembly:
    """What one problem's contributions sum to (A, b, products) on its fixed `layout`, with their CSR maps; Lagrange rows excluded."""

    def __init__(
        self, layout: tuple[tuple[ProblemType, _FunctionSpace, int, int], ...]
    ) -> None:
        self.layout = layout
        """`(field, space, dof_n, offset)` of each field, side by side"""
        self.size = sum(space.Nn * dof_n for _, space, dof_n, _ in layout)
        """number of dofs of the fields"""
        self.__A: sparse.csr_matrix | None = None
        self.__AKey: tuple = ()
        """`contributing` of A's map"""
        self.__ACoefs: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.__list_contributionA: list[_Contribution] = []
        """the K, C and M contributions A was summed from"""
        self.__maps: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        """`{contributing: csr map}`"""
        self.__dict_dofs_e: dict[tuple[str, _GroupElem], _types.IntArray] = {}
        """`{(field, group): dofs_e}`"""
        self.__connectedKey: tuple = ()
        self.__connected: _types.BoolArray | None = None

    def Connected_rows(
        self, list_contribution: list[_Contribution]
    ) -> _types.BoolArray:
        """Rows a K, C or M contribution fills, ghost elements included; a copy."""
        matrices = [
            contribution
            for contribution in list_contribution
            if contribution.slot in "KCM"
        ]
        key = _Contributing(matrices)
        if self.__connected is None or key != self.__connectedKey:
            rows = np.zeros(self.size, dtype=bool)
            for contribution in matrices:
                rows[self.__Dofs_e(contribution.block[0], contribution.group)] = True
            self.__connected = rows
            self.__connectedKey = key
        return self.__connected.copy()

    def Get_A(
        self,
        list_contribution: list[_Contribution],
        coefs: tuple[float, float, float],
    ) -> sparse.csr_matrix:
        """`coefK·K + coefC·C + coefM·M` from the contributions, on the map of every group it fills; the same read-only object while its K, C, M are the same cached arrays and `coefs` hold."""
        matrices = [
            contribution
            for contribution in list_contribution
            if contribution.slot in "KCM"
        ]
        key = _Contributing(matrices)
        if (
            self.__A is not None
            and key == self.__AKey
            and coefs == self.__ACoefs
            and Same_contributions(self.__list_contributionA, matrices)
        ):
            return self.__A

        self.__AKey = key
        A = _Sum(matrices, coefs, self.size, self.__Get_map)
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__list_contributionA = matrices
        return A

    def __Get_map(
        self, contributing: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """The CSR map of the contributing blocks, kept per `contributing`."""
        csrMap = self.__maps.get(contributing)
        if csrMap is None:
            csrMap = _Csr_map(self.layout, self.size, contributing)
            self.__maps[contributing] = csrMap
        return csrMap

    def Get_operator(
        self, list_contribution: list[_Contribution], slot: str
    ) -> sparse.csr_matrix:
        """The contributions' K, C or M alone, a new matrix on A's map."""
        matrices = [
            contribution
            for contribution in list_contribution
            if contribution.slot in "KCM"
        ]
        coefK, coefC, coefM = (float(s == slot) for s in "KCM")
        return _Sum(matrices, (coefK, coefC, coefM), self.size, self.__Get_map)

    def Get_load(self, list_contribution: list[_Contribution]) -> _types.FloatArray:
        """The F contributions, summed: the load the terms declare."""
        pieces = [
            (contribution, contribution.coef * contribution.array)
            for contribution in list_contribution
            if contribution.slot == "F"
        ]
        return self.__Scatter(pieces)

    def Get_b(
        self,
        list_contribution: list[_Contribution],
        load: _types.FloatArray,
        fields: tuple[_types.FloatArray | None, ...],
    ) -> _types.FloatArray:
        """`load − Σ coef·R − Σ coef·X_e·field` over the contributions, X_e of the terms declaring no R; a None field is skipped."""
        return load + self.__Scatter(self.__Pieces(list_contribution, fields, True))

    def Get_product(
        self,
        list_contribution: list[_Contribution],
        fields: tuple[_types.FloatArray | None, ...],
    ) -> _types.FloatArray:
        """`Σ coef·X_e·field` over the contributions' K, C and M, each slot on its field; a None field is skipped."""
        return self.__Scatter(self.__Pieces(list_contribution, fields, False))

    def __Pieces(
        self,
        list_contribution: list[_Contribution],
        fields: tuple[_types.FloatArray | None, ...],
        forB: bool,
    ) -> list[tuple[_Contribution, np.ndarray]]:
        """Element vectors of `coef·X_e·field`, or of b's `−coef·R` and `−coef·X_e·field` over the linear terms."""
        # contributions sharing one array (a term and its scaled copies) take one matmul on their summed fields
        shared: dict[tuple, list] = {}
        pieces = []
        for contribution in list_contribution:
            if contribution.slot == "R":
                if forB:
                    pieces.append(
                        (contribution, -contribution.coef * contribution.array)
                    )
            elif contribution.slot in "KCM" and not (
                forB and "R" in contribution.term.slots
            ):
                field = fields["KCM".index(contribution.slot)]
                if field is None:
                    continue
                coef = -contribution.coef if forB else contribution.coef
                key = (id(contribution.array), contribution.block, contribution.group)
                if key in shared:
                    shared[key][1] += coef * field
                else:
                    shared[key] = [contribution, coef * field]
        for contribution, field in shared.values():
            x_e = field[self.__Dofs_e(contribution.block[1], contribution.group)]
            pieces.append(
                (contribution, np.matmul(contribution.array, x_e[..., None])[..., 0])
            )
        return pieces

    def __Dofs_e(self, field: str, group: "_GroupElem") -> _types.IntArray:
        """`field`'s dofs on each element of `group`, in the problem's vector; kept."""
        dofs_e = self.__dict_dofs_e.get((field, group))
        if dofs_e is None:
            dofs_e = _Dofs_e(self.layout, field, group)
            dofs_e.flags.writeable = False
            self.__dict_dofs_e[field, group] = dofs_e
        return dofs_e

    def __Scatter(
        self, pieces: list[tuple[_Contribution, np.ndarray]]
    ) -> _types.FloatArray:
        """Element vectors summed into a vector of `size`, each into its contribution's rows."""
        if not pieces:
            return np.zeros(self.size)
        rows = np.concatenate(
            [
                self.__Dofs_e(contribution.block[0], contribution.group).ravel()
                for contribution, _ in pieces
            ]
        )
        data = np.concatenate([vector.ravel() for _, vector in pieces])
        if np.iscomplexobj(data):
            raise TypeError("Complex systems are not supported.")
        return np.bincount(rows, weights=data, minlength=self.size)


def _Dofs_e(layout: tuple, field, group) -> _types.IntArray:
    """`field`'s dofs on each element of `group`, in the problem's vector."""
    for problemType, space, dof_n, offset in layout:
        if problemType == field:
            return space.Get_assembly_e(group, dof_n) + offset
    raise KeyError(field)


def _Sizes(list_contribution: list[_Contribution]) -> dict:
    """`{block: {group: entries}}` of every block `list_contribution` fills, the order of its map."""
    sizes: dict = {}
    for contribution in list_contribution:
        sizes.setdefault(contribution.block, {}).setdefault(
            contribution.group, contribution.array.size
        )
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
    for contribution in list_contribution:
        weight = coefs["KCM".index(contribution.slot)]
        if weight == 0:
            continue
        if np.iscomplexobj(contribution.array):
            raise TypeError("Complex systems are not supported.")
        start = spans[contribution.block, contribution.group]
        entries[start : start + contribution.array.size] += (
            weight * contribution.coef
        ) * contribution.array.ravel()

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


def _Blocks(
    terms: dict[ProblemType, list[Term]],
) -> Iterator[tuple[str, ProblemType, ProblemType]]:
    """`(slot, rows, cols)` of each array the terms declare, a vector's `cols` its rows."""
    for problemType, fieldTerms in terms.items():
        for term in fieldTerms:
            for slot, (rows, cols) in zip(term.slots, term._Get_blocks(problemType)):
                yield slot, rows, cols
