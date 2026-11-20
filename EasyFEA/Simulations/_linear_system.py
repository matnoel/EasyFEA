# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Linear system `A x = b` of one problem."""

from typing import Callable, NamedTuple

import numpy as np
from scipy import sparse

from ..Utilities import _types
from ..Utilities._mpi import (
    MPI_SIZE,
    Exchange,
    Owned,
    Build_exchange,
    Scatter_forward,
    Exscan_sum,
    Reduce_max,
    Reduce_sum,
)
from ._terms import Same_contributions, _Contribution
from .Solvers import (
    ResolType,
    SolverType,
    _Check_saddle_point,
    _PETScSolver,
    _Solve_Axb,
    _Solver,
)


class _Lagrange(NamedTuple):
    """A condition `coefs · x[dofs] = value`."""

    dofs: _types.IntArray
    coefs: _types.FloatArray
    value: float


class _LinearSystem:
    """`A x = b` of one problem key, dropped with the mesh: last contributions, A, numbering, dof exchange and kept PETSc solver; `comm` is passed in, never stored."""

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
        self.__known: _types.BoolArray | None = None
        self.rows: _types.IntArray = np.empty(0, dtype=int)
        """the owned unknowns, in PETSc order"""
        self.ids: _types.IntArray = np.empty(0, dtype=np.int64)
        """PETSc id of each dof, -1 for a known one"""
        self.rowIds: _types.IntArray = np.empty(0, dtype=np.int64)
        """`ids`, -1 at the ghost rows"""
        self.__petsc = _PETScSolver()
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
            "_LinearSystem__petsc": _PETScSolver(),
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

    def Free_solver(self) -> None:
        """Destroys the kept PETSc solver. Collective."""
        self.__petsc.Free()

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

        self.__AKey = key
        A = _Sum(list_contribution, coefs, Ndof, Get_map)
        A.data.flags.writeable = False

        self.__A = A
        self.__ACoefs = coefs
        self.__list_contributionA = list_contribution
        return A

    def Solve(
        self,
        b: _types.FloatArray,
        x: _types.FloatArray,
        known: _types.BoolArray,
        solver: _Solver,
        comm,
        lagrange: list[_Lagrange] | None = None,
        resolution: ResolType = ResolType.r1,
    ) -> float:
        """Solves the kept A's `A x = b` for x's unknowns in place, x holding the known values and 0 elsewhere; the residual norm over the owned dof rows. Collective on `comm`."""
        A = self.__A
        assert A is not None
        lagrange = lagrange or []
        size = b.size - len(lagrange)
        if lagrange:
            if resolution != ResolType.r2:
                raise NotImplementedError(
                    f"{resolution} does not support Lagrange conditions."
                )
            A, b = _Border(A, b, lagrange)
        # the known values of the ghosts from their owner
        self.Scatter_forward(x, comm)
        if resolution == ResolType.r3:
            A, b, known = _Penalty(A, b, x, known)
        self.Set_known(known, comm)

        _Check_saddle_point(
            solver, lambda: bool(lagrange) or self.__Has_zero_diagonal(A, comm)
        )

        residual = b - A @ x
        # the multipliers past the dofs
        rhsNorm = _Residual_norm(residual[self.rows[self.rows < size]], comm)

        x0 = np.zeros(b.size)
        x0[: solver.x0.size] = solver.x0
        if solver.type == SolverType.petsc:
            x[self.rows] += self.__petsc.Solve(
                A,
                residual,
                x0,
                self.ids,
                self.rowIds,
                solver.options,
                comm,
            )
        else:
            unknown = np.flatnonzero(~known)
            x[unknown] += _Solve_Axb(
                A[unknown][:, unknown],
                residual[unknown],
                solver._replace(x0=x0[unknown]),
            )
        self.Scatter_forward(x, comm)
        return rhsNorm

    def __Has_zero_diagonal(self, A: sparse.csr_matrix, comm) -> bool:
        """Whether a numbered row has no diagonal entry, on any rank (a saddle point); only owned rows are complete under MPI."""
        diagonal = A.diagonal()[self.rows]
        return Reduce_sum(int(np.count_nonzero(diagonal == 0)), comm) > 0

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
    # canonical by construction: no sort before the backends
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


def _Border(
    A: sparse.csr_matrix, b: _types.FloatArray, lagrange: list[_Lagrange]
) -> tuple[sparse.csr_matrix, _types.FloatArray]:
    """A bordered by the Lagrange conditions and b's multiplier rows, both scaled by `max|A|`."""
    if MPI_SIZE > 1:
        raise NotImplementedError(
            "Simulations using Lagrangian boundary conditions are not yet supported in MPI environments."
        )
    alpha = float(np.abs(A.data).max(initial=0.0))
    size = b.size - len(lagrange)
    b = b.copy()
    rows, cols, values = [], [], []
    for i, condition in enumerate(lagrange, size):
        line = np.full(condition.dofs.size, i)
        rows += [line, condition.dofs]
        cols += [condition.dofs, line]
        values += [alpha * condition.coefs] * 2
        b[i] = alpha * condition.value
    # A's entries kept, explicit zeros included: the same pattern, the same ordering
    coo = A.tocoo()
    bordered = sparse.csr_matrix(
        (
            np.concatenate([coo.data, *values]),
            (np.concatenate([coo.row, *rows]), np.concatenate([coo.col, *cols])),
        ),
        shape=A.shape,
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
