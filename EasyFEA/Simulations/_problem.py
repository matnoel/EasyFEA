# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Fields of a simulation solved as one system: their system, assembly and contributions."""

from functools import cached_property
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import sparse

from ..FEM import BoundaryCondition, Operators
from ..FEM._function_space import _FunctionSpace
from ..Solvers import Constraints, LagrangeConstraint, System
from ..Utilities import Terminal, Tic, _types
from ..Utilities._mpi import Owned, Reduce_max, Reduce_sum
from ._assembly import _Assembly, _Blocks
from ._problem_type import ProblemType, TrialSolution
from ._terms import _Contribution
from .TimeSchemes import Elliptic

if TYPE_CHECKING:
    from ._simu import _Simu


def _Field_size(simu: "_Simu", field: ProblemType) -> int:
    """Number of dofs of one field."""
    return simu._Get_FunctionSpace(field).Nn * simu.Get_dof_n(field)


class _Problem:
    """Fields of a simulation solved as one system ``[u | p | … | Lagrange]``: its system, assembly and contributions, kept until the mesh changes."""

    def __init__(self, simu: "_Simu", fields: tuple[ProblemType, ...]) -> None:
        if len(set(fields)) < len(fields):
            raise ValueError(f"A field is given twice in {fields}.")
        self.simu = simu
        self.fields = fields
        self.needUpdate = True
        """whether the kept contributions are stale"""
        self.__list_contribution: list[_Contribution] = []
        self.orphanWarned = False
        """the current solve already warned about conditions on orphan dofs"""
        self.__pinnedFor: _types.BoolArray | None = None
        """the free dofs the pinned ones were found for"""
        self.__pinnedDofs: _types.IntArray = np.empty(0, dtype=np.int64)
        self.__pinnedWeights: dict[ProblemType, _types.FloatArray] = {}

    @property
    def problemType(self) -> ProblemType | tuple[ProblemType, ...]:
        """The key of the solver settings: the field alone, else the fields."""
        return self.fields[0] if len(self.fields) == 1 else self.fields

    @property
    def size(self) -> int:
        """Number of dofs of the fields, Lagrange multipliers excluded."""
        return sum(_Field_size(self.simu, field) for field in self.fields)

    @property
    def offsets(self) -> list[int]:
        """Where each field's dofs start."""
        sizes = [_Field_size(self.simu, field) for field in self.fields]
        return [0, *np.cumsum(sizes[:-1]).tolist()]

    @property
    def layout(self) -> tuple[tuple[ProblemType, _FunctionSpace, int, int], ...]:
        """`(field, space, dof_n, offset)` of each field, side by side."""
        layout = []
        offset = 0
        for field in self.fields:
            space = self.simu._Get_FunctionSpace(field)
            dof_n = self.simu.Get_dof_n(field)
            layout.append((field, space, dof_n, offset))
            offset += space.Nn * dof_n
        return tuple(layout)

    @property
    def Ndof(self) -> int:
        """Number of dofs, every Lagrange dof after the fields'."""
        return self.size + sum(self.simu._Bc_Lagrange_dim(f) for f in self.fields)

    @cached_property
    def system(self) -> System:
        """The linear system; a monolithic one refuses a non-elliptic scheme and Lagrange conditions when created."""
        if len(self.fields) > 1:
            self.Check_monolithic()
        owners, rank = self.Get_dof_owners()
        globalIds = None
        if self.simu.mesh.comm is not None:
            globalIds = self.__Get_dof_globalIds()
        return System(owners, rank, globalIds)

    @cached_property
    def assembly(self) -> _Assembly:
        """The assembly on `layout`; a monolithic one refuses a non-elliptic scheme and Lagrange conditions when created."""
        if len(self.fields) > 1:
            self.Check_monolithic()
        return _Assembly(self.layout)

    def Pad(self, A: sparse.csr_matrix) -> sparse.csr_matrix:
        """`A` grown by the zero Lagrange rows and columns; `A` itself without any."""
        Ndof = self.Ndof
        if Ndof == A.shape[0]:
            return A
        indptr = np.concatenate(
            [A.indptr, np.full(Ndof - A.shape[0], A.indptr[-1], A.indptr.dtype)]
        )
        padded = sparse.csr_matrix((A.data, A.indices, indptr), shape=(Ndof, Ndof))
        padded.has_canonical_format = True
        return padded

    def Free(self) -> None:
        """Destroys the solver kept by the system. Collective."""
        if "system" in self.__dict__:
            self.system.Free()

    def Check_monolithic(self) -> None:
        """Rejects what fields solved side by side cannot run yet."""
        names = ", ".join(self.fields)
        if not isinstance(self.simu.timeScheme, Elliptic):
            # each block would need its own time-scheme coefficients
            raise NotImplementedError(
                f"({names}) solved as one system supports the elliptic algorithm only, not {self.simu.timeScheme!r}."
            )
        if len(self.simu.Bc_Lagrange) > 0:
            raise NotImplementedError(
                f"({names}) solved as one system does not support Lagrange boundary conditions."
            )

    def Compose(
        self, get: Callable[[ProblemType], np.ndarray], isDofs: bool = False
    ) -> np.ndarray:
        """`get(field)` of each field end to end, dof ids shifted to the field's offset when `isDofs`; `get(field)` itself for a field alone."""
        if len(self.fields) == 1:
            return get(self.fields[0])
        parts = [np.asarray(get(field)) for field in self.fields]
        if isDofs:
            parts = [
                part.astype(int) + offset for part, offset in zip(parts, self.offsets)
            ]
        return np.concatenate(parts)

    def Get_dof_owners(self) -> tuple[_types.IntArray | None, int]:
        """Owner rank of each dof, None when never partitioned, and this rank."""
        simu = self.simu
        spaces = [simu._Get_FunctionSpace(field) for field in self.fields]
        rank = spaces[0].rank
        if any(space._dofOwners is None for space in spaces):
            return None, rank
        owners = [
            np.repeat(space._dofOwners, simu.Get_dof_n(field))
            for field, space in zip(self.fields, spaces)
        ]
        return np.concatenate(owners), rank

    def Owned_mask(self) -> _types.BoolArray:
        """Whether this rank owns each dof."""
        owners, rank = self.Get_dof_owners()
        isOwned = np.zeros(self.size, dtype=bool)
        isOwned[Owned(owners, rank)] = True
        return isOwned

    def __Get_dof_globalIds(self) -> _types.IntArray:
        """Global id of each dof, unique over the ranks."""
        globalIds = []
        for f, field in enumerate(self.fields):
            space = self.simu._Get_FunctionSpace(field)
            dof_n = self.simu.Get_dof_n(field)
            nodes = space.globalNodes[space._dofNodes].astype(np.int64)
            dofs = nodes[:, None] * dof_n + np.arange(dof_n)
            globalIds.append((dofs * len(self.fields) + f).ravel())
        return np.concatenate(globalIds)

    def Trial(self, x: _types.FloatArray) -> TrialSolution:
        """The trial solution of the iterate `x`: the fields views into `x`, the others committed."""
        trial = self.simu._Committed()
        for field, start in zip(self.fields, self.offsets):
            trial[field] = x[start : start + _Field_size(self.simu, field)]
        return trial

    # ----------------------------------------------
    # Contributions
    # ----------------------------------------------

    def Get_contributions(self, x: TrialSolution | None = None) -> list[_Contribution]:
        """The fields' contributions, kept, rebuilt at `x` (default committed) when stale."""
        if self.needUpdate:
            if x is None:
                x = self.simu._Committed()
            self.__list_contribution = self.simu._Get_list_contribution(self.fields, x)
            self.needUpdate = False
        return self.__list_contribution

    def Get_A(self, x: TrialSolution | None = None) -> sparse.csr_matrix:
        """A for the time scheme, summed from the contributions (rebuilt at `x` when stale) and kept by the assembly."""
        list_contribution = self.Get_contributions(x)
        return self.Pad(
            self.assembly.Get_A(list_contribution, self.simu.timeScheme.coefs)
        )

    def Get_load(self, x: TrialSolution | None = None) -> _types.FloatArray:
        """External load, Neumann + the terms' F, the contributions rebuilt at `x` when stale; Lagrange rows included."""
        simu = self.simu
        assembly = self.assembly
        list_contribution = self.Get_contributions(x)
        Ndof = self.Ndof
        dofs = self.Compose(simu.Bc_dofs_Neumann, isDofs=True)
        dofsValues = self.Compose(simu.Bc_values_Neumann)

        load = np.zeros(Ndof, dtype=float)
        np.add.at(load, np.asarray(dofs, dtype=int), dofsValues)
        load[: self.size] += assembly.Get_load(list_contribution)
        return load

    def Get_b(self, trial: TrialSolution | None = None) -> _types.FloatArray:
        """b of `A dx = b`: the load minus the internal forces, at `trial` (default committed) when nonlinear, at 0 when linear; Lagrange rows included."""
        simu = self.simu
        tic = Tic()

        if trial is None:
            trial = simu._Committed()
        load = self.Get_load(trial)
        list_contribution = self.Get_contributions()
        if simu.isNonLinear:
            x = self.Compose(trial.__getitem__)
        elif any(contribution.slot == "R" for contribution in list_contribution):
            # R is integrated at u_n, not at x = 0
            raise ValueError(
                "A term declaring an internal force R needs the Newton-Raphson algorithm: call `_Solver_Set_Newton_Raphson_Algorithm`."
            )
        else:
            x = np.zeros(self.size)
        u_n = simu._Get_u_n(self.problemType)
        v_n = simu._Get_v_n(self.problemType)
        a_n = simu._Get_a_n(self.problemType)
        fields = tuple(
            field if field is not None and field.any() else None
            for field in simu.timeScheme.Evaluate(x, u_n, v_n, a_n)
        )
        b = load.copy()
        b[: self.size] = self.assembly.Get_b(
            list_contribution, load[: self.size], fields
        )

        tic.Tac("Solver", f"b ({self.problemType})", simu._verbosity)

        return b

    def Get_A_b(
        self, x: TrialSolution | None = None
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]:
        """A and b at `x` (default committed), the contributions rebuilt at `x` when stale."""
        b = self.Get_b(x)
        return self.Get_A(x), b

    def Get_product(
        self, fields: tuple[_types.FloatArray | None, ...]
    ) -> _types.FloatArray:
        """`K·u + C·v + M·a` of the kept contributions, `fields = (u, v, a)`, a None skipped; complete on owned rows only."""
        return self.assembly.Get_product(self.Get_contributions(), fields)

    # ----------------------------------------------
    # Solve
    # ----------------------------------------------

    def Linearize(
        self, x: _types.FloatArray
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]:
        """Nonlinear only: the tangent and residual of `A dx = b`, integrated at the iterate x."""
        self.simu.Need_Update(self.problemType)
        return self.Get_A_b(self.Trial(x))

    def Constraints(self, x: _types.FloatArray | None = None) -> Constraints:
        """The known dofs at the Dirichlet values, orphan and pinned ones at x (default 0), every one at x when the scheme solves for a; the Lagrange constraints."""
        simu = self.simu
        self.__Warn_orphan_conditions()
        known = np.zeros(self.size, dtype=bool)
        known[simu.Bc_dofs_known_unknown(self.problemType)[0]] = True
        knownDofs = np.flatnonzero(known)
        target = np.zeros(self.size) if x is None else x.copy()
        if simu.timeScheme.unknown != "a":
            dofs = self.Compose(simu.Bc_dofs_Dirichlet, isDofs=True)
            dofs = np.asarray(dofs, dtype=int)
            target[dofs] = 0.0
            np.add.at(target, dofs, self.Compose(simu.Bc_values_Dirichlet))

        list_lagrangeConstraint = tuple(
            LagrangeConstraint(
                condition.dofs, condition.lagrangeCoefs, condition.dofsValues[0]
            )
            for condition in simu.Bc_Lagrange
            if condition.problemType in self.fields
        )
        return Constraints(
            knownDofs, target[knownDofs], list_lagrangeConstraint, simu._resolution
        )

    def Orphan_dofs(self) -> _types.IntArray:
        """Dofs in no element and no Lagrange condition: known, 0 unless a Dirichlet value sets them, loads dropped."""
        connected = self.assembly.Connected_rows(self.Get_contributions())
        lagrange = self.Compose(
            lambda field: BoundaryCondition.Get_dofs(field, self.simu.Bc_Lagrange),  # type: ignore [arg-type]
            isDofs=True,
        )
        connected[np.asarray(lagrange, dtype=int)] = True
        return np.flatnonzero(~connected)

    def __Warn_orphan_conditions(self) -> None:
        """Warns once per solve when a load or a Dirichlet condition touches an orphan dof. Collective on `mesh.comm`."""
        if self.orphanWarned:
            return
        self.orphanWarned = True
        simu = self.simu
        orphans = self.Orphan_dofs()
        touched = 0
        if orphans.size > 0:
            conditions = [
                self.Compose(Bc_dofs, isDofs=True)
                for Bc_dofs in (simu.Bc_dofs_Neumann, simu.Bc_dofs_Dirichlet)
            ]
            orphans = np.intersect1d(orphans, simu.Get_dofs(self.problemType))
            touched = np.isin(orphans, np.concatenate(conditions)).sum()
        touched = Reduce_sum(int(touched), simu.mesh.comm)
        if touched > 0:
            Terminal.MyPrintError(
                f"WARNING: {touched} dofs in no element carry a load or a Dirichlet condition: they stay at 0 (or the Dirichlet value), their loads dropped."
            )

    # ----------------------------------------------
    # Pinned dofs
    # ----------------------------------------------

    def Pinned_dofs(self, free: _types.BoolArray) -> _types.IntArray:
        """Dofs pinning the fields left free up to a constant, kept until `free` changes. Collective on `mesh.comm`."""
        stale = self.__pinnedFor is None or not np.array_equal(free, self.__pinnedFor)
        if Reduce_max(stale, self.simu.mesh.comm):
            self.__pinnedDofs, self.__pinnedWeights = self.__Find_pinned(free)
            self.__pinnedFor = free.copy()
        return self.__pinnedDofs

    def __Find_pinned(
        self, free: _types.BoolArray
    ) -> tuple[_types.IntArray, dict[ProblemType, _types.FloatArray]]:
        """DOF at global node 0 of each scalar field without a diagonal block whose constant the other fields' free equations are blind to, ``‖A_rc[free] · 1‖ ≤ ε ‖A_rc[free]‖``, and the field's zero-mean weights."""
        simu = self.simu
        committed = simu._Committed()
        terms = {field: simu.Get_terms(committed, field) for field in self.fields}
        diagonal = {
            rows
            for slot, rows, cols in _Blocks(terms)
            if slot in "KCM" and rows == cols
        }
        A = self.Get_A()
        owned = np.zeros(A.shape[0], dtype=bool)
        owned[simu.Get_dofs(self.problemType)] = True
        pinned: list[int] = []
        weights = {}
        comm = simu.mesh.comm
        for field, start in zip(self.fields, self.offsets):
            if simu.Get_dof_n(field) != 1 or field in diagonal:
                continue
            isField = np.zeros(A.shape[0], dtype=bool)
            isField[start : start + _Field_size(simu, field)] = True
            # rows complete only where owned under MPI
            A_rc = A[:, isField][owned & free & ~isField]
            constant = Reduce_sum(np.sum((A_rc @ np.ones(A_rc.shape[1])) ** 2), comm)
            scale = Reduce_sum(A_rc.multiply(A_rc).sum(), comm)
            if constant <= (1e-10) ** 2 * scale:
                space = simu._Get_FunctionSpace(field)
                node = space.Get_dof_nodes(np.flatnonzero(space.globalNodes == 0))
                pinned.extend(start + space.Get_dofs(node, 1))
                weights[field] = self.__Get_weights(field)
        return np.asarray(pinned, dtype=np.int64), weights

    def __Get_weights(self, field: ProblemType) -> _types.FloatArray:
        """``∫ N_i dΩ`` of each node of a scalar field."""
        simu = self.simu
        space = simu._Get_FunctionSpace(field)
        weights = np.zeros(space.Nn)
        dim = simu.mesh.dim
        for groupElem, spaceGroup in zip(
            simu.mesh.Get_list_groupElem(dim), space.Get_list_groupElem(dim)
        ):
            V_e = Operators.Linear.V(spaceGroup)
            np.add.at(weights, space.Get_assembly_e(groupElem, 1), V_e)
        return weights

    def Shift_pinned(self, x: _types.FloatArray) -> None:
        """Shifts each pinned field to zero mean, ``∫ field dΩ = 0`` with weights ``∫ N_i dΩ``, in place in `x`."""
        simu = self.simu
        comm = simu.mesh.comm
        offsets = self.offsets
        for field, weights in self.__pinnedWeights.items():
            start = offsets[self.fields.index(field)]
            values = x[start : start + weights.size]
            owned = simu.Get_dofs(field)
            mean = Reduce_sum(weights[owned] @ values[owned], comm) / Reduce_sum(
                weights[owned].sum(), comm
            )
            values -= mean
