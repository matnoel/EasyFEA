# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Fields of a simulation solved as one system: their system, assembly and contributions."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import sparse

from ..FEM import BoundaryCondition, Operators
from ..FEM._function_space import _FunctionSpace
from ..Solvers import Constraints, LagrangeConstraint, System
from ..Solvers._backends import Backend
from ..Utilities import Terminal, Tic, _types
from ..Utilities._mpi import Owned, Reduce_max, Reduce_sum
from ._assembly import _Assembly, _Blocks, _Size
from ._problem_type import ProblemType, TrialSolution
from ._terms import _Contribution
from .TimeSchemes import Elliptic

if TYPE_CHECKING:
    from ._simu import _Simu


def _Field_size(simu: "_Simu", problemType: ProblemType) -> int:
    """Number of dofs of one field."""
    return simu._Get_FunctionSpace(problemType).Nn * simu.Get_dof_n(problemType)


def _Slices(
    simu: "_Simu", problemTypes: tuple[ProblemType, ...]
) -> dict[ProblemType, slice]:
    """Where each problem type's dofs sit in `problemTypes` end to end."""
    slices = {}
    start = 0
    for problemType in problemTypes:
        stop = start + _Field_size(simu, problemType)
        slices[problemType] = slice(start, stop)
        start = stop
    return slices


def _Dof_owners(
    simu: "_Simu", problemTypes: tuple[ProblemType, ...]
) -> tuple[_types.IntArray | None, int]:
    """Owner rank of each dof of `problemTypes` end to end, None when never partitioned, and this rank."""
    spaces = [simu._Get_FunctionSpace(problemType) for problemType in problemTypes]
    rank = spaces[0].rank
    if any(space._dofOwners is None for space in spaces):
        return None, rank
    owners = [
        np.repeat(space._dofOwners, simu.Get_dof_n(problemType))
        for problemType, space in zip(problemTypes, spaces)
    ]
    return np.concatenate(owners), rank


class _Problem(ABC):
    """Fields of a simulation solved as one system ``[u | p | …]``: its system, assembly and contributions, kept until the mesh changes."""

    isNonLinear: bool
    """whether the terms read the iterate"""

    def __init__(self, simu: "_Simu", problemTypes: tuple[ProblemType, ...]) -> None:
        if len(set(problemTypes)) < len(problemTypes):
            raise ValueError(f"A field is given twice in {problemTypes}.")
        self.simu = simu
        self.problemTypes = problemTypes
        self.needUpdate = True
        """whether the contributions changed since last integrated"""
        self.__termRows: _types.BoolArray | None = None
        """rows a K, C or M term fills, kept until `Need_Update`"""
        self.__pinnedFor: _types.BoolArray | None = None
        """the free dofs the pinned ones were found for"""
        self.__pinned: tuple[_types.IntArray, dict[ProblemType, _types.FloatArray]] = (
            np.empty(0, dtype=np.int64),
            {},
        )

    @property
    def size(self) -> int:
        """Number of dofs of the fields."""
        return _Size(self.layout)

    @property
    def offsets(self) -> list[int]:
        """Where each field's dofs start."""
        return [offset for *_, offset in self.layout]

    @property
    def layout(self) -> tuple[tuple[ProblemType, _FunctionSpace, int, int], ...]:
        """`(field, space, dof_n, offset)` of each field, side by side."""
        layout = []
        offset = 0
        for problemType in self.problemTypes:
            space = self.simu._Get_FunctionSpace(problemType)
            dof_n = self.simu.Get_dof_n(problemType)
            layout.append((problemType, space, dof_n, offset))
            offset += space.Nn * dof_n
        return tuple(layout)

    @property
    def comm(self):
        """The mesh's communicator, None = no reduction."""
        return self.simu.mesh.comm

    @cached_property
    def system(self) -> System:
        """The linear system."""
        owners, rank = self.Get_dof_owners()
        globalIds = None
        if self.comm is not None:
            globalIds = self.__Get_dof_globalIds()
        return System(owners, rank, globalIds)

    @cached_property
    def assembly(self) -> _Assembly:
        """The assembly on `layout`."""
        return _Assembly(self.layout)

    def Free(self) -> None:
        """Destroys the solver kept by the system. Collective."""
        if "system" in self.__dict__:
            self.system.Free()

    def Check_monolithic(self) -> None:
        """Rejects what fields solved side by side cannot run yet; nothing for a field alone."""
        if len(self.problemTypes) == 1:
            return
        names = ", ".join(self.problemTypes)
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
        """`get(problemType)` of each problemType end to end, dof ids shifted to the problemType's offset when `isDofs`; `get(problemType)` itself for a problemType alone."""
        if len(self.problemTypes) == 1:
            return get(self.problemTypes[0])
        parts = [np.asarray(get(problemType)) for problemType in self.problemTypes]
        if isDofs:
            parts = [
                part.astype(int) + offset for part, offset in zip(parts, self.offsets)
            ]
        return np.concatenate(parts)

    def Get_dof_owners(self) -> tuple[_types.IntArray | None, int]:
        """Owner rank of each dof, None when never partitioned, and this rank."""
        return _Dof_owners(self.simu, self.problemTypes)

    def Owned_mask(self) -> _types.BoolArray:
        """Whether this rank owns each dof."""
        owners, rank = self.Get_dof_owners()
        isOwned = np.zeros(self.size, dtype=bool)
        isOwned[Owned(owners, rank)] = True
        return isOwned

    def __Get_dof_globalIds(self) -> _types.IntArray:
        """Global id of each dof, unique over the ranks."""
        globalIds = []
        for f, problemType in enumerate(self.problemTypes):
            space = self.simu._Get_FunctionSpace(problemType)
            dof_n = self.simu.Get_dof_n(problemType)
            nodes = space.globalNodes[space._dofNodes].astype(np.int64)
            dofs = nodes[:, None] * dof_n + np.arange(dof_n)
            globalIds.append((dofs * len(self.problemTypes) + f).ravel())
        return np.concatenate(globalIds)

    def Trial(self, x: _types.FloatArray) -> TrialSolution:
        """The trial solution of the iterate `x`: the fields views into `x`, the others committed."""
        trial = self.simu._Committed()
        for problemType, dofs in _Slices(self.simu, self.problemTypes).items():
            trial[problemType] = x[dofs]
        return trial

    # ----------------------------------------------
    # Contributions
    # ----------------------------------------------

    def _Integrate(self, trial: TrialSolution | None = None) -> list[_Contribution]:
        """The fields' contributions integrated at `trial` (default committed)."""
        if trial is None:
            trial = self.simu._Committed()
        self.needUpdate = False
        return self.simu._Get_list_contribution(self.problemTypes, trial)

    @abstractmethod
    def Get_contributions(self) -> list[_Contribution]:
        """The fields' contributions at the committed state."""

    def Need_Update(self) -> None:
        """Marks the contributions stale, forgets the connected rows."""
        self.needUpdate = True
        self.__termRows = None

    def Sum_A(self, list_contribution: list[_Contribution]) -> sparse.csr_matrix:
        """A for the time scheme, summed from `list_contribution`."""
        return self.assembly.Get_A(list_contribution, self.simu.timeScheme.coefs)

    def Sum_load(self, list_contribution: list[_Contribution]) -> _types.FloatArray:
        """External load, Neumann + the F of `list_contribution`."""
        simu = self.simu
        dofs = self.Compose(simu.Bc_dofs_Neumann, isDofs=True)
        dofsValues = self.Compose(simu.Bc_values_Neumann)

        load = np.zeros(self.size, dtype=float)
        np.add.at(load, np.asarray(dofs, dtype=int), dofsValues)
        load += self.assembly.Get_load(list_contribution)
        return load

    def Sum_b(
        self,
        list_contribution: list[_Contribution],
        load: _types.FloatArray,
        x: _types.FloatArray,
    ) -> _types.FloatArray:
        """b of `A dx = b` at the iterate x: `load` minus the internal forces of `list_contribution`."""
        simu = self.simu
        tic = Tic()

        u_n = simu._Get_u_n(self.problemTypes)
        v_n = simu._Get_v_n(self.problemTypes)
        a_n = simu._Get_a_n(self.problemTypes)
        fields = tuple(
            problemType if problemType is not None and problemType.any() else None
            for problemType in simu.timeScheme.Evaluate(x, u_n, v_n, a_n)
        )
        b = self.assembly.Get_b(list_contribution, load, fields)

        tic.Tac("Solver", f"b ({', '.join(self.problemTypes)})", simu._verbosity)

        return b

    def Get_A(self) -> sparse.csr_matrix:
        """A for the time scheme at the committed state."""
        return self.Sum_A(self.Get_contributions())

    def Get_load(self) -> _types.FloatArray:
        """External load at the committed state, Neumann + the terms' F."""
        return self.Sum_load(self.Get_contributions())

    def Get_product(
        self, fields: tuple[_types.FloatArray | None, ...]
    ) -> _types.FloatArray:
        """`K·u + C·v + M·a` of the committed contributions, `fields = (u, v, a)`, a None skipped; complete on owned rows only."""
        return self.assembly.Get_product(self.Get_contributions(), fields)

    # ----------------------------------------------
    # Solve
    # ----------------------------------------------

    def Constraints(self, x: _types.FloatArray | None = None) -> Constraints:
        """The known dofs at the Dirichlet values, orphan and pinned ones at x (default 0), every one at x when the scheme solves for a; the Lagrange constraints."""
        simu = self.simu
        knownDofs = np.flatnonzero(~self.Unknown_mask())
        target = np.zeros(self.size) if x is None else x.copy()
        if simu.timeScheme.unknown != "a":
            dofs = self.Compose(simu.Bc_dofs_Dirichlet, isDofs=True)
            dofs = np.asarray(dofs, dtype=int)
            target[dofs] = 0.0
            np.add.at(target, dofs, self.Compose(simu.Bc_values_Dirichlet))

        list_lagrangeConstraint = tuple(
            LagrangeConstraint(
                condition.dofs,
                condition.lagrangeCoefs.ravel(),
                float(condition.dofsValues[0]),
            )
            for condition in simu.Bc_Lagrange
            if condition.problemType in self.problemTypes
        )
        return Constraints(
            knownDofs, target[knownDofs], list_lagrangeConstraint, simu._resolution
        )

    def Free_mask(self) -> _types.BoolArray:
        """Whether each dof is free: not Dirichlet, not orphan."""
        dofs = self.Compose(self.simu.Bc_dofs_Dirichlet, isDofs=True)
        free = np.ones(self.size, dtype=bool)
        free[np.asarray(dofs, dtype=int)] = False
        free[self.Orphan_dofs()] = False
        return free

    def Unknown_mask(self) -> _types.BoolArray:
        """Whether each dof is solved for: free and not pinned. Collective on `mesh.comm`."""
        unknown = self.Free_mask()
        unknown[self.Pinned(unknown)[0]] = False
        return unknown

    def Orphan_dofs(self) -> _types.IntArray:
        """Dofs in no element and no Lagrange condition: known, 0 unless a Dirichlet value sets them, loads dropped."""
        return np.flatnonzero(~self.Connected_rows())

    def Connected_rows(self) -> _types.BoolArray:
        """Rows a K, C or M term fills on its groups, ghost elements included, and the Lagrange dofs; read from the committed terms, never integrated; a copy."""
        if self.__termRows is None:
            simu = self.simu
            committed = simu._Committed()
            rows = np.zeros(self.size, dtype=bool)
            for problemType in self.problemTypes:
                for term in simu.Get_terms(committed, problemType):
                    groups = term._Get_groups(simu.mesh)
                    for slot, block in zip(term.slots, term._Get_blocks(problemType)):
                        if slot in "KCM":
                            for group in groups:
                                rows[self.assembly.Dofs_e(block[0], group)] = True
            self.__termRows = rows
        lagrange = self.Compose(
            lambda problemType: BoundaryCondition.Get_dofs(
                problemType,
                self.simu.Bc_Lagrange,  # type: ignore [arg-type]
            ),
            isDofs=True,
        )
        connected = self.__termRows.copy()
        connected[np.asarray(lagrange, dtype=int)] = True
        return connected

    def Warn_orphan_conditions(self) -> None:
        """Warns when a load or a Dirichlet condition touches an orphan dof. Collective on `mesh.comm`."""
        simu = self.simu
        orphans = self.Orphan_dofs()
        touched = 0
        if orphans.size > 0:
            conditions = [
                self.Compose(Bc_dofs, isDofs=True)
                for Bc_dofs in (simu.Bc_dofs_Neumann, simu.Bc_dofs_Dirichlet)
            ]
            orphans = np.intersect1d(orphans, simu.Get_dofs(self.problemTypes))
            touched = np.isin(orphans, np.concatenate(conditions)).sum()
        touched = Reduce_sum(int(touched), simu.mesh.comm)
        if touched > 0:
            Terminal.MyPrintError(
                f"WARNING: {touched} dofs in no element carry a load or a Dirichlet condition: they stay at 0 (or the Dirichlet value), their loads dropped."
            )

    # ----------------------------------------------
    # Pinned dofs
    # ----------------------------------------------

    def Pinned(
        self, free: _types.BoolArray
    ) -> tuple[_types.IntArray, dict[ProblemType, _types.FloatArray]]:
        """Dofs pinning the fields left free up to a constant, and their zero-mean weights, kept until `free` changes. Collective on `mesh.comm`."""
        if len(self.problemTypes) == 1:
            return np.empty(0, dtype=np.int64), {}
        stale = self.__pinnedFor is None or not np.array_equal(free, self.__pinnedFor)
        if Reduce_max(stale, self.simu.mesh.comm):
            self.__pinned = self.__Find_pinned(free)
            self.__pinnedFor = free.copy()
        return self.__pinned

    def __Find_pinned(
        self, free: _types.BoolArray
    ) -> tuple[_types.IntArray, dict[ProblemType, _types.FloatArray]]:
        """DOF at global node 0 of each scalar field without a diagonal block whose constant the other fields' free equations are blind to, ``‖A_rc[free] · 1‖ ≤ ε ‖A_rc[free]‖``, and the field's zero-mean weights."""
        simu = self.simu
        committed = simu._Committed()
        terms = {
            problemType: simu.Get_terms(committed, problemType)
            for problemType in self.problemTypes
        }
        diagonal = {
            rows
            for slot, rows, cols in _Blocks(terms)
            if slot in "KCM" and rows == cols
        }
        A = self.Get_A()
        owned = self.Owned_mask()
        pinned: list[int] = []
        weights = {}
        comm = simu.mesh.comm
        for problemType, dofs in _Slices(simu, self.problemTypes).items():
            if simu.Get_dof_n(problemType) != 1 or problemType in diagonal:
                continue
            isField = np.zeros(A.shape[0], dtype=bool)
            isField[dofs] = True
            # rows complete only where owned under MPI
            A_rc = A[:, isField][owned & free & ~isField]
            constant = Reduce_sum(np.sum((A_rc @ np.ones(A_rc.shape[1])) ** 2), comm)
            scale = Reduce_sum(A_rc.multiply(A_rc).sum(), comm)
            if constant <= (1e-10) ** 2 * scale:
                space = simu._Get_FunctionSpace(problemType)
                node = space.Get_dof_nodes(np.flatnonzero(space.globalNodes == 0))
                pinned.extend(dofs.start + space.Get_dofs(node, 1))
                weights[problemType] = self.__Get_weights(problemType)
        return np.asarray(pinned, dtype=np.int64), weights

    def __Get_weights(self, problemType: ProblemType) -> _types.FloatArray:
        """``∫ N_i dΩ`` of each node of a scalar field."""
        simu = self.simu
        space = simu._Get_FunctionSpace(problemType)
        weights = np.zeros(space.Nn)
        dim = simu.mesh.dim
        for groupElem, spaceGroup in zip(
            simu.mesh.Get_list_groupElem(dim), space.Get_list_groupElem(dim)
        ):
            V_e = Operators.Linear.V(spaceGroup)
            np.add.at(weights, space.Get_assembly_e(groupElem, 1), V_e)
        return weights

    def Shift_pinned(self, x: _types.FloatArray) -> None:
        """Shifts each pinned field to zero mean, ``∫ field dΩ = 0`` with weights ``∫ N_i dΩ``, in place in `x`. Collective on `mesh.comm`."""
        simu = self.simu
        comm = simu.mesh.comm
        slices = _Slices(simu, self.problemTypes)
        _, dict_weights = self.Pinned(self.Free_mask())
        for problemType, weights in dict_weights.items():
            values = x[slices[problemType]]
            owned = simu.Get_dofs(problemType)
            mean = Reduce_sum(weights[owned] @ values[owned], comm) / Reduce_sum(
                weights[owned].sum(), comm
            )
            values -= mean


class _LinearProblem(_Problem):
    """A problem whose terms do not read the iterate: its contributions and A kept until dropped, A also on new time-scheme coefs."""

    isNonLinear = False

    def __init__(self, simu: "_Simu", problemTypes: tuple[ProblemType, ...]) -> None:
        super().__init__(simu, problemTypes)
        self.__list_contribution: list[_Contribution] | None = None
        self.__A: sparse.csr_matrix | None = None
        """the kept A, read-only"""
        self.__ACoefs: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def Need_Update(self) -> None:
        """Forgets the contributions and A."""
        super().Need_Update()
        self.__list_contribution = None
        self.__A = None

    def Get_contributions(self) -> list[_Contribution]:
        """The fields' contributions, kept."""
        if self.__list_contribution is None:
            self.__list_contribution = self._Integrate()
        return self.__list_contribution

    def Get_A(self) -> sparse.csr_matrix:
        """A for the time scheme, kept: the same object while the contributions and coefs hold."""
        coefs = self.simu.timeScheme.coefs
        if self.__A is None or coefs != self.__ACoefs:
            self.__A = self.assembly.Get_A(self.Get_contributions(), coefs)
            self.__ACoefs = coefs
        return self.__A

    def Get_A_b(self) -> tuple[sparse.csr_matrix, _types.FloatArray]:
        """A and b of `A x = b`; b at x = 0, never kept."""
        list_contribution = self.Get_contributions()
        if any(contribution.slot == "R" for contribution in list_contribution):
            # R is integrated at u_n, not at x = 0
            raise ValueError(
                "A term declaring an internal force R needs a nonlinear strategy: call `Solver_Set_Nonlinear_Strategy`, or declare a force independent of u in the F slot."
            )
        load = self.Sum_load(list_contribution)
        b = self.Sum_b(list_contribution, load, np.zeros(self.size))
        return self.Get_A(), b

    def Solve(self, backend: Backend) -> _types.FloatArray:
        """x of `A x = b`, one solve."""
        simu = self.simu
        tic = Tic()
        A, b = self.Get_A_b()
        x, _ = self.system.Solve(
            A,
            b,
            np.zeros(self.size),
            self.Constraints(),
            backend,
            simu.mesh.comm,
        )
        tic.Tac(
            "Solver",
            f"Solve {', '.join(self.problemTypes)} ({backend})",
            simu._verbosity,
        )
        return x


class _NonLinearProblem(_Problem):
    """A problem whose terms read the iterate: its contributions integrated on each read, never kept."""

    isNonLinear = True

    def Get_contributions(self) -> list[_Contribution]:
        """The fields' contributions at the committed state, integrated, never kept."""
        return self._Integrate()

    def Get_A_b(
        self, x: _types.FloatArray | None = None
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]:
        """K_t and R = load − f_int at the iterate x (default committed), integrated, never kept."""
        trial = self.simu._Committed() if x is None else self.Trial(x)
        list_contribution = self._Integrate(trial)
        load = self.Sum_load(list_contribution)
        b = self.Sum_b(list_contribution, load, self.Compose(trial.__getitem__))
        return self.Sum_A(list_contribution), b
