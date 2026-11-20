# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
import copy
import inspect
import pickle
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, Iterator, NamedTuple
import numpy as np
from scipy import sparse
import textwrap
from functools import cache

from ..__about__ import __version__

# utilities
from ..Utilities import Folder, Terminal, Tic, _params, _types
from ..Utilities._cache import (
    cache_computed_values,
    clear_cached_computed_values,
)
from ..Utilities._observers import Observable, _IObserver
from ..Utilities._mpi import (
    MPI_COMM,
    MPI_SIZE,
    Bcast,
    MPI_RANK,
    Gather,
    Place_rows,
    Reduce_max,
    Reduce_sum,
    Owned,
    Scatter_forward,
)

# fem
from ..FEM import (
    Mesh,
    Load_Mesh,
    _GroupElem,
    FeArray,
    BoundaryCondition,
    LagrangeCondition,
    MatrixType,
    Operators,
)
from ..FEM._function_space import _FunctionSpace
from ..FEM._mesh import _MeshPart

# materials
from ..Models import _IModel

# simu
from ._problem_type import ProblemType
from . import _iterations
from ._terms import Term, Fold_terms, _Contribution
from ._linear_system import _Lagrange, _LinearSystem
from .TimeSchemes import _TimeScheme, Elliptic, Parabolic, Hyperbolic
from .Solvers import SolverType, ResolType, CAN_USE_PYPARDISO, _Get_PETSc, _Solver


@cache
def _PETSc_Has(package: str) -> bool:
    """Whether petsc4py is usable and its PETSc build ships `package`."""
    PETSc = _Get_PETSc()
    return PETSc is not None and PETSc.Sys.hasExternalPackage(package)


# ----------------------------------------------
# _Simu
# ----------------------------------------------
class _Simu(_IObserver, _params.Updatable, ABC):
    r"""
    The following classes inherit from the parent class _Simu:
        - Elastic
        - HyperElastic
        - IncompressibleHyperElastic
        - InElastic
        - PhaseField
        - Beam
        - Thermal
        - WeakForms

    To create new simulation classes, you can take inspiration from existing implementations.\n
    Make sure to follow this interface.\n
    The Thermal class is relatively simple and can serve as a good starting point.\n
    See `Simulations/_thermal.py` for more details.

    To use the interface/inheritance, 10 methods and the class attributes ``problemTypes`` and ``dofFields`` need to be defined.

    General:
    --------

        - problemTypes = (U,)

        The problems the simulation solves, the first being its default ``problemType``.

        - def Get_unknowns(self, problemType=None) -> list[str]:

        - def Get_dof_n(self, problemType=None) -> int:

        These functions provides access to the available unknowns and degrees of freedom (dofs).

    Solve:
    ------

        - def Get_x0(self, problemType=None):

        - def Get_terms(self, problemType=None) -> list[Term]:

        These functions describe the matrix system :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`, which `_Simu` then assembles.

    Iterations:
    -----------

        - dofFields = {U: ("displacement", "speed", "accel")}

        The names of each problem type's fields in a saved iteration, by role: u, then v and a when the simulation has them. `Save_Iter` stores the roles the algorithm integrates (elliptic u, parabolic u and v, hyperbolic u, v and a), `Set_Iter` restores them, a role not saved as zero. The names are the keys of `Get_results`.

        - def _Trial_internals(self) -> dict[_GroupElem, dict[str, FeArray]]:

        - _externalNames

        Optional, none by default: the internal variables per group the current solution reaches, and the names of the external variables `Set_external` accepts. `_Commit_Internal_and_External_states` makes the trial internal variables and the external variables set the committed ones of `_variables` at the end of a converged `Solve`; `Save_Iter` stores the committed ones, `Set_Iter` restores copies, then calls `_Iter_Restored`.

    Results:
    --------

        - def Results_Available(self) -> list[str]:

        - def _Result(self, result: str, nodeValues=True) -> float | _types.FloatArray:

        - def Results_Iter_Summary(self) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:

        - def Results_dict_Energy(self) -> dict[str, float]:

        - def Results_displacement_matrix(self) -> _types.FloatArray:

        - def Results_nodeFields_elementFields(self, details=False) -> tuple[list[str], list[str]]:

        These functions are used to process the results. `_Result` reads the current state only: the base `Result` checks the name and reads a past iteration inside `_Iter_Visit`.
    """

    __indexMesh: int
    __mesh: Mesh
    __list_results: list[_iterations.Entry]
    __variables: _iterations.Variables
    __wholeRows: dict[int, list[_iterations.Rows]] = {}
    """By mesh index, the rows of the whole mesh each rank owns, set once the simulation is assembled from them; how an iteration saved over the ranks reads whole."""
    _resolution: ResolType | None = None
    """overrides the resolution, r2 with Lagrange conditions else r1; for tests"""
    __trial_solution: tuple | None = None
    """`(problem, x)` the solve iterates on, None outside a solve"""

    # ----------------------------------------------
    # Abstract method
    # ----------------------------------------------

    problemTypes: tuple[ProblemType, ...]
    """the problems the simulation solves, set on the class so that they are known before any instance; the first is its default"""

    problemType: ProblemType
    """the simulation's default problem type, the first of ``problemTypes``"""

    dofFields: dict[ProblemType, tuple[str, ...]]
    """each problem type's dof fields as named in a saved iteration, by role: u, then v and a when the simulation has them"""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "problemTypes" in cls.__dict__:
            cls.problemType = cls.problemTypes[0]

    def _Get_monolithic_problems(self) -> list[tuple[ProblemType, ...]]:
        """Problems solved as one system each, in solve order: one per problem type by default, several fields side by side ``[u | p]`` when their equations couple."""
        return [(problemType,) for problemType in self.problemTypes]

    def _Get_monolithic_problem(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> ProblemType | tuple[ProblemType, ...]:
        """The monolithic problem holding `problemType` (default the simulation's), as the solver path takes it: the problem type itself when solved alone, else the tuple of its fields."""
        if problemType is None:
            problemType = self.problemType
        problems = self._Get_monolithic_problems()
        for problem in problems:
            if problemType == problem or problemType in problem:
                return problem[0] if len(problem) == 1 else problem
        raise ValueError(
            f"{problemType} is not a problem of this simulation {problems}."
        )

    @staticmethod
    def __Fields(
        problem: ProblemType | tuple[ProblemType, ...],
    ) -> tuple[ProblemType, ...]:
        """The fields of a problem, side by side."""
        return problem if isinstance(problem, tuple) else (problem,)

    def __Get_offsets(self, problem: tuple[ProblemType, ...]) -> list[int]:
        """Where each field's DOFs start in the problem's system ``[u | p | …]``."""
        sizes = [self.__Get_size(problemType) for problemType in problem]
        return [0, *np.cumsum(sizes[:-1]).tolist()]

    def __Compose(
        self,
        problem: ProblemType | tuple[ProblemType, ...],
        get: Callable[[ProblemType], np.ndarray],
        isDofs: bool = False,
    ) -> np.ndarray:
        """`get(problemType)` of each field end to end, DOF ids shifted to the field's offset when `isDofs`; `get(problem)` itself for a problem type solved alone."""
        if not isinstance(problem, tuple):
            return get(problem)
        parts = [np.asarray(get(problemType)) for problemType in problem]
        if isDofs:
            offsets = self.__Get_offsets(problem)
            parts = [part.astype(int) + offset for part, offset in zip(parts, offsets)]
        return np.concatenate(parts)

    @abstractmethod
    def Get_unknowns(self, problemType: ProblemType = None) -> list[str]:
        """Returns a list of unknowns available in the simulation."""

    @abstractmethod
    def Get_dof_n(self, problemType: ProblemType | None = None) -> int:
        """Returns the number of degrees of freedom per node."""

    def _Get_FunctionSpace(
        self, problemType: ProblemType | None = None
    ) -> _FunctionSpace:
        """Function space of the problem type's field, held by the mesh."""
        if problemType is None:
            problemType = self.problemType
        return self.mesh._Get_FunctionSpace(problemType)

    def __Get_size(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> int:
        """Returns the number of degrees of freedom of the problem type's field, or of a problem's fields (Lagrange multiplier DOFs excluded)."""
        if problemType is None:
            problemType = self.problemType
        return sum(
            self._Get_FunctionSpace(field).Nn * self.Get_dof_n(field)
            for field in self.__Fields(problemType)
        )

    def __Get_Ndof(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> int:
        """Returns the total number of degrees of freedom (including Lagrange multiplier DOFs)."""
        if problemType is None:
            problemType = self.problemType
        fields = self.__Fields(problemType)
        return self.__Get_size(problemType) + sum(
            [self._Bc_Lagrange_dim(field) for field in fields]
        )

    def Calc_Energy(
        self,
        slot: str,
        x: _types.FloatArray,
        problemType: ProblemType | tuple[ProblemType, ...] | None = None,
        dofs: _types.IntArray | None = None,
    ) -> float:
        """Energy `1/2 xᵀ X x`, X the folded `slot` (`"K"` strain, `"M"` kinetic), over the owned `dofs`; collective on `mesh.comm`."""
        if slot not in ("K", "C", "M"):
            raise ValueError(f"slot must be 'K', 'C' or 'M', not {slot!r}.")
        problem = self._Get_monolithic_problem(problemType)
        if dofs is None:
            dofs = self.Get_dofs(problem)
        fields = tuple(x if s == slot else None for s in "KCM")
        product = self.__Get_product(problem, fields)
        energy = 0.5 * x[dofs] @ product[dofs]
        return Reduce_sum(energy, self.mesh.comm)

    def __Get_product(
        self,
        problem: ProblemType | tuple[ProblemType, ...],
        fields: tuple[_types.FloatArray | None, ...],
    ) -> _types.FloatArray:
        """`K·u + C·v + M·a` of the problem's kept fold, `fields = (u, v, a)`, a None skipped; complete on owned rows only."""
        system = self.__Get_folded(problem)
        layout, Ndof = self.__Get_layout(self.__Fields(problem))
        return system.Get_product(fields, layout, Ndof)[: self.__Get_size(problem)]

    def __Get_reaction(
        self, problem: ProblemType | tuple[ProblemType, ...], slots: str | None
    ) -> _types.FloatArray:
        """Internal force of the slots (default the scheme's), complete on owned rows only."""
        if self.isNonLinear:
            raise NotImplementedError
        if slots is None:
            roles = self.timeScheme.roles
            slots = "K" + "C" * ("v" in roles) + "M" * ("a" in roles)
        fields = (
            self._Get_u_n(problem) if "K" in slots else None,
            self._Get_v_n(problem) if "C" in slots else None,
            self._Get_a_n(problem) if "M" in slots else None,
        )
        return self.__Get_product(problem, fields)

    def __Get_owned_mask(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> _types.BoolArray:
        """Whether this rank owns each dof of the problem."""
        owners, rank = self.__Get_dof_owners(problem)
        isOwned = np.zeros(self.__Get_size(problem), dtype=bool)
        isOwned[Owned(owners, rank)] = True
        return isOwned

    def Calc_Reaction(
        self,
        dofs: _types.IntArray | None = None,
        problemType: ProblemType | None = None,
        slots: str | None = None,
    ) -> np.ndarray:
        """Internal force `K u + C v + M a` at `dofs` over the scheme's slots or the given `slots` (e.g. `"K"`), `F` not subtracted; under MPI the rank's rows."""
        problem = self._Get_monolithic_problem(problemType)
        reaction = self.__Get_reaction(problem, slots)
        if dofs is None:
            dofs = np.arange(reaction.size)
        reaction[~self.__Get_owned_mask(problem)] = 0.0
        # a ghost row from its owner
        self._Get_linear_system(problem).Scatter_forward(reaction, self.mesh.comm)
        return reaction[dofs]

    def Calc_Resultant(
        self,
        dofs: _types.IntArray,
        problemType: ProblemType | None = None,
        slots: str | None = None,
    ) -> float:
        """Sum of :meth:`Calc_Reaction` over `dofs`, each dof once over the ranks; collective on `mesh.comm`."""
        problem = self._Get_monolithic_problem(problemType)
        reaction = self.__Get_reaction(problem, slots)
        isOwned = self.__Get_owned_mask(problem)
        dofs = np.asarray(dofs, dtype=int)
        resultant = float(np.sum(reaction[dofs[isOwned[dofs]]]))
        return Reduce_sum(resultant, self.mesh.comm)

    @abstractmethod
    def Get_x0(self, problemType: ProblemType | None = None) -> _types.FloatArray:
        """Returns the solution from the previous iteration."""
        size = self.__Get_size(problemType)
        return np.zeros(size)

    def _Solver_Get_x0(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """:py:meth:`Get_x0` of each field of the problem, end to end."""
        return self.__Compose(problemType, self.Get_x0)

    def _Solver_Get_lb_ub(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """:py:meth:`Get_lb_ub` of each field of the problem, end to end."""
        if not isinstance(problemType, tuple):
            return self.Get_lb_ub(problemType)
        lb = self.__Compose(problemType, lambda field: self.Get_lb_ub(field)[0])
        ub = self.__Compose(problemType, lambda field: self.Get_lb_ub(field)[1])
        return lb, ub

    @abstractmethod
    def Get_terms(self, problemType: ProblemType | None = None) -> list[Term]:
        r"""Returns the operator terms building this simulation's local matrix system.

        A simulation describes itself as a list rather than a group loop::

            def Get_terms(self, problemType=None):
                return [Term("K", Operators.Bilinear.GradUGradV, coef=self.thermalModel.k)]

        Subclasses compose with ``super().Get_terms(problemType) + [...]``, the only way to extend assembly. Return ``[]`` when another problem's term fills this equation's rows.
        """
        return []

    # Iterations

    def Save_Iter(self, iter: dict[str, Any] | None = None) -> None:
        """Saves the current state as a new iteration, in memory or in `folder`, with the entries of `iter` (never mutated) as scalars; an array among them raises."""
        extras = {} if iter is None else iter
        arrays = [k for k, v in extras.items() if isinstance(v, np.ndarray)]
        assert not arrays, (
            f"Save_Iter stores no user arrays ({arrays}): declare them as the simulation's dof fields, internal or external variables"
        )
        scalars = {**extras, "indexMesh": self.__indexMesh}
        snapshot = self.__variables.Snapshot(
            scalars, self.__Dofs(), self._externalNames
        )
        partition = self.__Partition(self.mesh)
        if partition is not None and self.folder != "":
            self.__Write_rank_mesh(partition.rank)
        _iterations.Append(self.__list_results, snapshot, self.folder, partition)
        self.__Niter += 1

    def __Dofs(self) -> dict[str, _types.FloatArray]:
        """The current dof fields by name, those of the time scheme's roles."""
        nRoles = len(self.timeScheme.roles)
        getters = (self._Get_u_n, self._Get_v_n, self._Get_a_n)
        return {
            name: get(problemType)
            for problemType, names in self.dofFields.items()
            for name, get in zip(names[:nRoles], getters)
        }

    def Get_results(self, iter: int = -1) -> dict:
        """Return the iteration dict at index ``iter``, under flat keys (dof field names, ``internal/<elemType>/<name>``, ``external/<name>``, scalars), loading from disk if pickled. Pure read — no mutation of simulation state (use :meth:`Set_Iter` for that); collective on a partitioned mesh."""
        return _iterations.Flat(self.__Read(iter))

    def _Get_results_scalars(self, iter: int = -1) -> dict[str, Any]:
        """The scalars of the saved iteration `iter`, from memory; on rank 0 only under MPI."""
        return _iterations.Get_scalars(self.__list_results, iter, None)

    def Set_Iter(self, iter: int = -1) -> dict:
        """Sets the simulation to the saved iteration `iter` (the last by default) and returns its entries, as `Get_results`; collective on a partitioned mesh."""
        iteration = self.__Read(iter)
        self.__Restore(iteration)
        return _iterations.Flat(iteration)

    def __Read(self, iter: int) -> _iterations.Iteration:
        """A copy of the saved iteration `iter`, the rank's rows of it on a partitioned mesh."""
        entries = self.__list_results
        scalars = _iterations.Get_scalars(entries, iter, self.mesh.comm)
        index = scalars["indexMesh"]
        mesh = self.__Get_mesh(index)
        partition = self.__Partition(mesh)
        pieces = None
        if partition is None and index in self.__wholeRows:
            pieces = _iterations.Pieces(self.__Layout(mesh), self.__wholeRows[index])
        return _iterations.Get(
            entries,
            iter,
            self.folder,
            scalars,
            partition,
            pieces,
        )

    def __Partition(self, mesh: Mesh) -> _iterations.Layout | None:
        """How the rank's arrays on `mesh` split into the rows it owns; None unless the ranks share `mesh`."""
        return self.__Layout(mesh) if mesh.isPartitioned else None

    def __Layout(self, mesh: Mesh) -> _iterations.Layout:
        """How the arrays of a saved iteration lie on `mesh`: the rows the rank owns, every row of a mesh held whole."""
        dofs = {}
        for problemType, names in self.dofFields.items():
            space = mesh._Get_FunctionSpace(problemType)
            ownedNodes = space._dofNodes[Owned(space._dofOwners, space.rank)]
            rows = space.Get_dofs(ownedNodes, 1)
            dof_n = self.Get_dof_n(problemType)
            dofRows = _iterations.DofRows(rows, dof_n, space, ownedNodes)
            dofs.update({name: dofRows for name in names})
        groups = mesh.Get_list_groupElem(mesh.dim)
        spaceGroups = mesh._Get_FunctionSpace(self.problemType).Get_list_groupElem(
            mesh.dim
        )
        nodeSpace = mesh._Get_FunctionSpace()
        return _iterations.Layout(
            groups[0].rank,
            mesh.comm,
            dofs,
            nodeSpace._dofNodes[Owned(nodeSpace._dofOwners, nodeSpace.rank)],
            mesh._Get_FunctionSpace(),
            {g.elemType.value: (g, m.elemType) for g, m in zip(spaceGroups, groups)},
            {g.elemType: g for g in groups},
        )

    def __Get_mesh(self, index: int) -> Mesh:
        """The mesh of index `index`, loaded from `folder` once saved, shared by the ranks as the current one."""
        if index == self.__indexMesh:
            return self.__mesh
        mesh = self.__listMesh[index]
        if isinstance(mesh, str):
            mesh = Load_Mesh(Folder.Join(self.folder, mesh))
            if self.mesh.isPartitioned:
                mesh = mesh._Shared()
        return mesh

    def __Write_rank_mesh(self, rank: int) -> None:
        """Writes the rank's current mesh once to ``folder/Rank{rank}/Meshes/``, where saved iterations find the rows of the other ranks."""
        index = self.__indexMesh
        if isinstance(self.__listMesh[index], str):
            return
        path = Folder.Join(f"Rank{rank}", "Meshes", f"mesh{index}.pickle")
        self.mesh._Dump(Folder.Join(self.folder, path, mkdir=True))
        self.__listMesh[index] = path

    @contextmanager
    def _Iter_Visit(self) -> Iterator[None]:
        """Binds every attribute back on exit, so that a `Set_Iter` inside only reads; exact as long as nothing is written in place."""
        live = dict(self.__dict__)
        try:
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(live)
            # a reader inside rebuilt the linear systems' contributions in place
            self.Need_Update()

    def _Iter_Restored(self) -> None:
        """Called by `Set_Iter` once a saved iteration is restored, to drop what the subclass derived from the state before; nothing by default."""

    def __Restore(self, iteration: _iterations.Iteration) -> None:
        """Sets the simulation to copies of the saved `iteration`, a dof field it lacks to zero."""
        indexMesh = iteration["scalars"]["indexMesh"]
        if indexMesh != self.__indexMesh:
            self.__Update_mesh(indexMesh)
            self.__indexMesh = indexMesh

        dofs = iteration["dofs"]
        for problemType, names in self.dofFields.items():
            size = self.__Get_size(problemType)
            u, v, a = (
                np.array(dofs[name], dtype=float) if name in dofs else np.zeros(size)
                for name in (*names, None, None)[:3]
            )
            self._Set_solutions(problemType, u, v, a)

        groups = {
            groupElem.elemType.value: groupElem
            for groupElem in self._Get_FunctionSpace().Get_list_groupElem(self.mesh.dim)
        }
        self.__variables = self.__variables.Restored(iteration, groups)
        self._Iter_Restored()

    # Results

    @abstractmethod
    def Results_Available(self) -> list[str]:
        """Returns a list of available results in the simulation."""

    def Result(
        self, result: str, nodeValues: bool = True, iter: int | None = None
    ) -> _types.FloatArray | float:
        """Returns the result. Use Results_Available() to know the available results; at ``iter``, read with the simulation put back where it was.

        Examples
        --------
        For elastic simulation.

        >>> simu.Results_Available()   # list valid option strings

        Von Mises stress ():

        >>> svm = simu.Result("Svm", nodeValues=False)  # shape (Ne,)

        Node-interpolated von Mises stress:

        >>> svm = simu.Result("Svm")  # shape (Nn,)

        Result at a specific past iteration:

        >>> u5 = simu.Result("displacement_norm", iter=5)
        """
        if iter is not None:
            with self._Iter_Visit():
                self.Set_Iter(iter)
                return self.Result(result, nodeValues)

        if not self._Results_Check_Available(result):
            return None  # type: ignore [return-value]

        return self._Result(result, nodeValues)

    @abstractmethod
    def _Result(
        self, result: str, nodeValues: bool = True
    ) -> _types.FloatArray | float:
        """The available `result` of the current state, at nodes or elements."""

    @abstractmethod
    def Results_Iter_Summary(
        self,
    ) -> tuple[list[int], list[tuple[str, _types.FloatArray]]]:
        """Returns the values to be displayed in Plot_Iter_Summary."""
        return [], []

    @abstractmethod
    def Results_dict_Energy(self) -> dict[str, float]:
        """Returns a dictionary containing the names and values of the calculated energies."""
        return {}

    @abstractmethod
    def Results_displacement_matrix(self) -> _types.FloatArray:
        """Returns displacements as a matrix [dx, dy, dz] (Nn,3)."""
        Nn = self.mesh.Nn
        return np.zeros((Nn, 3))

    @abstractmethod
    def Results_nodeFields_elementFields(
        self, details=False
    ) -> tuple[list[str], list[str]]:
        """Returns lists of nodesFields and elementsFields displayed in paraview."""
        return [], []

    # ----------------------------------------------
    # core functions
    # ----------------------------------------------

    def _Check_dofs(self, problemType: ProblemType, unknowns: list) -> None:
        """Checks whether the specified unknowns are available for the problem."""
        dofs = self.Get_unknowns(problemType)
        for d in unknowns:
            assert d in dofs, f"{d} is not in {dofs}"

    def __Check_problemTypes(self, problemType: ProblemType) -> None:
        """Checks whether this type of problem is available through the simulation."""
        assert problemType in self.problemTypes, (
            f"This type of problem is not available in this simulation ({self.problemTypes})"
        )

    def _Check_dim_mesh_material(self) -> None:
        """Checks that the material dim matches the mesh dim."""
        dim = self.__model.dim
        assert dim == self.__mesh.dim and dim == self.__mesh.inDim, (
            "Material and mesh must share the same dimensions and belong to the same space."
        )

    def __str__(self) -> str:
        """Returns a string representation of the simulation; "" off rank 0."""

        if MPI_RANK != 0:
            return ""

        text = Terminal.Section("Mesh", False)
        text += "\n" + str(self.mesh)  # type: ignore

        text += Terminal.Section("Model", False)
        text += "\n" + str(self.model)

        text += "\n\nsolver:"

        if self.solver == SolverType.petsc:
            problems = self._Get_monolithic_problems()
            for problem in problems:
                isMonolithic = len(problem) > 1
                kspType, pcType, solverType = self._Solver_Get_PETSc4Py_Options(
                    problem if isMonolithic else problem[0]
                )
                if isMonolithic or len(problems) > 1:
                    names = ", ".join(problem)
                    text += f"\n({names}) {kspType}, {pcType}, {solverType}"
                else:
                    text += f" {kspType}, {pcType}, {solverType}"
        else:
            text += str(self.solver)

        results = self.Results_Get_Iteration_Summary()
        if results and not self.isGathered:
            warning = "Warning: values of rank 0's partition, ghosts included. For the whole mesh: simu._Gather() or Save(..., gather=True)."
            results = warning + "\n" + textwrap.dedent(results)

        # a section left unspecified by the simulation is dropped
        for title, summary in (
            ("Boundary Conditions", self.Results_Get_Bc_Summary()),
            ("Results", results),
        ):
            if summary:
                text += Terminal.Section(title, False)
                text += "\n" + textwrap.dedent(summary)

        text += Terminal.Section("TicTac", False)

        if Tic.nTic() > 0:
            text += Tic.Resume(False)

        return text

    def __init__(
        self, mesh: Mesh, model: _IModel, folder: str = "", verbosity: bool = True
    ):
        """Creates a simulation.

        Parameters
        ----------
        mesh : Mesh
            The mesh used.
        model : _IModel
            The model used.
        folder: str, optional
            save folder, by default "".
        verbosity : bool, optional
            If True, the simulation can write in the terminal. Defaults to True.
        """

        if verbosity:
            Terminal.Section("Simulation")

        self.__model = model

        self.folder = folder

        self.__dim: int = model.dim
        """Simulation dimension."""

        self.__Niter = 0
        self.__list_results = []
        """Saved iterations, see :data:`_iterations.Entry`."""
        self.__variables = _iterations.Variables()

        # Fill in the first mesh
        self.__NindexMesh: int = -1
        """Current mesh index in self.__listMesh"""
        self.__listMesh: list[str | Mesh] = []
        self.__linearSystems: dict[
            ProblemType | tuple[ProblemType, ...], _LinearSystem
        ] = {}
        self.__dict_needUpdate: dict[ProblemType | tuple[ProblemType, ...], bool] = {}
        """whether each problem key's kept fold is stale, a missing key stale"""
        self.mesh = mesh

        self.rho = 1.0

        self._Check_dim_mesh_material()

        self._verbosity: bool = verbosity
        """The simulation can write in the terminal"""

        self.__timeScheme: _TimeScheme = Elliptic()

        self.__isNonLinear = False
        """System type during simulation."""
        # a simulation is by default linear

        self.__orphanWarned = False
        """The current solve already warned about conditions on orphan dofs."""

        # Solver used
        self.__dict_solver: dict[ProblemType | tuple[ProblemType, ...], SolverType] = {}
        """Solver of a problem key overriding `solver`; kept across mesh changes."""
        if MPI_SIZE > 1:
            if _Get_PETSc() is None:
                raise ModuleNotFoundError(
                    "You must install petsc4py in order to run a simulation in parallel."
                )
            self.solver = SolverType.petsc
        elif CAN_USE_PYPARDISO:
            self.solver = SolverType.pypardiso
        elif _Get_PETSc() is not None:
            self.solver = SolverType.petsc
        else:
            self.solver = SolverType.scipy

        # petsc4py options; unset problem types get their defaults on first read.
        self.__dict_solver_petsc4py_options: dict[
            ProblemType | tuple[ProblemType, ...],
            tuple[str, str, str],
        ] = {}

        # Initialize solutions and boundary conditions
        self.__Init_Sols_n()
        self.Bc_Init()

        # simulation will look for material and mesh modifications
        model._Add_observer(self)
        mesh._Add_observer(self)

    @property
    def model(self) -> _IModel:
        """model used"""
        return self.__model

    rho = _params.PositiveParameter()
    """mass density"""

    def __Get_moments(self) -> _types.FloatArray:
        """Mass, then its first moment along x, y, z. Collective on `mesh.comm`."""
        moments = self.mesh._Sum_on_ownedElements_e(
            [
                groupElem._Get_moments_e(self.rho)
                for groupElem in self.mesh.Get_list_groupElem()
            ]
        )
        return moments * self.model.thickness if self.dim == 2 else moments

    @property
    def mass(self) -> float:
        """Mass of the simulated body. Collective on `mesh.comm`."""
        if self.dim == 1:
            return None  # type: ignore [return-value]
        return float(self.__Get_moments()[0])

    @property
    def center(self) -> _types.FloatArray:
        """Center of mass / barycenter / inertia center. Collective on `mesh.comm`."""

        if self.dim == 1:
            return None  # type: ignore [return-value]

        moments = self.__Get_moments()
        return moments[1:] / moments[0]

    # ----------------------------------------------
    # Solutions
    # ----------------------------------------------

    @property
    def folder(self) -> str:
        return self.__folder

    @folder.setter
    def folder(self, value: str) -> None:
        assert isinstance(value, str)
        if value != "" and not Folder.Exists(value):
            Folder.os.makedirs(value, exist_ok=True)
        self.__folder = value

    @property
    def Niter(self) -> int:
        """Number of iterations"""
        return self.__Niter

    def __Init_Sols_n(self) -> None:
        """Initializes the solutions."""
        self.__dict_u_n = {}
        self.__dict_v_n = {}
        self.__dict_a_n = {}
        for problemType in self.problemTypes:
            size = self.__Get_size(problemType)
            vectInit = np.zeros(size, dtype=float)
            self.__dict_u_n[problemType] = vectInit
            self.__dict_v_n[problemType] = vectInit
            self.__dict_a_n[problemType] = vectInit

    def __Check_New_Sol_Values(
        self, problemType: ProblemType, values: _types.FloatArray
    ) -> None:
        """Checks that the solution has the right size."""
        self.__Check_problemTypes(problemType)
        size = self.__Get_size(problemType)
        assert values.shape[0] == size, f"Must be size {size}"

    def _Get_u_n(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """Returns the solution associated with the given problem, its fields end to end."""
        return self.__Compose(problemType, self.__dict_u_n.__getitem__).copy()

    def __Set_u_n(self, problemType: ProblemType, values: _types.FloatArray) -> None:
        """Sets the solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_u_n = {**self.__dict_u_n, problemType: values}

    def _Get_v_n(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """Returns the speed solution associated with the given problem, its fields end to end."""
        return self.__Compose(problemType, self.__dict_v_n.__getitem__).copy()

    def __Set_v_n(self, problemType: ProblemType, values: _types.FloatArray) -> None:
        """Sets the speed solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_v_n = {**self.__dict_v_n, problemType: values}

    def _Get_a_n(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """Returns the acceleration solution associated with the given problem, its fields end to end."""
        return self.__Compose(problemType, self.__dict_a_n.__getitem__).copy()

    def __Set_a_n(self, problemType: ProblemType, values: _types.FloatArray) -> None:
        """Sets the acceleration solution associated with the given problem."""
        self.__Check_New_Sol_Values(problemType, values)
        self.__dict_a_n = {**self.__dict_a_n, problemType: values}

    # This method is overloaded in PhaseFieldSimu
    def Get_lb_ub(
        self, problemType: ProblemType
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the lower bound and upper bound."""
        return np.array([]), np.array([])

    # Properties
    @property
    def timeScheme(self) -> _TimeScheme:
        """The time scheme solves use, set by the `Solver_Set_*_Algorithm` methods."""
        return self.__timeScheme

    @property
    def inDim(self) -> int:
        list_dof_n = [
            min(self.Get_dof_n(problemType), 3) for problemType in self.problemTypes
        ]
        inDim = np.max([self.mesh.inDim, *list_dof_n])
        return inDim

    @property
    def mesh(self) -> Mesh:
        """simulation's mesh."""
        return self.__mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):
        if isinstance(mesh, Mesh):
            # For all old meshes, delete the matrices
            listMesh: list[Mesh] = self.__listMesh
            [m._ResetMatrix() for m in listMesh if isinstance(m, Mesh)]  # type: ignore [func-returns-value]

            # set mesh
            self.__NindexMesh += 1
            self.__indexMesh = self.__NindexMesh
            self.__listMesh.append(mesh)
            self.__mesh = mesh

            # New connectivity invalidates the geometry-derived caches.
            # This cannot live in `Need_Update`, which fires on every Newton
            # iteration — exactly the case those caches exist to serve.
            clear_cached_computed_values(self)
            self.__Drop_linear_systems()
            mesh._Add_observer(self)

            # The mesh changes, so the matrices must be reconstructed
            self.Need_Update()
            # Initialize boundary conditions
            self.Bc_Init()
            # initialize the solutions
            self.__Init_Sols_n()

    @property
    def Nmesh(self) -> int:
        return len(self.__listMesh)

    @property
    def isGathered(self) -> bool:
        """Whether the simulation holds the complete global mesh."""
        return not self.mesh.isPartitioned

    def _Gather(self):
        """Gives rank 0 the whole mesh and the current state placed by global id: u, v, a of each problem type, the internal and nodal external variables, the heterogeneous parameters and the iterations kept in memory; rank 0 becomes a serial simulation, the other ranks keep their partition. Collective."""
        # rank 0 already whole: every rank leaves together
        if MPI_SIZE == 1 or Bcast(self.isGathered, MPI_COMM):
            return

        # rank 0 turns serial: its solvers on the world comm are freed while every rank is here
        self.__Free_solvers()

        comm = self.mesh.comm
        parts = Gather(self.__Rank_part(), comm)
        if parts is not None:
            self.__Assemble(parts)

    def _Assemble_ranks(self, simus: list["_Simu"]) -> None:
        """Turns this simulation whole from the partitions of every rank, `simus` loaded from a run saved over the ranks, itself rank 0's."""
        self.__Assemble([simu.__Rank_part() for simu in simus])

    def __Rank_part(self) -> "_RankPart":
        """What the rank owns of the simulation, from which `__Assemble` builds the whole one; collective on a partitioned mesh."""
        meshes = [self.__Get_mesh(i) for i in range(self.Nmesh)]
        layouts = [self.__Layout(mesh) for mesh in meshes]
        layout = layouts[self.__indexMesh]
        variables = self.__variables
        state: _iterations.Iteration = {
            "scalars": {},
            "dofs": self.__Dofs(),
            "nodal": {
                **{f"externals/{k}": v for k, v in variables.externals.items()},
                **{f"next/{k}": v for k, v in variables.nextExternals.items()},
            },
            "quadrature": {
                groupElem.elemType.value: fields
                for groupElem, fields in variables.internals.items()
            },
        }
        mesh = self.mesh
        sources = mesh.Get_list_groupElem()
        splits = np.cumsum([groupElem.Ne for groupElem in sources])[:-1]
        parameters: list[list[tuple[_types.IntArray, np.ndarray]] | None] = []
        for owner, name in self.__Parameters():
            value = vars(owner)[name]
            # placed only when one value per element
            if np.ndim(value) > 0 and len(value) == mesh.Ne:
                parameters.append(
                    [
                        (
                            g._globalElements[Owned(g._elementOwners, g.rank)],
                            part[Owned(g._elementOwners, g.rank)],
                        )
                        for g, part in zip(sources, np.split(value, splits))
                    ]
                )
            else:
                parameters.append(None)
        return _RankPart(
            [mesh._Owned_part() for mesh in meshes],
            [_iterations.Share_of(layout) for layout in layouts],
            _iterations.Owned_rows(state, layout),
            _iterations.Owned_memory(
                self.__list_results, mesh.comm, lambda index: layouts[index]
            ),
            parameters,
        )

    def __Assemble(self, rankParts: list["_RankPart"]) -> None:
        """Turns the simulation whole, serial, from what every rank owns of it (`__Rank_part`)."""
        verbosity = self.mesh.verbosity
        for i in range(self.Nmesh):
            mesh = Mesh._Assembled([piece.meshes[i] for piece in rankParts], verbosity)
            mesh._ResetMatrix()
            self.__listMesh[i] = mesh
        self.__mesh = self.__listMesh[self.__indexMesh]
        self.Bc_Init()
        # the geometry-derived caches describe the partition
        clear_cached_computed_values(self)
        self.__Drop_linear_systems()
        self.Need_Update()

        wholes = [self.__Layout(self.__Get_mesh(i)) for i in range(self.Nmesh)]
        self.__wholeRows = {
            i: [_iterations.Rows_in(piece.shares[i], whole) for piece in rankParts]
            for i, whole in enumerate(wholes)
        }

        index = self.__indexMesh
        parts = [
            (f"rank {r}", rows, piece.state)
            for r, (rows, piece) in enumerate(zip(self.__wholeRows[index], rankParts))
        ]
        state = _iterations.Placed(parts, wholes[index])
        nodal = state["nodal"]
        externals = {
            k.removeprefix("externals/"): v
            for k, v in nodal.items()
            if k.startswith("externals/")
        }
        self.__Restore({**state, "scalars": {"indexMesh": index}, "nodal": externals})
        nextExternals = {
            k.removeprefix("next/"): v
            for k, v in nodal.items()
            if k.startswith("next/")
        }
        self.__variables = replace(self.__variables, nextExternals=nextExternals)

        _iterations.Place_memory(
            self.__list_results,
            [piece.memory for piece in rankParts],
            lambda i: _iterations.Pieces(wholes[i], self.__wholeRows[i]),
        )

        targets = self.mesh.Get_list_groupElem()
        for p, (owner, name) in enumerate(self.__Parameters()):
            values = [piece.parameters[p] for piece in rankParts]
            if any(v is None for v in values):
                continue
            value_e = [
                Place_rows([v[g] for v in values if v is not None], target.Ne)
                for g, target in enumerate(targets)
            ]
            setattr(owner, name, np.concatenate(value_e))

    def __Parameters(self) -> list[tuple[object, str]]:
        """The parameters of the simulation and of every model it reaches through attributes, as (owner, name)."""
        owners: list[object] = [self]
        for owner in owners:
            for value in vars(owner).values():
                if isinstance(value, _IModel) and all(value is not o for o in owners):
                    owners.append(value)
        return [
            (owner, name)
            for owner in owners
            for name in vars(owner)
            if isinstance(
                inspect.getattr_static(type(owner), name, None), _params._Parameter
            )
        ]

    @property
    def dim(self) -> int:
        """simulation's dimension"""
        return self.__dim

    @property
    def thickness(self) -> float:
        """Out-of-plane thickness in 2D, 1 otherwise. Operators integrate over the element's own dimension, so every 2D contribution is scaled by this once, in :func:`~EasyFEA.Simulations._terms.Fold_terms`."""
        return self.model.thickness if self.dim == 2 else 1.0

    def __Update_mesh(self, index: int) -> None:
        """Updates the mesh for the specified iteration.

        Parameters
        ----------
        index : int
            The mesh index in self.__listMesh.
        """

        mesh = self.__Get_mesh(index)

        self.__mesh = mesh

        # switching to another mesh in the history changes the connectivity
        clear_cached_computed_values(self)
        self.__Drop_linear_systems()

        self.Need_Update()  # need to reconstruct matrices

    def _Update(self, observable: Observable, event: str) -> None:
        if isinstance(observable, _IModel):
            self.Need_Update()
        elif isinstance(observable, Mesh):
            self._Check_dim_mesh_material()
            clear_cached_computed_values(self)
            self.__Drop_linear_systems()
            self.Need_Update()
        else:
            Terminal.MyPrintError("Notification not yet implemented")

    def _Get_linear_system(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> _LinearSystem:
        """The linear system of a problem key, kept until the mesh changes; a monolithic one refuses a non-elliptic scheme and Lagrange conditions when created."""
        system = self.__linearSystems.get(problem)
        if system is None:
            if isinstance(problem, tuple):
                self.__Check_monolithic(problem)
            owners, rank = self.__Get_dof_owners(problem)
            globalIds = None
            if self.mesh.comm is not None:
                globalIds = self.__Get_dof_globalIds(problem)
            system = _LinearSystem(owners, rank, globalIds)
            self.__linearSystems[problem] = system
        return system

    def __Get_dof_globalIds(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> _types.IntArray:
        """Global id of each dof of the problem's vector (its fields end to end), unique over the ranks."""
        fields = self.__Fields(problem)
        globalIds = []
        for f, field in enumerate(fields):
            space = self._Get_FunctionSpace(field)
            dof_n = self.Get_dof_n(field)
            nodes = space.globalNodes[space._dofNodes].astype(np.int64)
            dofs = nodes[:, None] * dof_n + np.arange(dof_n)
            globalIds.append((dofs * len(fields) + f).ravel())
        return np.concatenate(globalIds)

    def __Free_solvers(
        self, problems: list[ProblemType | tuple[ProblemType, ...]] | None = None
    ) -> None:
        """Destroys the solvers kept by the systems of `problems`, every one by default. Collective."""
        for problem, system in self.__linearSystems.items():
            if problems is None or problem in problems:
                system.Free_solver()

    def __Drop_linear_systems(self) -> None:
        """Forgets every linear system: the mesh they were built on changed. Collective."""
        self.__Free_solvers()
        self.__linearSystems = {}

    def Need_Update(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> None:
        """Marks stale the contributions of every problem sharing a field with `problemType`, of every problem by default: its next reader rebuilds them."""
        if problemType is None:
            self.__dict_needUpdate = {}
            return
        fields = set(self.__Fields(problemType))
        self.__dict_needUpdate = {
            key: needUpdate or not fields.isdisjoint(self.__Fields(key))
            for key, needUpdate in self.__dict_needUpdate.items()
        }

    @property
    def needUpdate(self) -> bool:
        """Whether the fold of a problem of the simulation is stale."""
        return any(
            self.__dict_needUpdate.get(self._Get_monolithic_problem(problem[0]), True)
            for problem in self._Get_monolithic_problems()
        )

    # ----------------------------------------------
    # Solver
    # ----------------------------------------------

    @cache_computed_values
    def __Get_csr_map(
        self,
        layout: tuple[tuple[ProblemType, _FunctionSpace, int, int], ...],
        isMatrix: bool,
        Ndof: int,
        contributing: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Reduction map from element entries to global CSR slots for one assembly key.

        Returns ``(inv, indices, indptr, nnz)``: ``inv[k]`` is the CSR-data slot that element entry ``k``
        feeds, and ``(indices, indptr, nnz)`` is scipy's canonical pattern. Depends only on connectivity, so
        ``@cache_computed_values`` (keyed on ``layout, isMatrix, Ndof, contributing``) reuses it across every
        assembly of a key and rebuilds it when the spaces / groups / Ndof / dof_n change or the mesh is reset. ``inv``
        is the single largest cached array; int32 halves it and is safe since slots are bounded by nnz < 2^31.
        """
        shape = (Ndof, Ndof) if isMatrix else (Ndof, 1)
        byField = {
            problemType: (functionSpace, dof_n, offset)
            for problemType, functionSpace, dof_n, offset in layout
        }

        list_rows, list_cols = [], []
        for block, groups in contributing:
            r, c = block if isMatrix else (block, block)
            space_r, dof_n_r, offset_r = byField[r]
            space_c, dof_n_c, offset_c = byField[c]
            for groupElem in groups:
                rows_e = space_r.Get_assembly_e(groupElem, dof_n_r) + offset_r
                if isMatrix:
                    cols_e = space_c.Get_assembly_e(groupElem, dof_n_c) + offset_c
                    list_rows.append(
                        _FunctionSpace._Get_rows_e(rows_e, cols_e.shape[1]).ravel()
                    )
                    list_cols.append(
                        _FunctionSpace._Get_columns_e(cols_e, rows_e.shape[1]).ravel()
                    )
                else:
                    list_rows.append(rows_e.ravel())
                    list_cols.append(np.zeros(rows_e.size, dtype=rows_e.dtype))
        rows = np.concatenate(list_rows)
        cols = np.concatenate(list_cols)

        # scipy owns the canonical sparsity pattern; build it once from a structure-only matrix (the values
        # are irrelevant — only the nonzero positions matter).
        matrix = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=shape)
        matrix.sort_indices()

        # `canon` is the sorted linear (row-major) index of each stored coefficient; `searchsorted` maps each
        # element entry to the slot whose `(row, col)` it matches. Building the map from scipy's own pattern
        # (rather than a parallel hand-built one) is correct by construction, independent of any sort order.
        ncol = Ndof if isMatrix else 1
        canon = (
            np.repeat(np.arange(shape[0]), np.diff(matrix.indptr)) * ncol
            + matrix.indices
        )
        inv = np.searchsorted(canon, rows.astype(np.int64) * ncol + cols).astype(
            np.int32
        )
        return inv, matrix.indices, matrix.indptr, matrix.nnz

    def __Get_layout(
        self, problemTypes: tuple[ProblemType, ...]
    ) -> tuple[tuple[tuple[ProblemType, _FunctionSpace, int, int], ...], int]:
        """`(problemType, space, dof_n, offset)` of each field, side by side in the given order, and Ndof, every Lagrange dof after them."""
        layout = []
        offset = 0
        for problemType in problemTypes:
            layout.append(
                (
                    problemType,
                    self._Get_FunctionSpace(problemType),
                    self.Get_dof_n(problemType),
                    offset,
                )
            )
            offset += self.__Get_size(problemType)
        Ndof = offset + sum(
            self._Bc_Lagrange_dim(problemType) for problemType in problemTypes
        )
        return tuple(layout), Ndof

    @staticmethod
    def __Blocks(
        terms: dict[ProblemType, list[Term]],
    ) -> Iterator[tuple[str, ProblemType, ProblemType]]:
        """`(slot, rows, cols)` of each array the terms declare, a vector's `cols` its rows."""
        for problemType, fieldTerms in terms.items():
            for term in fieldTerms:
                for slot, (rows, cols) in zip(
                    term.slots, term._Get_blocks(problemType)
                ):
                    yield slot, rows, cols

    def __Get_list_contribution(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> list[_Contribution]:
        """The contributions of every field's terms; raises on a block coupling to a field outside `problem`, or on a field no term fills."""
        problemTypes = self.__Fields(problem)
        fields = ", ".join(problemTypes)

        tic = Tic()

        terms = {
            problemType: self.Get_terms(problemType) for problemType in problemTypes
        }
        filled = set()
        for slot, rows, cols in self.__Blocks(terms):
            if slot in "KCM" and cols not in problemTypes:
                raise ValueError(
                    f"{rows}'s equation couples to {cols}: assemble ({fields}, {cols})."
                )
            if rows not in problemTypes:
                raise ValueError(
                    f"A term fills {rows}'s equation: assemble ({fields}, {rows})."
                )
            filled.add(rows)
        for problemType in problemTypes:
            if problemType not in filled:
                raise ValueError(
                    f"No term fills {problemType}'s equation: assemble {self._Get_monolithic_problem(problemType)}."
                )

        fold: list[_Contribution] = []
        for problemType in problemTypes:
            fold += Fold_terms(self, terms[problemType], problemType)

        tic.Tac("Matrix", f"Fold ({fields}).", self._verbosity)

        return fold

    def Assembly(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> tuple[
        sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, _types.FloatArray
    ]:
        r"""K, C, M (csr, one pattern) and F (1-D, the terms' load) of :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`, of one or several fields ``[u | p | Lagrange]``; built on each call, never read by a solve."""
        if problemType is None:
            problemType = self._Get_monolithic_problem()
        problemTypes = self.__Fields(problemType)
        if len(set(problemTypes)) < len(problemTypes):
            raise ValueError(f"A field is given twice in {problemTypes}.")
        problem = problemTypes[0] if len(problemTypes) == 1 else problemTypes

        system = self.__Get_folded(problem)
        layout, Ndof = self.__Get_layout(problemTypes)

        def Get_map(contributing: tuple):
            return self.__Get_csr_map(layout, True, Ndof, contributing)

        K, C, M = (system.Get_operator(slot, layout, Ndof, Get_map) for slot in "KCM")
        return K, C, M, system.Get_load(layout, Ndof)

    @property
    def solver(self) -> str:
        """Solver used to solve the simulation."""
        return self.__solver

    @solver.setter
    def solver(self, value: str):
        solvers = list(SolverType)
        if value not in solvers:
            Terminal.MyPrintError(
                f"The solver {value} cannot be used. The solver must be in {solvers}"
            )
        else:
            self.__Check_distributed(value)
            self.__solver = value
            self.__Free_solvers(
                [
                    problem
                    for problem in self.__linearSystems
                    if self._Solver_Get(problem) != SolverType.petsc
                ]
            )

    def _Solver_Set(
        self,
        solverType: SolverType | None,
        problemType: ProblemType | tuple[ProblemType, ...],
    ) -> None:
        """Sets the solver of one problem key, `None` = back to `solver`; frees that key's kept PETSc solver on leaving petsc."""
        if solverType is None:
            self.__dict_solver.pop(problemType, None)
        else:
            self.__Check_distributed(solverType)
            self.__dict_solver[problemType] = SolverType(solverType)
        if self._Solver_Get(problemType) != SolverType.petsc:
            self.__Free_solvers([problemType])

    @staticmethod
    def __Check_distributed(solverType: str) -> None:
        if MPI_SIZE > 1 and solverType != SolverType.petsc:
            raise ValueError(
                f"Only petsc solves a system distributed over the ranks, not {solverType}."
            )

    def _Solver_Get(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> SolverType:
        """Solver of one problem key: its own, else `solver`."""
        return SolverType(self.__dict_solver.get(problemType, self.solver))

    def __getstate__(self) -> dict:
        # PETSc objects do not pickle: a copy builds its own, in its own systems
        systems = {
            key: copy.copy(system) for key, system in self.__linearSystems.items()
        }
        return {
            **self.__dict__,
            "_Simu__linearSystems": systems,
            "_Simu__dict_needUpdate": {},
        }

    def _Solver_Set_TimeScheme(self, timeScheme: _TimeScheme) -> None:
        """Sets the time scheme of the next solves, any `_TimeScheme`."""
        if not isinstance(timeScheme, _TimeScheme):
            raise TypeError(f"{timeScheme!r} is not a _TimeScheme.")
        assert not (timeScheme.unknown != "u" and self.isNonLinear), (
            f"{type(timeScheme).__name__} solves for {timeScheme.unknown}: only for linear simulations."
        )
        self.__timeScheme = timeScheme

    def Solver_Set_Elliptic_Algorithm(self) -> None:
        r"""Sets the algorithm's resolution properties for an elliptic problem.

        Used to solve :math:`\Krm \, \mathrm{u} = \Frm`.

        Examples
        --------
        >>> simu.Solver_Set_Elliptic_Algorithm()
        >>> u = simu.Solve()
        """
        self._Solver_Set_TimeScheme(Elliptic())

    def Solver_Set_Parabolic_Algorithm(self, dt: float, alpha=1 / 2) -> None:
        r"""Sets the algorithm's resolution properties for a parabolic problem.

        Used to solve :math:`\Krm \, u^{n+1} + \Crm \, \vrm^{n+1} = F^{n+1}` with:

        :math:`\mathrm{u}^{n+1} = \mathrm{u}^n + \dt \, \vrm^{n+\alpha}`

        Parameters
        ----------
        dt : float
            The time increment.

        alpha : float, optional
            The alpha criterion in ]0, 1], by default 1/2\n
            - 1 -> Backward Euler
            - 1/2 -> midpoint

        Examples
        --------
        Transient heat equation with Crank-Nicolson (α=0.5):

        >>> simu.Solver_Set_Parabolic_Algorithm(dt=0.01, alpha=0.5)
        >>> simu.add_dirichlet(nodes, [100.0], ["t"])
        >>> for t in times:
        ...     simu.Solve()
        ...     simu.Save_Iter()
        """
        self._Solver_Set_TimeScheme(Parabolic(dt, alpha))

    def Solver_Set_Hyperbolic_Algorithm(
        self,
        dt: float,
        scheme: type[Hyperbolic._Hyperbolic] = Hyperbolic.Newmark,
        **params: float,
    ) -> None:
        r"""Sets the algorithm's resolution properties for a Hyperbolic problem.

        Used to solve :math:`\Krm \, \mathrm{u} + \Crm \, \vrm + \Mrm \, \arm = \Frm`.

        Parameters
        ----------
        dt : float
            The time increment.
        scheme : type[Hyperbolic._Hyperbolic], optional
            A scheme of `TimeSchemes.Hyperbolic`, by default `Newmark`.
        **params
            The scheme's parameters, e.g. `beta`, `gamma` of `Newmark` or `alpha` of `HHT`.

        Examples
        --------
        Newmark average acceleration (unconditionally stable, default):

        >>> simu.Solver_Set_Hyperbolic_Algorithm(dt=0.001)  # beta=0.25, gamma=0.5
        >>> simu.Solve()

        HHT-α (Hilber-Hughes-Taylor, numerical damping with α=1/6, β and γ derived from α):

        >>> from EasyFEA import TimeSchemes
        >>> simu.Solver_Set_Hyperbolic_Algorithm(0.001, TimeSchemes.Hyperbolic.HHT, alpha=1 / 6)

        Midpoint rule (second-order, unconditionally stable):

        >>> simu.Solver_Set_Hyperbolic_Algorithm(0.001, TimeSchemes.Hyperbolic.Midpoint)
        """
        if not (
            isinstance(scheme, type) and issubclass(scheme, Hyperbolic._Hyperbolic)
        ):
            raise TypeError(
                f"scheme must be a TimeSchemes.Hyperbolic one, not {scheme}."
            )
        self._Solver_Set_TimeScheme(scheme(dt, **params))

    def Get_u_v_a(
        self, problemType: ProblemType | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        r"""Returns the current iterate :math:`(\mathrm{u}_t, \vrm_t, \arm_t)`, evaluated at the time scheme's own evaluation point: from the trial solution when nonlinear, from the state at n otherwise; None for a role the scheme does not integrate."""

        if problemType is None:
            problemType = self.problemType

        u_n = self._Get_u_n(problemType)
        v_n = self._Get_v_n(problemType)
        a_n = self._Get_a_n(problemType)
        scheme = self.timeScheme

        if self.isNonLinear:
            x = self._Solver_Get_trial_solution(problemType)
        else:
            x = {"u": u_n, "v": v_n, "a": a_n}[scheme.unknown]

        return scheme.Evaluate(x, u_n, v_n, a_n)

    def _Solver_Set_Newton_Raphson_Algorithm(
        self,
        absTol: float = 1.0e-6,
        relTol: float = 1.0e-10,
        incTol: float = 1.0e-11,
        maxIter: int = 20,
    ) -> None:
        """Sets the algorithm's resolution properties for an non linear problem.\n

        Used to solve A(u) Δu = - R(u).

        Parameters
        ----------
        absTol : float, optional
            absolute tolerance, by default 1e-6
        relTol : float, optional
            relative tolerance, by default 1e-10
        incTol : float, optional
            incremental tolerance, by default 1e-11
        maxIter : int, optional
            maximum iteration, by default 20
        """

        self.__isNonLinear = True

        assert 0 < absTol < 1, "absTol must be between 0 and 1."
        assert 0 < relTol < 1, "relTol must be between 0 and 1."
        assert 0 < incTol < 1, "incTol must be between 0 and 1."
        assert maxIter > 1, "maxIter must be > 1."

        self.__newtonRaphsonParams = (absTol, relTol, incTol, maxIter)

        self._Iter_Record(newtonIter=None, timeIter=None, list_norm_r=None)

    def __Solver_Get_Newton_Raphson_Params(self) -> tuple[float, float, float, int]:
        """Returns (absTol, relTol, incTol, maxIter) Newton Raphson properties."""
        assert self.__isNonLinear, "the current simulation is not non linear."

        return self.__newtonRaphsonParams

    @property
    def isNonLinear(self):
        """Returns whether the simulation is non linear."""
        return self.__isNonLinear

    def Solve(self) -> _types.FloatArray:
        """Computes the solution field for the current boundary conditions, solving each monolithic problem in turn.

        Returns
        -------
        _types.FloatArray
            The solution of the simulation's problem type.
        """

        for problem in self._Get_monolithic_problems():
            self._Solver_Solve_problemType(problem)

        self._Commit_Internal_and_External_states()

        return self._Get_u_n(self.problemType)

    @property
    def _variables(self) -> _iterations.Variables:
        """The internal and external variables, and the record of the next saved iteration."""
        return self.__variables

    @property
    def _externalNames(self) -> tuple[str, ...]:
        """Names of the external variables the simulation reads; none by default."""
        return ()

    @property
    def external(self) -> dict[str, _types.FloatArray]:
        """The nodal external variables the next solves see, by name; set them with :meth:`Set_external`."""
        return {name: v.copy() for name, v in self.__variables.nextExternals.items()}

    def Set_external(self, **values: float | _types.FloatArray) -> None:
        """Sets external variables for the next solves, each a scalar or a ``(Nn,)`` nodal field on this mesh, e.g. ``T=thermal.thermal``; results show them once solved."""
        Nn = self.mesh.Nn
        for name, v in values.items():
            assert name in self._externalNames, (
                f"{type(self.model).__name__} reads no '{name}'"
            )
            v = np.asarray(v, dtype=float)
            v = np.full(Nn, float(v)) if v.ndim == 0 else v.copy()
            assert v.shape == (Nn,), f"'{name}' must be a scalar or a (Nn,) array"
            nextExternals = {**self.__variables.nextExternals, name: v}
            self.__variables = replace(self.__variables, nextExternals=nextExternals)

    def _Iter_Record(self, **entries: Any) -> None:
        """Replaces the entries `Save_Iter` stores with the next iteration (a loop's convergence); `Set_Iter` restores those of its iteration."""
        self.__variables = replace(self.__variables, record=entries)

    def _Trial_internals(self) -> dict[_GroupElem, dict[str, FeArray]]:
        """Internal variables per group reached from the committed ones by the current state; none by default."""
        return {}

    def _Commit_Internal_and_External_states(self) -> None:
        """Makes the trial internal variables and the external variables set the committed ones, at the end of a converged solve."""
        self.__variables = self.__variables.Committed(
            self._Trial_internals(), self._externalNames
        )

    def __Get_dof_owners(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> tuple[_types.IntArray | None, int]:
        """Owner rank of each dof of the problem's vector (its fields end to end), None when never partitioned, and this rank."""
        fields = self.__Fields(problem)
        spaces = [self._Get_FunctionSpace(field) for field in fields]
        rank = spaces[0].rank
        if any(space._dofOwners is None for space in spaces):
            return None, rank
        owners = [
            np.repeat(space._dofOwners, self.Get_dof_n(field))
            for field, space in zip(fields, spaces)
        ]
        return np.concatenate(owners), rank

    def _Set_solutions(
        self,
        problemType: ProblemType | tuple[ProblemType, ...],
        u: _types.FloatArray,
        v: _types.FloatArray | None = None,
        a: _types.FloatArray | None = None,
    ) -> None:
        if isinstance(problemType, tuple):
            # one composed vector per problem, sliced per field
            for field, start in zip(problemType, self.__Get_offsets(problemType)):
                end = start + self.__Get_size(field)
                self._Set_solutions(
                    field,
                    u[start:end].copy(),
                    None if v is None else v[start:end].copy(),
                    None if a is None else a[start:end].copy(),
                )
            return

        self.__Set_u_n(problemType, u)

        if isinstance(v, np.ndarray):
            self.__Set_v_n(problemType, v)

        if isinstance(a, np.ndarray):
            self.__Set_a_n(problemType, a)

    def _Solver_Solve_problemType(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """Solves the monolithic problem holding `problemType`.\n
        It is recommended to call the resolution via the Solve() function."""

        if MPI_SIZE > 1 and self.isGathered:
            raise RuntimeError(
                "Under MPI, Solve needs a partitioned mesh: mesh under mpirun, or load one saved partitioned, and solve before `_Gather`."
            )

        problem = self._Get_monolithic_problem(problemType)
        if isinstance(problem, tuple):
            # a scheme or a Lagrange condition set since the system was created
            self.__Check_monolithic(problem)
        system = self._Get_linear_system(problem)
        self.__orphanWarned = False

        if self.isNonLinear:
            x = self.__Solve_Newton(problem)
        else:
            x, _ = self.__Solve(problem)

        if isinstance(problem, tuple):
            for field, weights in system.pinnedWeights.items():
                self.__Shift_to_zero_mean(problem, field, weights, x)

        solutions = self.timeScheme.Evaluate_np1(
            x, self._Get_u_n(problem), self._Get_v_n(problem), self._Get_a_n(problem)
        )

        self._Set_solutions(problem, *solutions)

        return solutions[0]

    def __Check_monolithic(self, problem: tuple[ProblemType, ...]) -> None:
        """Rejects what fields solved side by side cannot run yet."""
        names = ", ".join(problem)
        if not isinstance(self.timeScheme, Elliptic):
            # each block would need its own time-scheme coefficients
            raise NotImplementedError(
                f"({names}) solved as one system supports the elliptic algorithm only, not {self.timeScheme!r}."
            )
        if len(self.Bc_Lagrange) > 0:
            raise NotImplementedError(
                f"({names}) solved as one system does not support Lagrange boundary conditions."
            )

    def __Shift_to_zero_mean(
        self,
        problem: tuple[ProblemType, ...],
        field: ProblemType,
        weights: _types.FloatArray,
        x: _types.FloatArray,
    ) -> None:
        """Shifts a field pinned at one node to zero mean, ``∫ field dΩ = 0`` with `weights` ``∫ N_i dΩ``, in place in the problem's vector `x`."""
        start = self.__Get_offsets(problem)[problem.index(field)]
        values = x[start : start + weights.size]
        owned = self.Get_dofs(field)
        comm = self.mesh.comm
        mean = Reduce_sum(weights[owned] @ values[owned], comm) / Reduce_sum(
            weights[owned].sum(), comm
        )
        values -= mean

    def __Get_weights(self, field: ProblemType) -> _types.FloatArray:
        """``∫ N_i dΩ`` of each node of a scalar field."""
        space = self._Get_FunctionSpace(field)
        weights = np.zeros(space.Nn)
        dim = self.mesh.dim
        for groupElem, spaceGroup in zip(
            self.mesh.Get_list_groupElem(dim), space.Get_list_groupElem(dim)
        ):
            V_e = Operators.Linear.V(spaceGroup)
            np.add.at(weights, space.Get_assembly_e(groupElem, 1), V_e)
        return weights

    def _Solver_Get_trial_solution(
        self, problemType: ProblemType | None = None
    ) -> np.ndarray:
        """The solve's iterate of the problem type's field (default the simulation's), a view into the problem's vector; `u_n` outside a solve."""
        if problemType is None:
            problemType = self.problemType
        if self.__trial_solution is None:
            return self._Get_u_n(problemType)
        problem, x = self.__trial_solution
        if not isinstance(problem, tuple):
            return x if problem == problemType else self._Get_u_n(problemType)
        if problemType not in problem:
            return self._Get_u_n(problemType)
        start = self.__Get_offsets(problem)[problem.index(problemType)]
        return x[start : start + self.__Get_size(problemType)]

    def __Solve_Newton(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """Newton-Raphson from `u_n`: `A dx = −R(x)`, `x += dx` until converged."""
        absTol, relTol, incTol, maxIter = self.__Solver_Get_Newton_Raphson_Params()
        name = ", ".join(self.__Fields(problem))
        if MPI_RANK == 0:
            Terminal.Section(f"{name} problem at iteration {self.Niter}")

        tic = Tic()
        x = self._Get_u_n(problem)
        norms: list[float] = []
        converged = False
        self.__trial_solution = (problem, x)
        try:
            for _ in range(maxIter):
                self.Need_Update(problem)
                dx, rhsNorm = self.__Solve(problem, x)
                x += dx
                norms.append(rhsNorm)
                relNorm = rhsNorm / norms[0] if norms[0] > 0 else 0.0
                if MPI_RANK == 0:
                    print(
                        f"At Newton iteration {len(norms)} relative norm is {relNorm:14.12e}"
                    )
                # over the owned rows, so that every rank takes the same branch
                owned = dx[self.Get_dofs(problem)]
                dxNorm = np.sqrt(Reduce_sum(owned @ owned, self.mesh.comm))
                converged = rhsNorm < absTol or relNorm < relTol or dxNorm < incTol
                if converged:
                    break
        finally:
            self.__trial_solution = None

        timeIter = tic.Tac(f"Resolution {name}", "Newton iterations", False)
        if not converged:
            raise AssertionError(
                f"Newton raphson algorithm did not converged in {len(norms)} iterations."
            )
        self._Iter_Record(newtonIter=len(norms), timeIter=timeIter, list_norm_r=norms)
        return x

    def __Solve(
        self,
        problem: ProblemType | tuple[ProblemType, ...],
        x: _types.FloatArray | None = None,
    ) -> tuple[_types.FloatArray, float]:
        """y of `A y = b`, its known dofs at the Dirichlet values, minus x's when given (Newton's increment), 0 when the scheme solves for a; and b's residual norm."""
        _, b = self._Solver_Get_A_b(problem)
        self.__Warn_orphan_conditions(problem)
        size = self.__Get_size(problem)
        if x is None:
            x = np.zeros(size)

        y = np.zeros(b.size)
        if self.timeScheme.unknown != "a":
            dofs = self.__Compose(problem, self.Bc_dofs_Dirichlet, isDofs=True)
            dofs = np.asarray(dofs, dtype=int)
            values = self.__Compose(problem, self.Bc_values_Dirichlet) - x[dofs]
            np.add.at(y, dofs, values)
        known = np.zeros(b.size, dtype=bool)
        known[self.Bc_dofs_known_unknown(problem)[0]] = True

        lagrange = []
        if b.size > size:
            for condition in self.Bc_Lagrange:
                coefs, dofs = condition.lagrangeCoefs, condition.dofs
                value = condition.dofsValues[0] - coefs @ x[dofs]
                lagrange.append(_Lagrange(dofs, coefs, value))
        resolution = self._resolution or (ResolType.r2 if lagrange else ResolType.r1)

        solverType = self._Solver_Get(problem)
        options = ("", "", "")
        if solverType == SolverType.petsc:
            options = self._Solver_Get_PETSc4Py_Options(problem)
        lb, ub = self._Solver_Get_lb_ub(problem)
        solver = _Solver(
            solverType,
            options,
            self._Solver_Get_x0(problem),
            lb,
            ub,
        )

        tic = Tic()
        rhsNorm = self._Get_linear_system(problem).Solve(
            b,
            y,
            known,
            solver,
            self.mesh.comm,
            lagrange=lagrange,
            resolution=resolution,
        )
        backend = ", ".join(options) if solverType == SolverType.petsc else solverType
        tic.Tac("Solver", f"Solve {problem} ({backend})", self._verbosity)
        return y[:size], rhsNorm

    def _Solver_Get_load(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> _types.FloatArray:
        """External load of the monolithic problem holding `problemType`, Neumann + the terms' F, at the end of the step; Lagrange rows included."""
        return self.__Get_load(self._Get_monolithic_problem(problemType))

    def __Get_load(
        self, problem: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """`_Solver_Get_load` of a problem key, its contributions rebuilt when stale."""
        system = self.__Get_folded(problem)
        layout, Ndof = self.__Get_layout(self.__Fields(problem))
        dofs = self.__Compose(problem, self.Bc_dofs_Neumann, isDofs=True)
        dofsValues = self.__Compose(problem, self.Bc_values_Neumann)

        load = np.zeros(Ndof, dtype=float)
        np.add.at(load, np.asarray(dofs, dtype=int), dofsValues)
        load += system.Get_load(layout, Ndof)
        return load

    def __Get_b(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.FloatArray:
        """b of `A dx = b`: the load minus the internal forces, at the trial solution when nonlinear, at 0 when linear; Lagrange rows included."""
        tic = Tic()

        load = self.__Get_load(problemType)
        system = self._Get_linear_system(problemType)
        if self.isNonLinear:
            x = self.__Compose(problemType, self._Solver_Get_trial_solution)
        elif system.internal:
            # R is folded at u_n, not at x = 0
            raise ValueError(
                "A term declaring an internal force R needs the Newton-Raphson algorithm: call `_Solver_Set_Newton_Raphson_Algorithm`."
            )
        else:
            x = np.zeros(self.__Get_size(problemType))
        u_n = self._Get_u_n(problemType)
        v_n = self._Get_v_n(problemType)
        a_n = self._Get_a_n(problemType)
        fields = tuple(
            field if field is not None and field.any() else None
            for field in self.timeScheme.Evaluate(x, u_n, v_n, a_n)
        )
        layout, _ = self.__Get_layout(self.__Fields(problemType))
        b = system.Get_b(load, fields, layout)

        tic.Tac("Solver", f"b ({problemType})", self._verbosity)

        return b

    def __Get_A(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> sparse.csr_matrix:
        """The problem's A for the time scheme, summed from its fold's contributions and kept by its linear system."""
        system = self.__Get_folded(problemType)
        layout, Ndof = self.__Get_layout(self.__Fields(problemType))
        return system.Get_A(
            self.timeScheme.coefs,
            layout,
            Ndof,
            lambda contributing: self.__Get_csr_map(layout, True, Ndof, contributing),
        )

    def __Get_folded(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _LinearSystem:
        """The problem's linear system, its contributions rebuilt when stale."""
        system = self._Get_linear_system(problemType)
        if self.__dict_needUpdate.get(problemType, True):
            system.Set_list_contribution(self.__Get_list_contribution(problemType))
            self.__dict_needUpdate = {**self.__dict_needUpdate, problemType: False}
        return system

    def _Solver_Get_A_b(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]:
        """A and b at the trial solution, before Dirichlet, of the monolithic problem holding `problemType` or of the fields given; A is the object the solve uses, never write into it."""
        if isinstance(problemType, tuple) and len(problemType) > 1:
            problem = problemType
        else:
            problem = self._Get_monolithic_problem(problemType)
        b = self.__Get_b(problem)
        return self.__Get_A(problem), b

    def _Bc_dofs_orphan(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> _types.IntArray:
        """Dofs in no element and no Lagrange condition: known, 0 unless a Dirichlet value sets them, loads dropped."""
        system = self.__Get_folded(problemType)
        layout, Ndof = self.__Get_layout(self.__Fields(problemType))
        connected = system.Connected_rows(layout, Ndof)[: self.__Get_size(problemType)]
        lagrange = self.__Compose(
            problemType,
            lambda field: BoundaryCondition.Get_dofs(field, self.__Bc_Lagrange),  # type: ignore [arg-type]
            isDofs=True,
        )
        connected[np.asarray(lagrange, dtype=int)] = True
        return np.flatnonzero(~connected)

    def __Warn_orphan_conditions(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> None:
        """Warns once per solve when a load or a Dirichlet condition touches an orphan dof. Collective on `mesh.comm`."""
        if self.__orphanWarned:
            return
        self.__orphanWarned = True
        orphans = self._Bc_dofs_orphan(problemType)
        touched = 0
        if orphans.size > 0:
            conditions = [
                self.__Compose(problemType, Bc_dofs, isDofs=True)
                for Bc_dofs in (self.Bc_dofs_Neumann, self.Bc_dofs_Dirichlet)
            ]
            orphans = np.intersect1d(orphans, self.Get_dofs(problemType))
            touched = np.isin(orphans, np.concatenate(conditions)).sum()
        touched = Reduce_sum(int(touched), self.mesh.comm)
        if touched > 0:
            Terminal.MyPrintError(
                f"WARNING: {touched} dofs in no element carry a load or a Dirichlet condition: they stay at 0 (or the Dirichlet value), their loads dropped."
            )

    def _Solver_Set_PETSc4Py_Options(
        self,
        kspType: str = "cg",
        pcType: str = "gamg",
        solverType: str = "petsc",
        problemType: ProblemType | tuple[ProblemType, ...] | None = None,
    ) -> None:
        """Configure PETSc KSP solver options for the linear system Ax = b.

        The constructor auto-selects defaults based on problem type and mesh dimension
        (see ``__init__``). Call this method to override those defaults.

        Rebuilds the kept PETSc solver, so ``PETSc.Options()`` set since are read again.

        Parameters
        ----------
        kspType : str, optional
            Krylov subspace method, by default ``"cg"``.
            Choose based on matrix symmetry:

            - SPD (symmetric positive definite): ``"cg"``, ``"groppcg"``
            - Symmetric indefinite (Lagrange/saddle-point): ``"minres"``, ``"symmlq"``
            - General / non-symmetric: ``"gmres"``, ``"fgmres"``, ``"lgmres"``, ``"bcgs"``
            - Direct solve (no KSP iterations): ``"preonly"``
            - Smoother / specialised: ``"chebyshev"`` (requires spectral bounds, best as MG smoother)

            https://petsc.org/release/manualpages/KSP/KSPType/
        pcType : str, optional
            Preconditioner type, by default ``"gamg"``.

            +----------------+---------------------+--------------------------------------------------+
            | pcType         | Context             | Notes                                            |
            +================+=====================+==================================================+
            | ``"gamg"``     | serial + MPI        | PETSc AMG, no extra pkg                          |
            | ``"hypre"``    | serial + MPI        | BoomerAMG, requires hypre                        |
            | ``"asm"``      | serial + MPI        | overlapping Schwarz                              |
            | ``"bjacobi"``  | serial + MPI        | block Jacobi (ILU/block)                         |
            | ``"ilu"``      | serial only         | incomplete LU; use bjacobi/asm in MPI            |
            | ``"icc"``      | serial only         | incomplete Cholesky, SPD; use bjacobi/asm in MPI |
            | ``"lu"``       | see solverType      | direct LU factorization                          |
            | ``"cholesky"`` | see solverType      | direct Cholesky, SPD only                        |
            | ``"sor"``      | serial only         | SOR / Gauss-Seidel                               |
            | ``"jacobi"``   | serial + MPI        | diagonal scaling                                 |
            | ``"none"``     | serial + MPI        | no preconditioning                               |
            +----------------+---------------------+--------------------------------------------------+

            https://petsc.org/release/manualpages/PC/PCType/
        solverType : str, optional
            Backend for direct-factorization PC (``"lu"`` / ``"cholesky"``),
            by default ``"petsc"``. Ignored when pcType is not a factorization type.

            +--------------------+---------------+-------------------------------+
            | solverType         | Context       | Notes                         |
            +====================+===============+===============================+
            | ``"petsc"``        | serial only   | built-in, seqaij only         |
            | ``"superlu_dist"`` | serial + MPI  | most common parallel direct   |
            | ``"mumps"``        | serial + MPI  | alternative to superlu_dist   |
            | ``"mkl_pardiso"``  | serial + MPI  | Intel, fast on Intel CPUs     |
            | ``"superlu"``      | serial only   |                               |
            | ``"umfpack"``      | serial only   |                               |
            | ``"cholmod"``      | serial only   | SPD only                      |
            +--------------------+---------------+-------------------------------+

            https://petsc.org/release/manual/ksp/#using-external-linear-solvers
        problemType : ProblemType | tuple[ProblemType, ...], optional
            Problem type, or monolithic problem of several fields, to configure. ``None`` applies to all of them.

        Raises
        ------
        ValueError
            - Unknown kspType, pcType, or solverType.
            - ``pcType="lu"/"cholesky"`` with ``kspType`` other than ``"preonly"``
              (direct factorization PCs require no KSP iterations).
            - External solverType paired with a non-factorization pcType.
            - ``pcType="sor"`` in MPI context (serial-only PC).
            - ``pcType="lu"/"cholesky"`` with ``solverType="petsc"`` in MPI context
              (seqaij only — use ``solverType="mumps"`` or ``"superlu_dist"``).
            - Requested external package not available in the current PETSc build.
        """

        # ── Valid value sets ──────────────────────────────────────────────────
        # fmt: off
        _valid_ksp = {
            "cg", "groppcg", "pipecg",                     # SPD
            "minres", "symmlq",                             # symmetric indefinite
            "gmres", "fgmres", "lgmres", "bcgs", "bicg",   # non-symmetric
            "cgne", "cgls", "richardson", "chebyshev",     # specialised (chebyshev: MG smoother)
            "preonly",                                      # direct (no KSP iterations)
        }
        _valid_pc = {
            "gamg", "hypre", "ml",                         # algebraic multigrid
            "asm", "gasm", "bjacobi",                      # domain decomposition
            "ilu", "icc",                                   # incomplete factorization
            "lu", "cholesky",                              # direct (backend = solverType)
            "sor",                                         # SOR / Gauss-Seidel (serial only)
            "jacobi", "none",                              # basic
            "ksp", "fieldsplit",                           # composite
        }
        _valid_solver = {
            "petsc",                                       # built-in (seqaij only)
            "mumps", "superlu_dist", "mkl_pardiso",        # parallel direct
            "superlu", "umfpack", "cholmod",               # serial direct
        }
        # fmt: on

        # ── Input validation ──────────────────────────────────────────────────
        if kspType not in _valid_ksp:
            raise ValueError(
                f"Unknown kspType '{kspType}'. Valid options: {sorted(_valid_ksp)}\n"
                "See https://petsc.org/release/manualpages/KSP/KSPType/"
            )
        if pcType not in _valid_pc:
            raise ValueError(
                f"Unknown pcType '{pcType}'. Valid options: {sorted(_valid_pc)}\n"
                "See https://petsc.org/release/manualpages/PC/PCType/"
            )
        if solverType not in _valid_solver:
            raise ValueError(
                f"Unknown solverType '{solverType}'. Valid options: {sorted(_valid_solver)}\n"
                "See https://petsc.org/release/manual/ksp/#using-external-linear-solvers"
            )

        # Direct-factorization PCs must use preonly (no KSP iterations).
        _direct_pc = {"lu", "cholesky"}
        if pcType in _direct_pc and kspType != "preonly":
            raise ValueError(
                f"pcType='{pcType}' is a direct factorization and requires kspType='preonly' "
                f"(no KSP iterations). Got kspType='{kspType}'.\n"
                "See https://petsc.org/release/overview/linear_solve_table/#direct-solvers."
            )

        # External direct-solver backends only apply to factorization PCs.
        if solverType != "petsc" and pcType not in _direct_pc:
            raise ValueError(
                f"solverType='{solverType}' is a direct-solver backend and requires "
                f"pcType in {sorted(_direct_pc)}, got pcType='{pcType}'."
            )

        # ── MPI compatibility ─────────────────────────────────────────────────
        if MPI_SIZE > 1:
            # PCSOR is sequential: does not support mpiaij.
            if pcType == "sor":
                raise ValueError(
                    "pcType='sor' is sequential and cannot be used in MPI context. "
                    "Use pcType='bjacobi' or 'asm' instead."
                )
            # PETSc built-in LU/Cholesky only supports seqaij, not mpiaij.
            if pcType in _direct_pc and solverType == "petsc":
                raise ValueError(
                    f"solverType='petsc' with pcType='{pcType}' only supports sequential "
                    "matrices (seqaij) and cannot be used in MPI context. "
                    "Use a parallel direct solver: solverType='mumps' or 'superlu_dist'."
                )

        # ── External package availability ─────────────────────────────────────
        _external_pc = {"hypre", "ml"}
        if (solverType != "petsc" or pcType in _external_pc) and _Get_PETSc():
            if solverType != "petsc" and not _PETSc_Has(solverType):
                raise ValueError(
                    f"solverType='{solverType}' is not available in this PETSc build. "
                    "Check your PETSc installation or use solverType='petsc'."
                )
            if pcType in _external_pc and not _PETSc_Has(pcType):
                raise ValueError(
                    f"pcType='{pcType}' requires the '{pcType}' package but it is not "
                    "available in this PETSc build. "
                    "Try pcType='gamg' (built-in AMG) instead."
                )

        # ── Store ─────────────────────────────────────────────────────────────
        list_problemTypes = (
            list(self.problemTypes)
            + [
                problem
                for problem in self._Get_monolithic_problems()
                if len(problem) > 1
            ]
            if problemType is None
            else [problemType]
        )
        self.__Free_solvers(list_problemTypes)
        for problemType in list_problemTypes:
            self.__dict_solver_petsc4py_options[problemType] = (
                kspType,
                pcType,
                solverType,
            )

    def _Solver_Get_PETSc4Py_Options(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> tuple[str, str, str]:
        """Returns (kspType, pcType, solverType) petsc4py options for the given (or default) problem type, or monolithic problem."""

        problemType = self.problemType if problemType is None else problemType

        if problemType not in self.__dict_solver_petsc4py_options:
            self._Solver_Set_PETSc4Py_Options(
                *self.__Default_PETSc4Py_Options(problemType),
                problemType=problemType,
            )

        return self.__dict_solver_petsc4py_options[problemType]

    def __Default_PETSc4Py_Options(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> tuple[str, str, str]:
        """Best available (kspType, pcType, solverType) for the problem type."""
        if isinstance(problemType, tuple):
            # several fields side by side, possibly a saddle point: an external LU, the ones a saddle point is accepted with under MPI
            if _PETSc_Has("mumps"):
                return "preonly", "lu", "mumps"
            return (
                "preonly",
                "lu",
                "superlu_dist" if _PETSc_Has("superlu_dist") else "petsc",
            )
        if self.Get_dof_n(problemType) == 1:
            # Scalar problems
            # cg+gamg: works in serial and MPI, no external packages needed.
            # Good default for scalar SPD problems
            return "cg", "gamg", "petsc"
        # Vector problems
        if _PETSc_Has("superlu_dist"):
            return "preonly", "lu", "superlu_dist"
        if _PETSc_Has("mumps"):
            return "preonly", "lu", "mumps"
        if MPI_SIZE == 1:
            # No external direct solver: built-in LU (serial only).
            return "preonly", "lu", "petsc"
        # MPI context without any parallel direct solver: iterative fallback.
        return "cg", "gamg", "petsc"

    def Get_dofs(
        self, problemType: ProblemType | tuple[ProblemType, ...] | None = None
    ) -> _types.IntArray:
        """Returns (owned) dofs associated with the problem type, or with a monolithic problem's fields end to end."""

        if problemType is None:
            problemType = self.problemType

        if isinstance(problemType, tuple):
            return self.__Compose(problemType, self.Get_dofs, isDofs=True)

        owners, rank = self.__Get_dof_owners(problemType)
        return np.arange(self.__Get_size(problemType))[Owned(owners, rank)]

    # ----------------------------------------------
    # Boundary conditions
    # ----------------------------------------------

    def Bc_Init(self) -> None:
        """Initializes Dirichlet, Neumann and Lagrange boundary conditions"""
        # DIRICHLET
        self.__Bc_Dirichlet: list[BoundaryCondition] = []
        """Dirichlet conditions list[BoundaryCondition]"""
        # NEUMANN
        self.__Bc_Neumann: list[BoundaryCondition] = []
        """Neumann conditions list[BoundaryCondition]"""
        # LAGRANGE
        self.__Bc_Lagrange: list[LagrangeCondition] = []
        """Lagrange conditions list[BoundaryCondition]"""
        self.__Bc_Display: list[BoundaryCondition | LagrangeCondition] = []
        """Boundary conditions for display list[BoundaryCondition]"""

    @property
    def Bc_Dirichlet(self) -> list[BoundaryCondition]:
        """Returns a copy of the Dirichlet conditions."""
        return self.__Bc_Dirichlet.copy()

    @property
    def Bc_Neuman(self) -> list[BoundaryCondition]:
        """Returns a copy of the Neumann conditions."""
        return self.__Bc_Neumann.copy()

    @property
    def Bc_Lagrange(self) -> list[LagrangeCondition]:
        """Returns a copy of the Lagrange conditions."""
        return self.__Bc_Lagrange.copy()

    def _Bc_Add_Lagrange(self, newBc: LagrangeCondition):
        """Adds Lagrange conditions."""
        assert isinstance(newBc, LagrangeCondition)
        self.__Bc_Lagrange.append(newBc)
        # triger the update because when we use lagrange multiplier we need to update the matrix system
        self.Need_Update()

    def _Bc_Lagrange_dim(self, problemType: ProblemType = None) -> int:
        """Calculates the dimension required to resize the system to use Lagrange multipliers."""
        if problemType is None:
            problemType = self.problemType
        # get the number of lagrange conditions applied to the problem
        # len(self.Bc_Lagrange) won't work because we need to filter the problemType
        nBc = BoundaryCondition.Get_nBc(
            problemType,
            self.Bc_Lagrange,  # type: ignore [arg-type]
        )
        return nBc

    @property
    def Bc_Display(self) -> list[BoundaryCondition | LagrangeCondition]:
        """Returns a copy of the boundary conditions for display."""
        return self.__Bc_Display.copy()

    def Bc_vector_Dirichlet(self, problemType: ProblemType = None) -> _types.FloatArray:
        """Returns a vector filled with Dirichlet boundary conditions values."""
        if problemType is None:
            problemType = self.problemType
        dofs = self.Bc_dofs_Dirichlet(problemType)
        dofsValues = self.Bc_values_Dirichlet(problemType)
        size = self.__Get_size(problemType)
        csr_vector = sparse.csr_matrix(
            (dofsValues, (dofs, np.zeros_like(dofs))), shape=(size, 1)
        )
        vector = csr_vector.toarray().ravel()
        return vector

    def Bc_vector_Neumann(self, problemType: ProblemType = None) -> _types.FloatArray:
        """Returns a vector filled with Neuman boundary conditions values."""
        if problemType is None:
            problemType = self.problemType
        dofs = self.Bc_dofs_Neumann(problemType)
        dofsValues = self.Bc_values_Neumann(problemType)
        size = self.__Get_size(problemType)
        csr_vector = sparse.csr_matrix(
            (dofsValues, (dofs, np.zeros_like(dofs))), shape=(size, 1)
        )
        vector = csr_vector.toarray().ravel()
        return vector

    def Bc_dofs_Dirichlet(self, problemType: ProblemType = None) -> _types.IntArray:
        """Returns dofs related to Dirichlet conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_dofs(problemType, self.__Bc_Dirichlet)

    def Bc_values_Dirichlet(self, problemType: ProblemType = None) -> _types.FloatArray:
        """Returns dofs values related to Dirichlet conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Dirichlet)

    def Bc_dofs_Neumann(self, problemType: ProblemType = None) -> _types.IntArray:
        """Returns dofs related to Neumann conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_dofs(problemType, self.__Bc_Neumann)

    def Bc_values_Neumann(self, problemType: ProblemType = None) -> _types.FloatArray:
        """Returns dofs values related to Neumann conditions."""
        if problemType is None:
            problemType = self.problemType
        return BoundaryCondition.Get_values(problemType, self.__Bc_Neumann)

    def Bc_dofs_known_unknown(
        self, problemType: ProblemType | tuple[ProblemType, ...]
    ) -> tuple[_types.IntArray, _types.IntArray]:
        """Returns known and unknown dofs; of a monolithic problem, the fields end to end, a field left free up to a constant pinned at its first DOF."""
        tic = Tic()

        # Build unknown and known dofs
        nDof = self.__Get_size(problemType)
        dofsKnown = np.asarray(
            self.__Compose(problemType, self.Bc_dofs_Dirichlet, isDofs=True), dtype=int
        )
        mask = np.ones(nDof, dtype=bool)
        mask[dofsKnown] = False  # handles duplicates implicitly
        mask[self._Bc_dofs_orphan(problemType)] = False
        if isinstance(problemType, tuple):
            mask[self.__Bc_dofs_pinned(problemType, mask)] = False
        dofsKnown = np.where(~mask)[0]  # unique, sorted
        dofsUnknown = np.where(mask)[0]

        test = dofsKnown.size + dofsUnknown.size
        assert test == nDof, (
            f"Problem under conditions dofsKnown + dofsUnknown - nDof = {test - nDof}"
        )

        tic.Tac("Solver", f"Get dofs ({problemType})", self._verbosity)

        return dofsKnown, dofsUnknown

    def __Bc_dofs_pinned(
        self, problem: tuple[ProblemType, ...], free: _types.BoolArray
    ) -> _types.IntArray:
        """Dofs pinning the fields left free up to a constant, kept by the problem's system until `free` changes. Collective on `mesh.comm`."""
        system = self._Get_linear_system(problem)
        comm = self.mesh.comm
        if Reduce_max(not system.Pinned_for(free), comm):
            system.Set_pinned(free, *self.__Find_pinned(problem, free))
        return system.pinnedDofs

    def __Find_pinned(
        self, problem: tuple[ProblemType, ...], free: _types.BoolArray
    ) -> tuple[_types.IntArray, dict[ProblemType, _types.FloatArray]]:
        """DOF at global node 0 of each scalar field without a diagonal block whose constant the other fields' free equations are blind to, ``‖A_rc[free] · 1‖ ≤ ε ‖A_rc[free]‖``, and the field's zero-mean weights."""
        terms = {field: self.Get_terms(field) for field in problem}
        diagonal = {
            rows
            for slot, rows, cols in self.__Blocks(terms)
            if slot in "KCM" and rows == cols
        }
        A = self.__Get_A(problem)
        owned = np.zeros(A.shape[0], dtype=bool)
        owned[self.Get_dofs(problem)] = True
        pinned: list[int] = []
        weights = {}
        comm = self.mesh.comm
        for field, start in zip(problem, self.__Get_offsets(problem)):
            if self.Get_dof_n(field) != 1 or field in diagonal:
                continue
            isField = np.zeros(A.shape[0], dtype=bool)
            isField[start : start + self.__Get_size(field)] = True
            # rows complete only where owned under MPI
            A_rc = A[:, isField][owned & free & ~isField]
            constant = Reduce_sum(np.sum((A_rc @ np.ones(A_rc.shape[1])) ** 2), comm)
            scale = Reduce_sum(A_rc.multiply(A_rc).sum(), comm)
            if constant <= (1e-10) ** 2 * scale:
                space = self._Get_FunctionSpace(field)
                node = space.Get_dof_nodes(np.flatnonzero(space.globalNodes == 0))
                pinned.extend(start + space.Get_dofs(node, 1))
                weights[field] = self.__Get_weights(field)
        return np.asarray(pinned, dtype=np.int64), weights

    def Bc_dofs_nodes(
        self,
        nodes: _types.IntArray,
        unknowns: list[str],
        problemType: ProblemType = None,
    ) -> _types.IntArray:
        """Returns degrees of freedom associated with the nodes, based on the problem type and unknowns.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes.
        unknowns : list
            unknowns (e.g ["x","y","rz"])
        problemType : str
            Problem type.

        Returns
        -------
        _types.IntArray
            Degrees of freedom.
        """

        if len(nodes) == 0:
            return np.array([], dtype=int)

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        availableUnknowns = self.Get_unknowns(problemType)
        components = BoundaryCondition._Get_components(availableUnknowns, unknowns)

        return self._Get_FunctionSpace(problemType).Get_dofs(
            nodes, len(availableUnknowns), components
        )

    def __Bc_evaluate(
        self, coord: _types.FloatArray, values, option="nodes"
    ) -> _types.FloatArray:
        """Evaluates values at nodes or gauss points."""

        assert option in ["nodes", "gauss"], "Must be in ['nodes','gauss']"
        if option == "nodes":
            values_eval = np.zeros(coord.shape[0])
        elif option == "gauss":
            values_eval = np.zeros((coord.shape[0], coord.shape[1]))
        else:
            raise TypeError("option error")

        if callable(values):
            # Evaluate function at coordinates
            if option == "nodes":
                values_eval[:] = values(coord[:, 0], coord[:, 1], coord[:, 2])
            elif option == "gauss":
                values_eval[:, :] = values(
                    coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]
                )
            else:
                raise TypeError("option error")

        else:
            if option == "nodes":
                values_eval[:] = values
            elif option == "gauss":
                values_eval[:, :] = values
            else:
                raise TypeError("option error")

        return values_eval

    @staticmethod
    def __Bc_values_on(value: Any, kept: _types.BoolArray) -> Any:
        """The values of a per-node array on the ``kept`` nodes (a function or a constant as is); another size raises."""
        if callable(value) or np.size(value) == 1:
            return value
        if np.size(value) != kept.size:
            raise ValueError(
                f"{np.size(value)} values for {kept.size} nodes: pass a function of (x, y, z)."
            )
        return np.asarray(value)[kept]

    def __Bc_check_inputs(self, nodes, values, unknowns) -> bool:
        """Validates inputs shared by all add_* boundary condition methods.

        Returns False when inputs are empty (caller should return early).
        Raises AssertionError when len(values) != len(unknowns).
        """
        if len(nodes) == 0 or len(values) == 0:
            return False
        assert len(values) == len(unknowns), (
            f"len(values)={len(values)} must equal len(unknowns)={len(unknowns)}."
        )
        return True

    def add_dirichlet(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType | None = None,
        description="",
    ) -> None:
        """Adds Dirichlet's boundary conditions.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes
        values : list
            list of values that can contains floats, arrays or functions or functions.\n
            e.g [10, lambda x,y,z: 10*x - 20*y + x*z, _types.FloatArray]\n
            The functions use the x, y and z nodes coordinates.\n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        unknowns : list[str]
            unknowns where values will be applied (e.g ['y', 'x'])
        problemType : ProblemType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Fix all displacement DOFs to zero on the left wall (elastic simulation):

        >>> simu.add_dirichlet(nodes, [0, 0], ["x", "y"])

        Impose temperature 100°C on the top boundary (thermal simulation):

        >>> simu.add_dirichlet(nodes, [100.0], ["t"])

        Spatially varying temperature (uses node coordinates):

        >>> simu.add_dirichlet(nodes, [lambda x, y, z: 100 * x], ["t"])
        """

        if not self.__Bc_check_inputs(nodes, values, unknowns):
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        assert len(nodes) > 0, "Empty node list"
        nodes = np.asarray(nodes)

        # the selected nodes carrying a DOF, in their order (Np,); which of the given nodes are kept (nNodes,)
        space = self._Get_FunctionSpace(problemType)
        spaceNodes = space.Get_dof_nodes(nodes)
        kept = np.isin(nodes, spaceNodes)
        if spaceNodes.size == 0:
            raise ValueError(f"No selected node carries a {problemType} DOF.")
        coordo_n = space.coord[spaceNodes]

        # initialize the value vector for each node
        dofsValues_dir = np.zeros((spaceNodes.size, len(unknowns)))

        for d, _ in enumerate(unknowns):
            value = self.__Bc_values_on(values[d], kept)
            eval_n = self.__Bc_evaluate(coordo_n, value, option="nodes")
            dofsValues_dir[:, d] = eval_n.ravel()

        dofsValues = dofsValues_dir.ravel()

        dofs = self.Bc_dofs_nodes(spaceNodes, unknowns, problemType)

        self._Bc_Add_Dirichlet(
            problemType,
            spaceNodes,
            dofsValues,
            dofs,
            unknowns,
            description,
        )

    def add_neumann(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType = None,
        description="",
    ) -> None:
        """Adds Neumann's boundary conditions.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes
        values : list
            list of values that can contains floats, arrays or functions or functions.\n
            e.g [10, lambda x,y,z: 10*x - 20*y + x*z, _types.FloatArray]\n
            The functions use the x, y and z nodes coordinates.\n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        unknowns : list[str]
            unknowns where values will be applied (e.g ['y', 'x'])
        problemType : ProblemType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Concentrated force of -500 N in y at the beam tip (beam simulation):

        >>> nodes = mesh.Nodes_Point((L, h/2))
        >>> simu.add_neumann(nodes, [-500], ["y"])
        """

        if not self.__Bc_check_inputs(nodes, values, unknowns):
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dofsValues, dofs, nodes = self.__Bc_pointLoad(
            problemType, nodes, values, unknowns
        )

        self._Bc_Add_Neumann(
            problemType, nodes, dofsValues, dofs, unknowns, description
        )

    def add_lineLoad(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType = None,
        description="",
    ) -> None:
        """Adds a linear load.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, _types.FloatArray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        unknowns : list[str]
            unknowns where values will be applied (e.g ['y', 'x'])
        problemType : ProblemType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Distributed load of -1000 N/m along y (elastic simulation):

        >>> simu.add_lineLoad(nodes, [-1000], ["y"])

        Spatially varying load (uses integration-point coordinates):

        >>> simu.add_lineLoad(nodes, [lambda x, y, z: -500 * (1 + x)], ["y"])
        """

        if not self.__Bc_check_inputs(nodes, values, unknowns):
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dofsValues, dofs, nodes = self.__Bc_lineLoad(
            problemType, nodes, values, unknowns
        )

        self._Bc_Add_Neumann(
            problemType, nodes, dofsValues, dofs, unknowns, description
        )

    def add_surfLoad(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType = None,
        description="",
    ) -> None:
        """Adds a surface load.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, _types.FloatArray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        unknowns : list[str]
            unknowns where values will be applied (e.g ['y', 'x'])
        problemType : ProblemType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Uniform pressure of -800 Pa along x:

        >>> simu.add_surfLoad(nodes, [-800], ["x"])
        """

        if not self.__Bc_check_inputs(nodes, values, unknowns):
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dim = self.mesh.dim

        if dim == 2:
            dofsValues, dofs, nodes = self.__Bc_lineLoad(
                problemType, nodes, values, unknowns
            )
            # multiplied by thickness
            dofsValues *= self.model.thickness
        elif dim == 3:
            dofsValues, dofs, nodes = self.__Bc_surfload(
                problemType, nodes, values, unknowns
            )
        else:
            raise NotImplementedError("Unknown configuration.")

        self._Bc_Add_Neumann(
            problemType, nodes, dofsValues, dofs, unknowns, description
        )

    def add_pressureLoad(
        self,
        nodes: _types.IntArray,
        magnitude: float,
        problemType: ProblemType = None,
        description="",
    ) -> None:
        """Adds a pressure.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes. (must belong to the edge of the mesh.)
        magnitude : float
            pressure magnitude
        problemType : str, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Normal inward/outward pressure of 1 MPa on a cylinder surface (elastic simulation):

        >>> simu.add_pressureLoad(cylinder_nodes, magnitude=1e6)
        """

        if len(nodes) == 0 or magnitude == 0:
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        if self.dim == 1:
            Terminal.MyPrintError("Cant apply pressure on 1D mesh.")
            return

        if len(self.Get_unknowns(problemType)) == 0:
            Terminal.MyPrintError("Cant apply pressure on scalar problems.")
            return

        dofsValues, dofs, nodes = self.__Bc_pressureload(problemType, nodes, magnitude)

        unknowns = self.Get_unknowns(problemType)[: self.mesh.inDim]

        self._Bc_Add_Neumann(
            problemType, nodes, dofsValues, dofs, unknowns, description
        )

    def add_volumeLoad(
        self,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
        problemType: ProblemType = None,
        description="",
    ) -> None:
        """Adds a volumetric load.

        Parameters
        ----------
        nodes : _types.IntArray
            nodes
        values : list
            list of values that can contain floats, arrays or functions or lambda functions.\n
            e.g = [10, lambda x,y,z: 10*x - 20*y + x*z, _types.FloatArray] \n
            functions use x, y and z integration points coordinates (x,y,z are in this case arrays of dim (e,p)) \n
            Please note that the functions must take 3 input parameters in the order x, y, z, whether the problem is 1D, 2D or 3D.
        unknowns : list[str]
            unknowns where values will be applied (e.g ['y', 'x'])
        problemType : ProblemType, optional
            problem type, if not specified, we take the basic problem of the problem
        description : str, optional
            Description of the condition, by default "".

        Examples
        --------
        Gravity in -y direction, ρ = 7800 kg/m³ (elastic simulation):

        >>> simu.add_volumeLoad(mesh.nodes, [-7800 * 9.81], ["y"])
        """

        if not self.__Bc_check_inputs(nodes, values, unknowns):
            return

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dim = self.mesh.dim

        if dim == 2:
            dofsValues, dofs, nodes = self.__Bc_surfload(
                problemType, nodes, values, unknowns
            )
            # multiplied by thickness
            dofsValues = dofsValues * self.model.thickness
        elif dim == 3:
            dofsValues, dofs, nodes = self.__Bc_volumeload(
                problemType, nodes, values, unknowns
            )
        else:
            raise NotImplementedError("Unknown configuration.")

        self._Bc_Add_Neumann(
            problemType, nodes, dofsValues, dofs, unknowns, description
        )

    def __Bc_pointLoad(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        values: list,
        unknowns: list,
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Adds a point load shared between the selected nodes carrying a DOF.
        return dofsValues, dofs, those nodes"""

        nodes = np.asarray(nodes)
        space = self._Get_FunctionSpace(problemType)
        spaceNodes = space.Get_dof_nodes(nodes)
        if spaceNodes.size == 0:
            raise ValueError(f"No selected node carries a {problemType} DOF.")
        kept = np.isin(nodes, spaceNodes)
        coord_n = space.coord[spaceNodes]

        # initialize the value vector for each node (Nn, Ndof)
        values_n_d = np.zeros((spaceNodes.size, len(unknowns)))

        for i in range(len(unknowns)):
            value = self.__Bc_values_on(values[i], kept)
            eval_n = self.__Bc_evaluate(coord_n, value, option="nodes")
            eval_n /= spaceNodes.size
            values_n_d[:, i] = eval_n.ravel()

        dofsValues = values_n_d.ravel()

        dofs = self.Bc_dofs_nodes(spaceNodes, unknowns, problemType)

        return dofsValues, dofs, spaceNodes

    @staticmethod
    def __Bc_elements(
        groupElem: "_GroupElem", spaceGroup: "_GroupElem", nodes: _types.IntArray
    ) -> _types.IntArray:
        """Elements whose nodes the selection holds, on the group with the fewest: a space's selection misses a TRI6's mid nodes under P1, a geometry's a TRI3's added nodes under P2."""
        group = min(groupElem, spaceGroup, key=lambda group: group.nPe)
        return group.Get_Elements_Nodes(nodes, exclusively=True)

    def __Bc_Integration_Dim(
        self,
        dim: int,
        problemType: ProblemType,
        nodes: _types.IntArray,
        values: list,
        unknowns: list[str],
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Integrates on the elements of the specified dimension, with the problem type's space (TRI3 → TRI6 under P2).
        return dofsValues, dofs, nodesUsedByElements"""

        list_dofsValues: list = [np.array([])]
        list_dofs: list = [np.array([], dtype=int)]
        list_nodesUsed: list = [np.array([], dtype=int)]
        # nodes != Nodes dont remove
        Nn = self.mesh.Nn

        space = self._Get_FunctionSpace(problemType)

        # For each group element
        for groupElem, spaceGroup in zip(
            self.mesh.Get_list_groupElem(dim), space.Get_list_groupElem(dim)
        ):
            elements = self.__Bc_elements(groupElem, spaceGroup, nodes)
            if elements.shape[0] == 0:
                continue
            connect = groupElem.connect[elements]
            spaceNodes = spaceGroup.connect[elements]
            Ne = elements.shape[0]
            list_nodesUsed.append(spaceNodes.ravel())

            # Get the coordinates of the Gauss points if you need to devaluate the function
            matrixType = MatrixType.mass
            coord_e_p = spaceGroup.Get_GaussCoordinates_e_pg(matrixType, elements)

            N_pg = spaceGroup.Get_N_pg(matrixType)
            wJ_e_pg = spaceGroup.Get_weightedJacobian_e_pg(matrixType)[elements]

            # initialize the matrix of values for each point used by the elements and each gauss point (Ne*nPe, dir)
            dofsValues_u = np.zeros((Ne * spaceGroup.nPe, len(unknowns)))
            # initialize the dofs vector
            dofs_u = np.zeros_like(dofsValues_u, dtype=int)

            # Integrated for all unknowns
            for u, unknown in enumerate(unknowns):
                if isinstance(values[u], (int, float)) or callable(values[u]):
                    # evaluate on gauss points (Ne, nPg)
                    eval_e_p = self.__Bc_evaluate(coord_e_p, values[u], option="gauss")
                    # integrate the elements (Ne, nPg, nPe)
                    values_e_p = np.einsum(
                        "ep,ep,pin->epn", wJ_e_pg, eval_e_p, N_pg, optimize="optimal"
                    )

                else:
                    # evaluate on nodes
                    eval_n = np.zeros(Nn, dtype=float)
                    # the nodes off the geometry are interpolated below
                    onGeometry = np.asarray(nodes) < Nn
                    eval_n[np.asarray(nodes)[onGeometry]] = self.__Bc_values_on(
                        values[u], onGeometry
                    )
                    eval_e = eval_n[connect]  # (Ne, nPe)
                    if spaceGroup is not groupElem:
                        # the node values interpolated at the space's nodes (Ne, nPe space)
                        xi = spaceGroup.Get_Local_Coords()
                        N = groupElem._Eval_Functions(groupElem._N(), xi)[:, 0]
                        eval_e = eval_e @ N.T
                    # integrate the elements (Ne, nPg, nPe)
                    values_e_p = np.einsum(
                        "ep,en,pin->epn", wJ_e_pg, eval_e, N_pg, optimize="optimal"
                    )

                # set calculated (sum on integration points) values and dofs
                dofsValues_u[:, u] = np.sum(values_e_p, axis=1).ravel()
                dofs_u[:, u] = self.Bc_dofs_nodes(
                    spaceNodes.ravel(), [unknown], problemType
                )

            list_dofsValues.append(dofsValues_u.ravel())
            list_dofs.append(dofs_u.ravel())

        dofsValues = np.concatenate(list_dofsValues)
        dofs = np.concatenate(list_dofs).astype(int)
        nodesUsedByElements = np.concatenate(list_nodesUsed).astype(int)

        return dofsValues, dofs, nodesUsedByElements

    def __Bc_lineLoad(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        values: list,
        unknowns: list,
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Adds a linear load.\n
        returns dofsValues, dofs, nodes"""

        self._Check_dofs(problemType, unknowns)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(
            dim=1,
            problemType=problemType,
            nodes=nodes,
            values=values,
            unknowns=unknowns,
        )

        return dofsValues, dofs, nodes

    def __Bc_surfload(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        values: list,
        unknowns: list,
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Apply a surface force.\n
        returns dofsValues, dofs, nodes"""

        self._Check_dofs(problemType, unknowns)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(
            dim=2,
            problemType=problemType,
            nodes=nodes,
            values=values,
            unknowns=unknowns,
        )

        return dofsValues, dofs, nodes

    def __Bc_volumeload(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        values: list,
        unknowns: list,
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Adds a volumetric load.\n
        returns dofsValues, dofs, nodes"""

        self._Check_dofs(problemType, unknowns)

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(
            dim=3,
            problemType=problemType,
            nodes=nodes,
            values=values,
            unknowns=unknowns,
        )

        return dofsValues, dofs, nodes

    def __Bc_pressureload(
        self, problemType: ProblemType, nodes: _types.IntArray, magnitude: float
    ) -> tuple[_types.FloatArray, _types.IntArray, _types.IntArray]:
        """Adds a pressure load.\n
        returns dofsValues, dofs, nodes"""

        # here we need to get the normal vector

        mesh = self.mesh
        dim = mesh.dim
        inDim = mesh.inDim

        if dim == 2:
            # will use 1D elements
            magnitude *= self.model.thickness

        # issue #29 revealed an error in this function.
        # Both methods below yield similar (though not identical) results.

        # -------------------
        # Method 1
        # -------------------
        # the normals live on the geometry's nodes of the selected elements
        space = self._Get_FunctionSpace(problemType)
        geometryNodes = np.unique(
            np.concatenate(
                [
                    groupElem.connect[
                        self.__Bc_elements(groupElem, spaceGroup, nodes)
                    ].ravel()
                    for groupElem, spaceGroup in zip(
                        mesh.Get_list_groupElem(dim - 1),
                        space.Get_list_groupElem(dim - 1),
                    )
                ]
            )
        )
        unknowns = self.Get_unknowns(problemType)[:inDim]
        if geometryNodes.size == 0:
            return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
        normals, nodes = mesh.Get_normals(geometryNodes)

        values = [val * magnitude for val in normals[:, :inDim].T]

        dofsValues, dofs, nodes = self.__Bc_Integration_Dim(
            dim - 1,
            problemType=problemType,
            nodes=nodes,
            values=values,
            unknowns=unknowns,
        )

        return dofsValues, dofs, nodes

        # # -------------------
        # # Method 2:
        # # -------------------

        # unknowns = self.Get_unknowns(problemType)[:inDim]
        # dofsValues = np.array([], dtype=float)
        # dofs = np.array([], dtype=int)
        # Nodes = np.array([], dtype=int)  # Nodes used by the elements
        # # nodes != Nodes dont remove

        # # For each group element
        # for groupElem in self.mesh.Get_list_groupElem(dim - 1):
        #     # Retrieve elements that exclusively use nodes
        #     elements = groupElem.Get_Elements_Nodes(nodes, exclusively=True)
        #     if elements.shape[0] == 0:
        #         continue
        #     connect = groupElem.connect[elements]
        #     Nodes = np.append(Nodes, np.reshape(connect, -1))

        #     matrixType = MatrixType.mass
        #     N_pg = groupElem.Get_N_pg(matrixType)
        #     wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)[elements]

        #     # evaluate normals on gauss points (Ne, nPg, inDim)
        #     normals_e_pg = groupElem.Get_normals_e_pg(matrixType)[elements, :, :inDim]
        #     # integrate the elements (Ne, nPg, nPe, inDim)
        #     values_e_pg = magnitude * np.einsum(
        #         "ep,epd,pin->epnd", wJ_e_pg, normals_e_pg, N_pg, optimize="optimal"
        #     )

        #     # sum over integration points (Ne, nPe, inDim)
        #     values_e = np.sum(values_e_pg, axis=1)
        #     # append dofs values
        #     dofsValues = np.append(dofsValues, np.ravel(values_e))

        #     # append new dofs
        #     new_dofs = self.Bc_dofs_nodes(connect.ravel(), unknowns, problemType)
        #     dofs = np.append(dofs, new_dofs.ravel())

        # return dofsValues, dofs, Nodes

    def _Bc_Add_Neumann(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        dofsValues: _types.FloatArray,
        dofs: _types.IntArray,
        unknowns: list[str],
        description="",
    ) -> None:
        """Adds Neumann's boundary conditions.\n
        If a neumann condition is already applied to the dof, the condition will not be taken into account for the dof.
        """

        # selected nodes on this rank, but no complete element of the selection
        if len(dofs) == 0:
            if MPI_SIZE > 1:
                return
            raise ValueError(f"No {problemType} element lies in the selection.")

        tic = Tic()

        self._Check_dofs(problemType, unknowns)

        new_Bc = BoundaryCondition(
            problemType, nodes, dofs, unknowns, dofsValues, f"Neumann {description}"
        )
        self.__Bc_Neumann.append(new_Bc)

        tic.Tac("Boundary Conditions", "Add Neumann condition ", self._verbosity)

    def _Bc_Add_Dirichlet(
        self,
        problemType: ProblemType,
        nodes: _types.IntArray,
        dofsValues: _types.FloatArray,
        dofs: _types.IntArray,
        unknowns: list,
        description="",
    ) -> None:
        """Adds Dirichlet's boundary conditions.\n
        If a Dirichlet's dof is entered more than once, the conditions are added together.
        """

        tic = Tic()

        self.__Check_problemTypes(problemType)

        new_Bc = BoundaryCondition(
            problemType, nodes, dofs, unknowns, dofsValues, f"Dirichlet {description}"
        )

        self.__Bc_Dirichlet.append(new_Bc)

        tic.Tac("Boundary Conditions", "Add Dirichlet condition", self._verbosity)

    # Functions to create links between degrees of freedom

    def _Bc_Add_Display(
        self,
        nodes: _types.IntArray,
        unknowns: list[str],
        description: str,
        problemType: ProblemType = None,
    ) -> None:
        """Adds a display condition."""

        if problemType is None:
            problemType = self.problemType

        self.__Check_problemTypes(problemType)

        dofs = self.Bc_dofs_nodes(nodes, unknowns, problemType)

        dofsValues = np.array([0] * len(dofs))

        new_Bc = BoundaryCondition(
            problemType, nodes, dofs, unknowns, dofsValues, description
        )
        self.__Bc_Display.append(new_Bc)

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    def _Results_Check_Available(self, result: str) -> bool:
        """Check that the result is available"""
        availableResults = self.Results_Available()
        if result in availableResults:
            return True
        else:
            Terminal.MyPrintError(
                f"\nFor a {self.problemType} problem result must be in : \n {availableResults}"
            )
            return False

    def Results_Set_Iteration_Summary(self, text: str, remove: bool = False) -> None:
        """Sets and print the iteration's summary."""
        end = "\r" if remove else "\n"
        Terminal.MyPrint(text, end=end)

    def Results_Get_Iteration_Summary(self) -> str:
        """Returns the iteration's summary, empty when the simulation specifies none."""
        return ""

    def Results_Set_Bc_Summary(self) -> None:
        """Sets the simulation loading summary."""

    def Results_Get_Bc_Summary(self) -> str:
        """Returns the simulation loading summary, empty when the simulation specifies none."""
        return ""

    def Results_Reshape_values_n(
        self, values_n: _types.FloatArray, nodeValues: bool
    ) -> _types.FloatArray:
        """Node values ``(Nn, ...)`` or raveled, at nodes or averaged over each element's nodes."""
        mesh = self.mesh
        Nn = mesh.Nn
        if values_n.size % Nn != 0:
            raise ValueError(f"{values_n.size} values cannot be at the {Nn} nodes.")
        is1d = values_n.ndim == 1
        values_n = values_n.reshape(Nn, -1)

        if nodeValues:
            return values_n.ravel() if is1d else values_n

        # element order matches Get_list_groupElem(dim) / Get_Node_Values
        values_e = np.concatenate(
            [
                np.mean(values_n[groupElem.connect], axis=1)
                for groupElem in mesh.Get_list_groupElem(mesh.dim)
            ]
        )
        return values_e.ravel() if is1d else values_e

    def Results_Reshape_values_e(
        self, values_e: _types.FloatArray, nodeValues: bool
    ) -> _types.FloatArray:
        """Element values ``(Ne, ...)`` or raveled, on elements or averaged at nodes; at nodes collective on a partitioned mesh, the ghost rows their owners'."""
        mesh = self.mesh
        Ne = mesh.Ne
        if values_e.size % Ne != 0:
            raise ValueError(f"{values_e.size} values cannot be on the {Ne} elements.")
        is1d = values_e.ndim == 1
        values_e = values_e.reshape(Ne, -1)

        if not nodeValues:
            return values_e.ravel() if is1d else values_e

        # exact on the owned rows only
        values_n = mesh.Get_Node_Values(values_e)
        Scatter_forward(values_n, mesh._Get_FunctionSpace()._Get_exchange())
        return values_n.ravel() if is1d else values_n

    def Save(
        self,
        folder: str,
        filename: str = "simulation",
        gather=False,
        additionalInfos: str = "",
    ) -> str | None:
        """Saves the simulation as 'filename.pickle' and its summary in the folder; on a partitioned mesh, each rank its partition in 'Rank{r}/', the iterations kept in memory written to it as `Save_Iter` would. Collective unless `gather`."""

        if gather:
            self._Gather()
            if MPI_RANK != 0:
                return None

        partition = self.__Partition(self.mesh)
        root = (
            folder
            if partition is None
            else Folder.Join(folder, f"Rank{partition.rank}")
        )
        path_simu = Folder.Join(root, f"{filename}.pickle", mkdir=True)

        # meshes read from the folder they were saved in, before it changes
        list_mesh = []
        for i in range(self.Nmesh):
            path = Folder.Join(root, "Meshes", f"mesh{i}.pickle", mkdir=True)
            self.__Get_mesh(i)._Dump(path)
            list_mesh.append(Folder.os.path.relpath(path, folder))
        self.__listMesh = list_mesh
        self.folder = folder

        # the rank's rows of the iterations kept in memory, in the same files as Save_Iter's
        if partition is not None:
            # an earlier serial save would read first, one over more ranks as this one's
            if partition.rank == 0 and Folder.Exists(
                old := Folder.Join(folder, f"{filename}.pickle")
            ):
                Folder.os.remove(old)
            stale = partition.size
            while partition.rank == 0 and Folder.Exists(
                old := Folder.Join(folder, f"Rank{stale}", f"{filename}.pickle")
            ):
                Folder.os.remove(old)
                stale += 1
            _iterations.Write_memory(
                self.__list_results,
                folder,
                partition.comm,
                lambda index: self.__Partition(self.__Get_mesh(index)),
            )

        # Save simulation
        with open(path_simu, "wb") as file:
            pickle.dump(self, file)

        # Save simulation summary
        path_summary = Folder.Join(folder, "summary.txt", mkdir=True)
        summary = f"Simulation completed on: {datetime.now()}\n"
        summary += f"version: {__version__}"
        if MPI_SIZE > 1:
            summary += f"\nNproc: {MPI_SIZE}"
        summary += str(self)
        if str(additionalInfos) != "":
            summary += Terminal.Section("Additional information", False)
            summary += "\n" + str(additionalInfos)

        if MPI_RANK == 0:
            with open(path_summary, "w", encoding="utf8") as file:
                file.write(summary)

            path = folder.replace(Folder.EASYFEA_DIR, "")
            Terminal.MyPrint(f"Saved simulation and summary in:\n{path}\n", "green")

        return path_simu


def Load_Simu(folder: str, filename: str = "simulation", gather=False) -> _Simu:
    """Loads the simulation from the specified folder; under MPI a simulation saved over as many ranks, each rank its partition, to resume it.

    Parameters
    ----------
    folder : str
        simulation's folder.
    filename : str, optional
        The simualtion's name, by default "simulation".
    gather : bool, optional
        Gather simu on root rank, by default False.

    Returns
    -------
    _Simu
        The loaded simulation.
    """

    path_simu = Folder.Join(folder, f"{filename}.pickle")
    ranks = 0
    while Folder.Exists(Folder.Join(folder, f"Rank{ranks}", f"{filename}.pickle")):
        ranks += 1
    if MPI_SIZE > 1:
        if ranks != MPI_SIZE:
            raise ValueError(
                f"{folder} holds a simulation saved over {ranks or 1} rank(s): load it on as many, not {MPI_SIZE}."
            )
        path_simu = Folder.Join(folder, f"Rank{MPI_RANK}", f"{filename}.pickle")
    elif ranks > 0 and not Folder.Exists(path_simu):
        paths = [
            Folder.Join(folder, f"Rank{r}", f"{filename}.pickle") for r in range(ranks)
        ]
        simus = [_Load_pickle(path, folder) for path in paths]
        if any(simu is None for simu in simus):
            return None  # type: ignore [return-value]
        simus[0]._Assemble_ranks(simus)
        return simus[0]
    assert Folder.Exists(path_simu), f"The file {path_simu} cannot be found."

    if MPI_SIZE > 1:
        simu = _iterations._Collectively(MPI_COMM, _Loaded, path_simu, folder)
    else:
        simu = _Load_pickle(path_simu, folder)
    if simu is None:
        return None  # type: ignore [return-value]
    # the same partition on the same ranks: shared again
    simu.mesh.isPartitioned = MPI_SIZE > 1

    if gather:
        simu._Gather()
    return simu


def _Loaded(path: str, folder: str) -> _Simu:
    """:func:`_Load_pickle`, raising when the file is empty."""
    simu = _Load_pickle(path, folder)
    if simu is None:
        raise ValueError(f"{path} is empty or corrupted.")
    return simu


def _Load_pickle(path: str, folder: str) -> _Simu:
    """The simulation pickled to `path`, saved in `folder`; None when the file is empty."""
    try:
        with open(path, "rb") as file:
            simu: _Simu = pickle.load(file)
    except EOFError:
        Terminal.MyPrintError(f"The file:\n{path}\nis empty or corrupted.")
        return None  # type: ignore [return-value]
    simu.folder = folder
    return simu


class _RankPart(NamedTuple):
    """What a rank owns of a simulation, in global ids, from which `_Gather` and a serial `Load_Simu` assemble the whole one."""

    meshes: list[_MeshPart]
    shares: list[_iterations.Share]
    """Where the rank's rows of the iterations on each mesh go."""
    state: _iterations.Iteration
    """The current dof fields, internal and external variables (``externals/``, ``next/``)."""
    memory: dict[int, _iterations.Iteration]
    """The iterations kept in memory, by index."""
    parameters: list[list[tuple[_types.IntArray, np.ndarray]] | None]
    """Each parameter's values per group with their element global ids, None unless one per element."""
