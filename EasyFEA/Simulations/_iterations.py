# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Iteration store: a simulation's saved iterations, a list of :data:`Entry`."""

import copy
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple, TypedDict


import numpy as np

from ..FEM import ElemType, FeArray, _GroupElem
from ..FEM._function_space import _FunctionSpace
from ..Utilities import Folder, _types
from ..Utilities._mpi import MPI_RANK, Allgather, Bcast


class Iteration(TypedDict):
    """A saved iteration, by kind."""

    scalars: dict[str, Any]
    """``indexMesh``, the record and the user's entries."""
    dofs: dict[str, _types.FloatArray]
    """Dof fields by name."""
    nodal: dict[str, _types.FloatArray]
    """External variables by name."""
    quadrature: dict[str, dict[str, FeArray]]
    """Internal variables by ``elemType.value``, then name."""


class Entry(NamedTuple):
    """A saved iteration: its arrays, in memory or written to a results file, and its scalars."""

    folder: str
    """The simulation folder the results file was written in, ``""`` in memory."""
    file: "str | Iteration"
    """The results file in ``folder/Results/``, or the iteration itself in memory, its scalars empty."""
    ranks: int | None
    """The ranks the iteration was saved over, each its own rows; None when saved whole."""
    scalars: dict[str, Any] | None
    """The iteration's scalars, held by rank 0's entries only."""


@dataclass(frozen=True)
class Variables:
    """A simulation's internal and external variables and record, rebound on any change so that `_Iter_Visit` binds them back."""

    internals: dict[_GroupElem, dict[str, FeArray]] = field(default_factory=dict)
    """Internal variables by group, then name, committed at the last converged solve; none before any."""
    externals: dict[str, _types.FloatArray] = field(default_factory=dict)
    """External variables the last converged solve ran with; none before any."""
    nextExternals: dict[str, _types.FloatArray] = field(default_factory=dict)
    """External variables set for the next solves."""
    record: dict[str, Any] = field(default_factory=dict)
    """Entries the next saved iteration records, a loop's convergence."""

    def Solved_externals(self, names: tuple[str, ...]) -> dict[str, _types.FloatArray]:
        """External variables of the last converged solve, those set for the next before any; all of `names`, or raises."""
        return _Checked(self.externals or self.nextExternals, names)

    def Committed(
        self, internals: dict[_GroupElem, dict[str, FeArray]], names: tuple[str, ...]
    ) -> "Variables":
        """With `internals` and the external variables set as those of the last converged solve."""
        externals = _Checked(self.nextExternals, names)
        return replace(self, internals=internals, externals=externals)

    def Snapshot(
        self,
        scalars: dict[str, Any],
        dofs: dict[str, _types.FloatArray],
        names: tuple[str, ...],
    ) -> Iteration:
        """Copies of the variables beside `scalars` and `dofs`, as an `Iteration`, the record among the scalars."""
        return {
            "scalars": {**scalars, **self.record},
            "dofs": dofs,
            "nodal": {k: v.copy() for k, v in self.Solved_externals(names).items()},
            "quadrature": {
                groupElem.elemType.value: {name: v.copy() for name, v in fields.items()}
                for groupElem, fields in self.internals.items()
            },
        }

    def Restored(
        self, iteration: Iteration, groups: dict[str, _GroupElem]
    ) -> "Variables":
        """Copies of the variables of `iteration`, its internal ones on `groups` by ``elemType.value``, and the recorded entries it holds."""
        internals = {
            groups[elemType]: {name: v.copy() for name, v in fields.items()}
            for elemType, fields in iteration["quadrature"].items()
        }
        externals = {name: v.copy() for name, v in iteration["nodal"].items()}
        record = {
            name: iteration["scalars"].get(name, v) for name, v in self.record.items()
        }
        return Variables(internals, externals, externals, record)


def _Checked(
    externals: dict[str, _types.FloatArray], names: tuple[str, ...]
) -> dict[str, _types.FloatArray]:
    missing = set(names) - set(externals)
    assert not missing, (
        f"set {sorted(missing)} with Set_external before solving or saving"
    )
    return externals


def Flat(iteration: Iteration) -> dict[str, Any]:
    """`iteration` under the flat keys of `Get_results`: dof field names, ``internal/<elemType>/<name>``, ``external/<name>`` and the scalars."""
    return {
        **iteration["dofs"],
        **{
            f"internal/{elemType}/{name}": v
            for elemType, fields in iteration["quadrature"].items()
            for name, v in fields.items()
        },
        **{f"external/{name}": v for name, v in iteration["nodal"].items()},
        **iteration["scalars"],
    }


class DofRows(NamedTuple):
    """A dof field's owned rows on a rank."""

    rows: _types.IntArray
    """The owned rows of its nodes."""
    dof_n: int
    space: _FunctionSpace


@dataclass(frozen=True)
class Partition:
    """A rank's share of a mesh the ranks share: the rows of its arrays it owns, which a saved iteration holds, and how the others are filled back."""

    rank: int
    comm: Any
    dofs: dict[str, DofRows]
    nodes: _types.IntArray
    """The owned mesh nodes, the rows of the external variables."""
    nodeSpace: _FunctionSpace
    """The space of the mesh nodes."""
    groups: dict[str, tuple[_GroupElem, ElemType]]
    """By ``elemType.value``: the group of the internal variables and the type of its mesh group, its key in a rank's mesh file."""

    @property
    def size(self) -> int:
        return self.comm.Get_size()


def Append(
    entries: list[Entry],
    snapshot: Iteration,
    folder: str,
    partition: Partition | None,
) -> None:
    """Stores `snapshot` in memory if `folder` is empty, else in a pickle there without its scalars, kept in rank 0's entry; over a partition, its owned rows in ``Rank{r}/``, collective."""
    scalars = copy.deepcopy(snapshot["scalars"])
    arrays: Iteration = {**snapshot, "scalars": {}}
    if folder == "":
        entries.append(_Entry_of(folder, arrays, partition, scalars))
        return
    file = f"results{len(entries)}.pickle"
    _Write(arrays, folder, file, partition)
    entries.append(_Entry_of(folder, file, partition, scalars))


def Write_memory(
    entries: list[Entry],
    folder: str,
    comm,
    partition_of: Callable[[int], Partition | None],
) -> None:
    """Writes the iterations kept in memory to `folder` as `Append` would have, each over the partition of its mesh index; collective on `comm`."""
    for index in range(len(entries)):
        entry = _Entry_at(entries, index)
        if not isinstance(entry.file, dict):
            continue
        partition = partition_of(Get_scalars(entries, index, comm)["indexMesh"])
        file = f"results{index}.pickle"
        _Write(entry.file, folder, file, partition)
        entries[index] = entry._replace(folder=folder, file=file)


def Get(
    entries: list[Entry],
    index: int,
    folder: str,
    scalars: dict[str, Any],
    partition: Partition | None,
) -> Iteration:
    """A copy of the iteration at `index` with `scalars`, read from the folder it was written in, else from `folder` for a moved run; over a partition, collective: the ghost rows from their owners."""
    entry = _Entry_at(entries, index)
    ranks = None if partition is None else partition.size
    if entry.ranks != ranks:
        raise ValueError(
            f"Iteration {index} was saved over {entry.ranks or 1} rank(s) and reads on as many, here {ranks or 1}."
        )
    if isinstance(entry.file, dict):
        arrays = copy.deepcopy(entry.file)
    elif partition is None:
        path = _Results_path(entry.folder, entry.file)
        if not Folder.Exists(path):
            path = _Results_path(folder, entry.file)
        arrays = _Load(path)
    else:
        own = Folder.Join(
            _Rank_dir(entry.folder, partition.rank), "Results", entry.file
        )
        isThere = Allgather(Folder.Exists(own), partition.comm)
        written = entry.folder if all(isThere) else folder
        arrays = _Read(written, entry.file, scalars["indexMesh"], partition)
    return {**arrays, "scalars": scalars}


def Get_scalars(entries: list[Entry], index: int, comm) -> dict[str, Any]:
    """A copy of the scalars of the iteration at `index`, from memory: rank 0's on every rank of `comm`, collective; refused on another rank when `comm` is None."""
    scalars = Bcast(_Entry_at(entries, index).scalars, comm)
    if scalars is None:
        raise ValueError("A saved iteration's scalars are held by rank 0 only.")
    return copy.deepcopy(scalars)


def _Entry_of(
    folder: str,
    file: "str | Iteration",
    partition: Partition | None,
    scalars: dict[str, Any],
) -> Entry:
    if partition is None:
        return Entry(folder, file, None, scalars)
    return Entry(folder, file, partition.size, scalars if partition.rank == 0 else None)


def _Write(
    arrays: Iteration, folder: str, file: str, partition: Partition | None
) -> None:
    """Pickles `arrays` to ``folder/Results/file``; over a partition, the owned rows to ``folder/Rank{r}/Results/file``, then waits for every rank, raising on all if one failed."""
    if partition is None:
        _Dump(arrays, _Results_path(folder, file, mkdir=True))
        return
    p = partition
    owned: Iteration = {
        "scalars": {},
        "dofs": {
            name: values.reshape(-1, p.dofs[name].dof_n)[p.dofs[name].rows].ravel()
            for name, values in arrays["dofs"].items()
        },
        "nodal": {name: values[p.nodes] for name, values in arrays["nodal"].items()},
        "quadrature": {
            elemType: {
                name: values[p.groups[elemType][0]._ownedElements]
                for name, values in fields.items()
            }
            for elemType, fields in arrays["quadrature"].items()
        },
    }
    path = Folder.Join(_Rank_dir(folder, p.rank), "Results", file, mkdir=True)
    _Collectively(p.comm, _Dump, owned, path)


def _Read(folder: str, file: str, indexMesh: int, partition: Partition) -> Iteration:
    """The rank's arrays of the iteration written to ``folder/Rank{r}/Results/file``: the owned rows from its file, the ghost nodes by scatter, the ghost elements from their owners' files, found in their mesh files. Collective, raising on every rank if one cannot read its rows."""
    p = partition
    arrays = _Collectively(p.comm, _Read_rows, folder, file, indexMesh, p)
    for name, values in arrays["dofs"].items():
        p.dofs[name].space.Scatter_forward(values, p.dofs[name].dof_n)
    for values in arrays["nodal"].values():
        p.nodeSpace.Scatter_forward(values, values[:1].size)
    return arrays


def _Read_rows(folder: str, file: str, indexMesh: int, p: Partition) -> Iteration:
    """The rank's arrays of the iteration, its ghost nodes still zero; raises if a file holds the rows of another partition."""
    path = Folder.Join(_Rank_dir(folder, p.rank), "Results", file)
    own: Iteration = _Load(path)

    def place(kind: str, values_f: np.ndarray, rows, size: int) -> np.ndarray:
        values = np.zeros_like(values_f, shape=(size, *values_f.shape[1:]))
        _Check_rows(path, kind, values_f.shape[0], values[rows].shape[0])
        values[rows] = values_f
        return values

    dofs: dict[str, _types.FloatArray] = {}
    for name, values_f in own["dofs"].items():
        rows, dof_n, space = p.dofs[name]
        values_f = values_f.reshape(-1, dof_n)
        dofs[name] = place(f"dof field {name}", values_f, rows, space.Nn).ravel()

    nodal = {
        name: place(f"external variable {name}", values_f, p.nodes, p.nodeSpace.Nn)
        for name, values_f in own["nodal"].items()
    }

    owners: dict[int, tuple[str, Iteration, dict[ElemType, _types.IntArray]]] = {}
    quadrature: dict[str, dict[str, FeArray]] = {}
    for elemType, fields in own["quadrature"].items():
        group, meshType = p.groups[elemType]
        elementOwners = group._elementOwners
        assert elementOwners is not None
        owned = elementOwners == group.rank
        quadrature[elemType] = {}
        for name, values_f in fields.items():
            kind = f"internal variable {elemType}/{name}"
            values = place(kind, values_f, owned, group.Ne)
            for q in np.unique(elementOwners[~owned]).tolist():
                if q not in owners:
                    pathOwner = Folder.Join(_Rank_dir(folder, q), "Results", file)
                    owners[q] = (
                        pathOwner,
                        _Load(pathOwner),
                        _Owned_elements(folder, q, indexMesh),
                    )
                pathOwner, iterationOwner, ownedElements = owners[q]
                valuesOwner = iterationOwner["quadrature"][elemType][name]
                ownedIds = ownedElements[meshType]
                _Check_rows(pathOwner, kind, valuesOwner.shape[0], ownedIds.size)
                ghosts = np.flatnonzero(elementOwners == q)
                ids = group._globalElements[ghosts]
                values[ghosts] = valuesOwner[np.searchsorted(ownedIds, ids)]
            quadrature[elemType][name] = values

    return {"scalars": {}, "dofs": dofs, "nodal": nodal, "quadrature": quadrature}


def _Collectively(comm, function: Callable, *args) -> Any:
    """`function(*args)` on every rank of `comm`, each raising the first rank's error if one failed, never leaving the others waiting."""
    try:
        result, error = function(*args), None
    except (OSError, KeyError, ValueError, pickle.UnpicklingError) as e:
        result, error = None, f"Rank {comm.Get_rank()}: {type(e).__name__}: {e}"
    errors = [e for e in Allgather(error, comm) if e is not None]
    if errors:
        raise ValueError(errors[0])
    return result


def _Owned_elements(
    folder: str, rank: int, indexMesh: int
) -> dict[ElemType, _types.IntArray]:
    """Global ids of the elements `rank` owns per group, sorted, from its mesh file."""
    data = _Load(
        Folder.Join(_Rank_dir(folder, rank), "Meshes", f"mesh{indexMesh}.pickle")
    )
    owned = {}
    for elemType, (_, partition, *_) in data.items():
        globalElements, owners = partition["globalElements"], partition["elementOwners"]
        if owners is not None:
            owned[elemType] = globalElements[owners == partition["rank"]]
    return owned


def _Check_rows(path: str, kind: str, rows: int, owned: int) -> None:
    if rows != owned:
        raise ValueError(
            f"{path} holds {rows} rows of the {kind} where its rank owns {owned}: saved over another partition."
        )


def _Entry_at(entries: list[Entry], index: int) -> Entry:
    index = int(index)
    Niter = len(entries)
    if index < 0:
        index += Niter
    assert 0 <= index < Niter, f"`iter` must be in [0, {Niter})"
    return _Entry(entries[index])


def _Entry(entry: "Entry | Iteration | tuple") -> Entry:
    """`entry`, or an older run's: an `Iteration` in memory or ``(scalars, folder, file)``."""
    if isinstance(entry, Entry):
        return entry
    if isinstance(entry, dict):
        return Entry("", {**entry, "scalars": {}}, None, entry["scalars"])
    scalars, folder, file = entry
    return Entry(folder, file, None, scalars)


def _Results_path(folder: str, file: str, mkdir: bool = False) -> str:
    return Folder.Join(folder, "Results", file, mkdir=mkdir)


def _Rank_dir(folder: str, rank: int) -> str:
    return Folder.Join(folder, f"Rank{rank}")


def _Dump(obj: Any, path: str) -> None:
    """Pickles `obj` to `path` through a temporary file replaced at once, so that a reader never sees a file half written."""
    tmp = f"{path}.tmp{MPI_RANK}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _Load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
