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
    nodes: _types.IntArray
    """The owned nodes, in the space's node ids."""


@dataclass(frozen=True)
class Layout:
    """How the arrays of a saved iteration lie on a rank's mesh: the rows it owns, which a saved iteration holds, and how the others are filled back; every row of a mesh held whole."""

    rank: int
    comm: Any
    dofs: dict[str, DofRows]
    nodes: _types.IntArray
    """The owned mesh nodes, the rows of the external variables."""
    nodeSpace: _FunctionSpace
    """The space of the mesh nodes."""
    groups: dict[str, tuple[_GroupElem, ElemType]]
    """By ``elemType.value``: the group of the internal variables and the type of its mesh group, its key in a rank's mesh file."""
    meshGroups: dict[ElemType, _GroupElem]
    """The main-dimension mesh groups, whose elements hold the dof nodes."""

    @property
    def size(self) -> int:
        return self.comm.Get_size()


class Anchors(NamedTuple):
    """An element holding each of some nodes of a space, by which they are found on another mesh of the same elements whatever its node numbering."""

    elemTypes: tuple[ElemType, ...]
    groups: _types.IntArray
    """The position of each node's mesh group in `elemTypes`."""
    elements: _types.IntArray
    """The global id of its element."""
    columns: _types.IntArray
    """Its column in the element's connect."""


class Share(NamedTuple):
    """Where a rank's owned rows of a saved iteration go in the whole mesh: a dof node by an element holding it, a mesh node and an element by global id."""

    dofs: dict[str, Anchors]
    nodes: _types.IntArray
    elements: dict[str, _types.IntArray]
    """By ``elemType.value``."""


class Rows(NamedTuple):
    """The rows of the whole mesh a rank owns, those its saved iterations hold, in their order."""

    dofs: dict[str, _types.IntArray]
    """The dof-node rows of each dof field."""
    nodes: _types.IntArray
    elements: dict[str, _types.IntArray]
    """By ``elemType.value``."""


class Pieces(NamedTuple):
    """The whole mesh of a run saved over the ranks, and the rows each rank owns of it: how an iteration saved over the ranks reads whole."""

    whole: Layout
    rows: list[Rows]


def Share_of(layout: Layout) -> Share:
    """Where the rows `layout` owns go in the whole mesh."""
    groups = layout.meshGroups
    anchors: dict[int, Anchors] = {}
    for d in layout.dofs.values():
        if id(d.space) not in anchors:
            anchors[id(d.space)] = _Anchors(d.space, groups, d.nodes)
    return Share(
        {name: anchors[id(d.space)] for name, d in layout.dofs.items()},
        layout.nodeSpace.globalNodes[layout.nodes],
        {
            key: group._globalElements[group._ownedElements]
            for key, (group, _) in layout.groups.items()
        },
    )


def Rows_in(share: Share, whole: Layout) -> Rows:
    """The rows of `whole`, a whole mesh, where those of `share` go."""
    return Rows(
        {
            name: whole.dofs[name].space.Get_dofs(
                _Anchored(anchors, whole.dofs[name].space, whole.meshGroups), 1
            )
            for name, anchors in share.dofs.items()
        },
        share.nodes,
        share.elements,
    )


def Placed(parts: list[tuple[str, Rows, Iteration]], whole: Layout) -> Iteration:
    """The iteration on `whole`, a whole mesh, from every rank's owned rows (`Owned`), each labelled for the error raised when its rows are not the rank's."""

    def place(kind: str, size: int, rows: list, values: list) -> Any:
        placed = np.zeros_like(values[0], shape=(size, *values[0].shape[1:]))
        for (label, _, _), ids, values_r in zip(parts, rows, values):
            _Check_rows(label, kind, values_r.shape[0], ids.size)
            placed[ids] = values_r
        return placed

    first = parts[0][2]
    dofs = {}
    for name in first["dofs"]:
        dof_n, space = whole.dofs[name].dof_n, whole.dofs[name].space
        dofs[name] = place(
            f"dof field {name}",
            space.Nn,
            [rows.dofs[name] for _, rows, _ in parts],
            [owned["dofs"][name].reshape(-1, dof_n) for _, _, owned in parts],
        ).ravel()
    nodal = {
        name: place(
            f"external variable {name}",
            whole.nodeSpace.Nn,
            [rows.nodes for _, rows, _ in parts],
            [owned["nodal"][name] for _, _, owned in parts],
        )
        for name in first["nodal"]
    }
    quadrature = {
        key: {
            name: place(
                f"internal variable {key}/{name}",
                whole.groups[key][0].Ne,
                [rows.elements[key] for _, rows, _ in parts],
                [owned["quadrature"][key][name] for _, _, owned in parts],
            )
            for name in fields
        }
        for key, fields in first["quadrature"].items()
    }
    return {"scalars": {}, "dofs": dofs, "nodal": nodal, "quadrature": quadrature}


def _Anchors(
    space: _FunctionSpace, groups: dict[ElemType, _GroupElem], nodes: _types.IntArray
) -> Anchors:
    group, element, column = (np.full(space.Ncoords, -1) for _ in range(3))
    for g, groupElem in enumerate(groups.values()):
        connect = space.Get_connect(groupElem)
        group[connect] = g
        element[connect] = groupElem._globalElements[:, None]
        column[connect] = np.arange(connect.shape[1])
    assert (group[nodes] >= 0).all(), "A dof node lies on no main-dimension element."
    return Anchors(tuple(groups), group[nodes], element[nodes], column[nodes])


def _Anchored(
    anchors: Anchors, space: _FunctionSpace, groups: dict[ElemType, _GroupElem]
) -> _types.IntArray:
    """The node ids of `space` on the whole mesh of `groups` that `anchors` point at."""
    nodes = np.empty(anchors.groups.size, dtype=int)
    for g, elemType in enumerate(anchors.elemTypes):
        isIn = anchors.groups == g
        connect = space.Get_connect(groups[elemType])
        nodes[isIn] = connect[anchors.elements[isIn], anchors.columns[isIn]]
    return nodes


def Append(
    entries: list[Entry],
    snapshot: Iteration,
    folder: str,
    partition: Layout | None,
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
    partition_of: Callable[[int], Layout | None],
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
    partition: Layout | None,
    pieces: Pieces | None,
) -> Iteration:
    """A copy of the iteration at `index` with `scalars`, read from the folder it was written in, else from `folder` for a moved run; over a partition, collective: the ghost rows from their owners; on a whole mesh, one saved over the ranks placed from `pieces`."""
    entry = _Entry_at(entries, index)
    if partition is not None and entry.ranks != partition.size:
        raise ValueError(
            f"Iteration {index} was saved over {entry.ranks or 1} rank(s) and reads on as many or serially, here {partition.size}."
        )
    if isinstance(entry.file, dict):
        assert partition is not None or entry.ranks is None, (
            f"Iteration {index} holds a rank's rows in memory."
        )
        arrays = copy.deepcopy(entry.file)
    elif entry.ranks is None:
        path = _Results_path(entry.folder, entry.file)
        if not Folder.Exists(path):
            path = _Results_path(folder, entry.file)
        arrays = _Load(path)
    elif partition is None:
        if pieces is None or len(pieces.rows) != entry.ranks:
            raise ValueError(
                f"Iteration {index} was saved over {entry.ranks} ranks, whose meshes this simulation lacks: gather or load it again."
            )
        own = Folder.Join(_Rank_dir(entry.folder, 0), "Results", entry.file)
        written = entry.folder if Folder.Exists(own) else folder
        arrays = _Read_pieces(written, entry.file, pieces)
    else:
        own = Folder.Join(
            _Rank_dir(entry.folder, partition.rank), "Results", entry.file
        )
        isThere = Allgather(Folder.Exists(own), partition.comm)
        written = entry.folder if all(isThere) else folder
        arrays = _Read(written, entry.file, scalars["indexMesh"], partition)
    return {**arrays, "scalars": scalars}


def Place_memory(
    entries: list[Entry],
    owned: list[dict[int, Iteration]],
    pieces_of: Callable[[int], Pieces],
) -> None:
    """Rank 0's entries kept in memory, each set whole from every rank's owned rows (`Owned_memory`)."""
    for index, iteration in owned[0].items():
        entry = _Entry_at(entries, index)
        assert entry.scalars is not None
        pieces = pieces_of(entry.scalars["indexMesh"])
        parts = [
            (f"rank {r}", rows, part[index])
            for r, (rows, part) in enumerate(zip(pieces.rows, owned))
        ]
        arrays = Placed(parts, pieces.whole)
        entries[index] = Entry("", arrays, None, entry.scalars)


def Owned_memory(
    entries: list[Entry], comm, layout_of: Callable[[int], Layout]
) -> dict[int, Iteration]:
    """The owned rows of each iteration kept in memory over the ranks, by index; collective on `comm`."""
    owned = {}
    for index in range(len(entries)):
        entry = _Entry_at(entries, index)
        if isinstance(entry.file, dict) and entry.ranks is not None:
            layout = layout_of(Get_scalars(entries, index, comm)["indexMesh"])
            owned[index] = Owned(entry.file, layout)
    return owned


def _Read_pieces(folder: str, file: str, pieces: Pieces) -> Iteration:
    """The iteration written to ``folder/Rank{r}/Results/file`` by every rank, placed whole."""
    parts = []
    for r, rows in enumerate(pieces.rows):
        path = Folder.Join(_Rank_dir(folder, r), "Results", file)
        parts.append((path, rows, _Load(path)))
    return Placed(parts, pieces.whole)


def Get_scalars(entries: list[Entry], index: int, comm) -> dict[str, Any]:
    """A copy of the scalars of the iteration at `index`, from memory: rank 0's on every rank of `comm`, collective; refused on another rank when `comm` is None."""
    scalars = Bcast(_Entry_at(entries, index).scalars, comm)
    if scalars is None:
        raise ValueError("A saved iteration's scalars are held by rank 0 only.")
    return copy.deepcopy(scalars)


def _Entry_of(
    folder: str,
    file: "str | Iteration",
    partition: Layout | None,
    scalars: dict[str, Any],
) -> Entry:
    if partition is None:
        return Entry(folder, file, None, scalars)
    return Entry(folder, file, partition.size, scalars if partition.rank == 0 else None)


def _Write(arrays: Iteration, folder: str, file: str, partition: Layout | None) -> None:
    """Pickles `arrays` to ``folder/Results/file``; over a partition, the owned rows to ``folder/Rank{r}/Results/file``, then waits for every rank, raising on all if one failed."""
    if partition is None:
        _Dump(arrays, _Results_path(folder, file, mkdir=True))
        return
    path = Folder.Join(_Rank_dir(folder, partition.rank), "Results", file, mkdir=True)
    _Collectively(partition.comm, _Dump, Owned(arrays, partition), path)


def Owned(arrays: Iteration, layout: Layout) -> Iteration:
    """The rows of `arrays` that `layout` owns, without scalars."""
    p = layout
    return {
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


def _Read(folder: str, file: str, indexMesh: int, partition: Layout) -> Iteration:
    """The rank's arrays of the iteration written to ``folder/Rank{r}/Results/file``: the owned rows from its file, the ghost nodes by scatter, the ghost elements from their owners' files, found in their mesh files. Collective, raising on every rank if one cannot read its rows."""
    p = partition
    arrays = _Collectively(p.comm, _Read_rows, folder, file, indexMesh, p)
    for name, values in arrays["dofs"].items():
        p.dofs[name].space.Scatter_forward(values, p.dofs[name].dof_n)
    for values in arrays["nodal"].values():
        p.nodeSpace.Scatter_forward(values, values[:1].size)
    return arrays


def _Read_rows(folder: str, file: str, indexMesh: int, p: Layout) -> Iteration:
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
        rows, dof_n, space, _ = p.dofs[name]
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
