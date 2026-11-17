# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Iteration store: a simulation's saved iterations, a list of :data:`Entry`."""

import copy
import os
import pickle
from dataclasses import dataclass, field, replace
from typing import Any, TypedDict


from ..FEM import FeArray, _GroupElem
from ..Utilities import Folder, _types
from ..Utilities._mpi import MPI_RANK, MPI_SIZE


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


Entry = Iteration | tuple[dict[str, Any], str, str]
"""A saved iteration: the `Iteration` itself in memory, or ``(scalars, folder, file)`` once written to ``folder/Results/file``, its scalars kept in memory."""


@dataclass(frozen=True)
class Variables:
    """A simulation's internal and external variables and record, rebound on any change so that `_Reading` binds them back."""

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


def Append(entries: list[Entry], snapshot: Iteration, folder: str) -> None:
    """Stores `snapshot` in memory if `folder` is empty, else in a pickle there (one per rank under MPI) with its scalars kept in memory too."""
    iteration: Iteration = {**snapshot, "scalars": copy.deepcopy(snapshot["scalars"])}
    if folder == "":
        entries.append(iteration)
        return
    suffix = f"_rank{MPI_RANK}" if MPI_SIZE > 1 else ""
    file = f"results{len(entries)}{suffix}.pickle"
    _Dump(iteration, _Results_path(folder, file, mkdir=True))
    entries.append((iteration["scalars"], folder, file))


def Get(entries: list[Entry], index: int, folder: str, isGathered: bool) -> Iteration:
    """A copy of the iteration at `index`, read from the folder it was written in, else from `folder` for a moved run; under MPI after a `_Gather`, refused if it holds internal or external variables."""
    entry = _Entry_at(entries, index)
    if isinstance(entry, dict):
        iteration = copy.deepcopy(entry)
    else:
        scalars, writtenFolder, file = entry
        path = _Results_path(writtenFolder, file)
        if not Folder.Exists(path):
            path = _Results_path(folder, file)
        loaded: Iteration = _Load(path)
        iteration = {**loaded, "scalars": copy.deepcopy(scalars)}
    assert not (
        MPI_SIZE > 1 and isGathered and (iteration["nodal"] or iteration["quadrature"])
    ), (
        "Internal and external variables are saved per rank, so they cannot be read after a _Gather: read them before gathering."
    )
    return iteration


def Get_scalars(entries: list[Entry], index: int) -> dict[str, Any]:
    """A copy of the scalars of the iteration at `index`, from memory."""
    entry = _Entry_at(entries, index)
    return copy.deepcopy(entry[0] if isinstance(entry, tuple) else entry["scalars"])


def _Entry_at(entries: list[Entry], index: int) -> Entry:
    index = int(index)
    Niter = len(entries)
    if index < 0:
        index += Niter
    assert 0 <= index < Niter, f"`iter` must be in [0, {Niter})"
    return entries[index]


def _Results_path(folder: str, file: str, mkdir: bool = False) -> str:
    return Folder.Join(folder, "Results", file, mkdir=mkdir)


def _Dump(obj: Any, path: str) -> None:
    """Pickles `obj` to `path` through a temporary file replaced at once, so that a reader never sees a file half written."""
    tmp = f"{path}.tmp{MPI_RANK}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _Load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
