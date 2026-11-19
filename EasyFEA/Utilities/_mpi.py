# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import os
import warnings

import numpy as np
from functools import wraps
from typing import Any, Callable, NamedTuple

from ._requires import Create_requires_decorator, Params, R

# MPI launchers (mpirun, srun, mpiexec, …) set at least one of these env vars.
# Only import mpi4py when actually running under an MPI launcher to avoid
# MPI_Init hanging when invoked as a plain `python script.py`.
_MPI_LAUNCHER_VARS = [
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH / Hydra, Slurm pmi2
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2
    "PMIX_RANK",  # PMIx (Open MPI 4+, Slurm pmix)
]
_UNDER_MPIRUN = any(os.environ.get(v) for v in _MPI_LAUNCHER_VARS)

# EASYFEA_MPI=off: every rank runs its own serial simulation, e.g. a parameter sweep the script spreads with mpi4py
if _UNDER_MPIRUN and os.environ.get("EASYFEA_MPI", "").lower() != "off":
    try:
        from mpi4py import MPI

        CAN_USE_MPI = True
        MPI_COMM = MPI.COMM_WORLD
        MPI_SIZE = MPI_COMM.Get_size()
        MPI_RANK = MPI_COMM.Get_rank()

    except Exception as error:
        CAN_USE_MPI = False
        MPI_COMM = None
        MPI_SIZE = 1
        MPI_RANK = 0
        warnings.warn(
            f"launched under an MPI launcher but mpi4py could not be loaded ({error}): every process "
            "runs the whole simulation on its own. See docs/howto/use_mpi.md.",
            RuntimeWarning,
        )
else:
    CAN_USE_MPI = False
    MPI_COMM = None
    MPI_SIZE = 1
    MPI_RANK = 0

requires_mpi = Create_requires_decorator("mpi4py")


def rank0_only(func: Callable[Params, R]) -> Callable[Params, R | None]:
    """Decorator: only rank 0 executes the function. Non-root ranks return `None`.

    Use on file-output and visualization functions (Matplotlib, GLTF, USD, Vizir)
    to prevent non-root MPI ranks from writing files or rendering plots.
    """

    @wraps(func)
    def wrapper(*args: Params.args, **kwargs: Params.kwargs) -> R | None:
        if MPI_RANK == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


# Reductions
# ----------
# A partitioned run computes every integral, extremum and count over the rank's own rows, so a result is only global once reduced over the mesh's communicator: `Reduce_sum(value, mesh.comm)`. `comm` is None when the rank holds the mesh whole (serial, gathered, loaded), and then nothing is reduced, so a script is written once for any number of ranks.
#
# Collective on `comm`: every rank of it must call them, in the same order.


def Reduce_sum(value, comm):
    """Sum of `value` over the ranks of `comm`, elementwise for arrays; `value` itself when `comm` is None."""

    if comm is None:
        return value

    return comm.allreduce(value, op=MPI.SUM)


def Reduce_min(value: float, comm) -> float:
    """Smallest `value` over the ranks of `comm`; scalars only, a rank with nothing to contribute passes `np.inf`."""

    if comm is None:
        return value

    return comm.allreduce(value, op=MPI.MIN)


def Reduce_max(value: float, comm) -> float:
    """Largest `value` over the ranks of `comm`; scalars only, a rank with nothing to contribute passes `-np.inf`."""

    if comm is None:
        return value

    return comm.allreduce(value, op=MPI.MAX)


def Exscan_sum(value: int, comm) -> int:
    """Sum of `value` over the ranks of `comm` before this one, 0 on the first; 0 when `comm` is None."""

    if comm is None:
        return 0

    return comm.exscan(value, op=MPI.SUM) or 0


def Allgather(value, comm) -> list:
    """Every rank's `value` in rank order, `[value]` when `comm` is None; reduce locally first and gather the candidate, never the raw data."""

    if comm is None:
        return [value]

    return comm.allgather(value)


def Bcast(value, comm):
    """Rank 0's `value` on every rank of `comm`; `value` itself when `comm` is None."""

    if comm is None:
        return value

    return comm.bcast(value, root=0)


def Gather(value, comm) -> list | None:
    """Every rank's `value` in rank order on rank 0 of `comm`, None on the others; `[value]` when `comm` is None."""

    if comm is None:
        return [value]

    return comm.gather(value, root=0)


# Owned and ghost rows
# --------------------
# A partitioned array holds its rank's rows, ghosts included: the leading axis indexes rows, each with a global id and an owner rank (-1 for none). Every owned/ghost question goes through these four functions.


class Exchange(NamedTuple):
    """The rows a rank sends its neighbours and the ghost rows it receives, built once by `Build_exchange`; never pickled (its graph communicator does not)."""

    size: int
    """number of rows of the arrays it scatters"""
    send: np.ndarray
    """rows sent, grouped by destination"""
    recv: np.ndarray
    """ghost rows received, grouped by source"""
    sendCounts: np.ndarray
    recvCounts: np.ndarray
    graph: Any
    """the neighbour graph communicator"""


def Owned(owners: np.ndarray | None, rank: int) -> np.ndarray | slice:
    """Rows `rank` owns: a mask over `owners`, `slice(None)` when there are none (never partitioned)."""
    if owners is None:
        return slice(None)
    return owners == rank


def Sum_owned(values, owners: np.ndarray | None, rank: int, comm):
    """Sum over the leading axis of the rows `rank` owns, reduced over `comm`: each row counted once over the ranks. Collective on `comm`."""
    return Reduce_sum(np.asarray(values)[Owned(owners, rank)].sum(axis=0), comm)


def Build_exchange(globalIds: np.ndarray, owners: np.ndarray, comm) -> Exchange | None:
    """The exchange of the rows of global ids `globalIds` and owners `owners` (-1 for none, never a ghost), None when `comm` is None. Collective on `comm`."""
    if comm is None:
        return None
    rank = comm.Get_rank()
    rows = np.arange(globalIds.size)
    isGhost = (owners != rank) & (owners >= 0)
    order = np.argsort(owners[isGhost], kind="stable")
    ghostOwners = owners[isGhost][order]
    ghostIds = globalIds[isGhost][order]
    recv = rows[isGhost][order]
    sources, recvCounts = np.unique(ghostOwners, return_counts=True)

    # the ghosts each rank owns, asked of it; what the others ask of this rank
    splits = np.searchsorted(ghostOwners, np.arange(1, comm.Get_size()))
    asked = comm.alltoall(np.split(ghostIds, splits))
    sendCounts = np.array([ids.size for ids in asked], dtype=int)
    destinations = np.flatnonzero(sendCounts)
    asked = np.concatenate([np.empty(0, dtype=globalIds.dtype), *asked])
    sorter = np.argsort(globalIds, kind="stable")
    found = np.searchsorted(globalIds, asked, sorter=sorter)
    found = sorter[np.minimum(found, sorter.size - 1)] if sorter.size else found
    assert np.array_equal(globalIds[found], asked), (
        "A rank owns a row it does not hold."
    )

    graph = comm.Create_dist_graph_adjacent(
        sources.tolist(), destinations.tolist(), reorder=False
    )
    return Exchange(
        globalIds.size,
        found,
        recv,
        sendCounts[destinations],
        recvCounts,
        graph,
    )


def Scatter_forward(values: np.ndarray, exchange: Exchange | None) -> None:
    """Copies onto the ghost rows of `values` (rows on the leading axis, any trailing shape and dtype) their owners' values, in place; nothing when `exchange` is None. Collective on its graph."""
    if exchange is None:
        return
    assert values.shape[0] == exchange.size, (
        f"{values.shape[0]} rows, the exchange has {exchange.size}."
    )
    rows = values.reshape(exchange.size, -1)
    assert np.shares_memory(rows, values) or values.size == 0, (
        "values must be contiguous."
    )
    width = rows.shape[1]
    send = np.ascontiguousarray(rows[exchange.send])
    recv = np.empty((exchange.recv.size, width), dtype=rows.dtype)

    def Counts(counts):
        counts = counts * width
        return counts.tolist(), (np.cumsum(counts) - counts).tolist()

    exchange.graph.Neighbor_alltoallv(
        [send, Counts(exchange.sendCounts)], [recv, Counts(exchange.recvCounts)]
    )
    rows[exchange.recv] = recv


def Place_rows(parts: list[tuple[np.ndarray, np.ndarray]], size: int) -> np.ndarray:
    """An array of `size` rows holding each part's values at its rows: ``[(ids, values), ...]``."""
    values = parts[0][1]
    placed = np.zeros_like(values, shape=(size, *values.shape[1:]))
    for ids, values in parts:
        placed[ids] = values
    return placed


@requires_mpi
def Concatenate_array(array: np.ndarray) -> np.ndarray:
    """Returns the rank-ordered concatenation of every rank's ``array``.

    Equivalent to ``np.concatenate(MPI_COMM.allgather(array))`` but uses a buffer-based ``Allgatherv`` instead of pickled ``allgather``. The small
    int-array case (owned-DOF indices) is the hot caller; switching to ``Allgatherv`` avoids the per-rank pickle/unpickle round-trip.

    Ranks may contribute arrays of different sizes — ``counts`` and ``displs`` drive the variable layout. All ranks must agree on the trailing dims;
    only the leading dim (size) may vary per rank.
    """
    array = np.ascontiguousarray(array)
    counts = np.array(MPI_COMM.allgather(array.size), dtype=int)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=array.dtype)
    displs = np.zeros_like(counts)
    displs[1:] = counts.cumsum()[:-1]

    mpi_type = MPI._typedict[array.dtype.char]
    buf = np.empty(total, dtype=array.dtype)
    MPI_COMM.Allgatherv(
        sendbuf=[array, mpi_type],
        recvbuf=[buf, counts, displs, mpi_type],
    )
    return buf
