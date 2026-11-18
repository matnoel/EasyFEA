# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import os
import warnings

import numpy as np
from functools import wraps
from typing import Callable

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


def Gather(value, comm) -> list | None:
    """Every rank's `value` in rank order on rank 0 of `comm`, None on the others; `[value]` when `comm` is None."""

    if comm is None:
        return [value]

    return comm.gather(value, root=0)


def Gather_rows(
    ids: np.ndarray, values: np.ndarray, size: int, comm
) -> np.ndarray | None:
    """On rank 0 of `comm`, every rank's `values` at their rows `ids` of an array like them of `size` rows (rank 0's `size` counts); None on the others."""
    parts = Gather((ids, values), comm)
    if parts is None:
        return None
    placed = np.zeros_like(values, shape=(size, *values.shape[1:]))
    for ids_r, values_r in parts:
        placed[ids_r] = values_r
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
