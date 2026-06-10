# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import os
import numpy as np
from functools import wraps

from ._requires import Create_requires_decorator

# MPI launchers (mpirun, srun, mpiexec, …) set at least one of these env vars.
# Only import mpi4py when actually running under an MPI launcher to avoid
# MPI_Init hanging when invoked as a plain `python script.py`.
_MPI_LAUNCHER_VARS = [
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH / Hydra
    "SLURM_NTASKS",  # Slurm srun
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2
    "PMIX_RANK",  # PMIx (Open MPI 4+, Slurm)
]
_UNDER_MPIRUN = any(os.environ.get(v) for v in _MPI_LAUNCHER_VARS)

if _UNDER_MPIRUN:
    try:
        from mpi4py import MPI

        CAN_USE_MPI = True
        MPI_COMM = MPI.COMM_WORLD
        MPI_SIZE = MPI_COMM.Get_size()
        MPI_RANK = MPI_COMM.Get_rank()

    except Exception:
        CAN_USE_MPI = False
        MPI_COMM = None
        MPI_SIZE = 1
        MPI_RANK = 0
else:
    CAN_USE_MPI = False
    MPI_COMM = None
    MPI_SIZE = 1
    MPI_RANK = 0

requires_mpi = Create_requires_decorator("mpi4py")


def rank0_only(func):
    """Decorator: only rank 0 executes the function. Non-root ranks return `None`.

    Use on file-output and visualization functions (Display, GLTF, USD, Vizir)
    to prevent non-root MPI ranks from writing files or rendering plots.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if MPI_RANK == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


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


@requires_mpi
def Sync_dofsValues(
    dofsValues: np.ndarray,
    dofs: np.ndarray,
    ordering: np.ndarray = None,  # type: ignore[assignment]
) -> np.ndarray:
    """Reconstruct the full DOF solution vector from distributed partial contributions.

    Each rank holds correct values only at its owned ``dofs``.
    This function gathers each rank's owned slice into a single global buffer via ``Allgatherv`` and scatters the values back to their EasyFEA-global positions.
    ``Allgatherv`` moves ``N`` doubles total across the communicator (vs ``N × MPI_SIZE`` for the previous ``Allreduce(SUM)``
    implementation), so the per-call communication volume drops by a factor ``MPI_SIZE`` for the same correctness contract.

    Parameters
    ----------
    dofsValues : np.ndarray
        DOF vector of shape ``(N,)``. Values at ``dofs`` are authoritative;
        values at all other indices are ignored.
    dofs : np.ndarray
        Indices of the DOFs owned by this rank. Must be disjoint from the
        owned sets of all other ranks and their union must cover ``[0, N)``.
    ordering : np.ndarray, optional
        Concatenation of every rank's ``dofs`` array in rank order (``[rank0.dofs, rank1.dofs, ...]``).
        Pass when the caller already has it — e.g. computed once for paired calls on ``x`` and ``lagrange`` — to avoid a second ``allgather`` collective.
        Falls back to an internal ``Concatenate_array(dofs)``.

    Returns
    -------
    np.ndarray
        Full DOF vector of shape ``(N,)`` with correct values on every rank.
        Positions not covered by any rank's ``dofs`` remain ``0`` (same as the previous Allreduce-SUM semantic).
    """
    # Owned slice in this rank's local order
    local = np.ascontiguousarray(dofsValues[dofs])

    # Per-rank slice sizes — small-int allgather, negligible cost
    counts = np.array(MPI_COMM.allgather(local.size), dtype=int)
    displs = np.zeros_like(counts)
    displs[1:] = counts.cumsum()[:-1]

    # Buffer covering the concatenated owned slices, in rank order
    buf = np.empty(int(counts.sum()), dtype=dofsValues.dtype)
    MPI_COMM.Allgatherv(
        sendbuf=[local, MPI.DOUBLE], recvbuf=[buf, counts, displs, MPI.DOUBLE]
    )

    if ordering is None:
        ordering = Concatenate_array(dofs)

    # Non-owned positions stay 0 (matches the previous Allreduce-SUM result
    # for positions not in any rank's `dofs`).
    full = np.zeros_like(dofsValues)
    full[ordering] = buf
    return full
