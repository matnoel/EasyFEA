# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import numpy as np
from functools import wraps

from ._requires import Create_requires_decorator


try:
    from mpi4py import MPI

    CAN_USE_MPI = True
    MPI_COMM = MPI.COMM_WORLD
    MPI_SIZE = MPI_COMM.Get_size()
    MPI_RANK = MPI_COMM.Get_rank()

except ModuleNotFoundError:
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
    "Returns `np.concatenate(MPI_COMM.allgather(array))`."
    all_array = MPI_COMM.allgather(array)
    array = np.concatenate(all_array)
    return array
