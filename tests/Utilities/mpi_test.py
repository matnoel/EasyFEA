# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Serial tests run under plain `pytest`; `mpi` tests under `make test-mpi` (2 ranks)."""

import numpy as np
import pytest

from EasyFEA.Utilities import (
    MPI_RANK,
    MPI_SIZE,
    rank0_only,
    Reduce_sum,
    Reduce_min,
    Reduce_max,
    Allgather,
)


@rank0_only
def _answer() -> int:
    return 42


def test_rank0_only_runs_the_function_in_serial():
    assert _answer() == 42


@pytest.mark.mpi
def test_rank0_only_returns_none_off_rank_0():
    assert _answer() == (42 if MPI_RANK == 0 else None)


def test_Reduce_sum_is_the_identity_in_serial():
    array = np.array([1.0, 2.0])
    assert Reduce_sum(3.0) == 3.0
    assert Reduce_sum(array) is array


@pytest.mark.mpi
def test_Reduce_sum_sums_over_the_ranks():
    assert Reduce_sum(MPI_RANK + 1) == MPI_SIZE * (MPI_SIZE + 1) // 2
    np.testing.assert_array_equal(
        Reduce_sum(np.array([1.0, MPI_RANK])),
        [MPI_SIZE, MPI_SIZE * (MPI_SIZE - 1) / 2],
    )


def test_Reduce_min_is_the_identity_in_serial():
    assert Reduce_min(-2.0) == -2.0


@pytest.mark.mpi
def test_Reduce_min_ignores_a_rank_with_nothing_to_contribute():
    assert Reduce_min(np.inf if MPI_RANK == 0 else float(MPI_RANK)) == 1.0


def test_Reduce_max_is_the_identity_in_serial():
    assert Reduce_max(-2.0) == -2.0


@pytest.mark.mpi
def test_Reduce_max_returns_the_largest_over_the_ranks():
    assert Reduce_max(float(MPI_RANK)) == MPI_SIZE - 1


def test_Allgather_wraps_the_value_in_serial():
    assert Allgather("a") == ["a"]


@pytest.mark.mpi
def test_Allgather_returns_every_rank_value_in_rank_order():
    assert Allgather(MPI_RANK) == list(range(MPI_SIZE))
