# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Drivers solving a `Problem` through a `System`: `Linear`, `Newton`."""

from typing import Protocol

import numpy as np
from scipy import sparse

from ..Utilities import _types
from ..Utilities._mpi import MPI_RANK
from ._backends import _Backend
from ._constraints import Constraints
from ._system import System


class Problem(Protocol):
    """A problem of `size` unknowns: `A dx = b` at an iterate x, and the constraints x + dx meets."""

    @property
    def size(self) -> int: ...

    def Linearize(
        self, x: _types.FloatArray
    ) -> tuple[sparse.csr_matrix, _types.FloatArray]: ...

    def Constraints(self, x: _types.FloatArray) -> Constraints: ...


class NotConvergedError(AssertionError):
    """A nonlinear driver ran out of iterations."""


def Linear(
    problem: Problem, system: System, backend: _Backend, comm
) -> _types.FloatArray:
    """x of `A x = b`: one step from 0, the known dofs at their targets. Collective on `comm`."""
    dx, _ = _Step(problem, system, backend, np.zeros(problem.size), comm)
    return dx


def Newton(
    problem: Problem,
    system: System,
    backend: _Backend,
    x: _types.FloatArray,
    comm,
    absTol: float,
    relTol: float,
    incTol: float,
    maxIter: int,
) -> tuple[_types.FloatArray, list[float]]:
    """Newton-Raphson from x, `x += dx` until b's norm, its ratio to the first or the owned `|dx|` is under its tolerance; x and b's norms. Collective on `comm`."""
    x = x.copy()
    norms: list[float] = []
    for _ in range(maxIter):
        dx, rhsNorm = _Step(problem, system, backend, x, comm)
        x += dx
        norms.append(rhsNorm)
        relNorm = rhsNorm / norms[0] if norms[0] > 0 else 0.0
        if MPI_RANK == 0:
            print(f"At Newton iteration {len(norms)} relative norm is {relNorm:14.12e}")
        if rhsNorm < absTol or relNorm < relTol or system.Norm(dx, comm) < incTol:
            return x, norms
    raise NotConvergedError(
        f"Newton raphson algorithm did not converged in {len(norms)} iterations."
    )


def _Step(
    problem: Problem, system: System, backend: _Backend, x: _types.FloatArray, comm
) -> tuple[_types.FloatArray, float]:
    """dx of `A dx = b` at x, x + dx meeting the constraints, and b's norm. Collective on `comm`."""
    A, b = problem.Linearize(x)
    return system.Solve(
        A,
        b,
        x,
        problem.Constraints(x),
        backend,
        comm,
    )
