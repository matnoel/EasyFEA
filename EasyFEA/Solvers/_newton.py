# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Newton-Raphson over callables, through a `System`."""

from typing import Callable

from scipy import sparse

from ..Utilities import _types
from ..Utilities._mpi import MPI_RANK
from ._backends import _Backend
from ._constraints import Constraints
from ._system import System


class NotConvergedError(AssertionError):
    """Newton ran out of iterations."""


def Newton(
    Get_A_b: Callable[[_types.FloatArray], tuple[sparse.csr_matrix, _types.FloatArray]],
    Get_constraints: Callable[[_types.FloatArray], Constraints],
    system: System,
    backend: _Backend,
    x: _types.FloatArray,
    comm,
    tol: tuple[float, float, float, int],
) -> tuple[_types.FloatArray, list[float]]:
    """Newton-Raphson from x, `x += dx` until b's norm, its ratio to the first or the owned `|dx|` is under its tolerance, `tol = (absTol, relTol, incTol, maxIter)`; x and b's norms. Collective on `comm`."""
    absTol, relTol, incTol, maxIter = tol
    x = x.copy()
    norms: list[float] = []
    for _ in range(maxIter):
        A, b = Get_A_b(x)
        dx, rhsNorm = system.Solve(
            A,
            b,
            x,
            Get_constraints(x),
            backend,
            comm,
        )
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
