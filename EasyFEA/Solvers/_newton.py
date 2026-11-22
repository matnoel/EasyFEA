# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Nonlinear strategies over a `_Problem`, through its `System`: Newton-Raphson."""

from typing import Protocol

from ..Utilities import _params, _types
from ..Utilities._mpi import MPI_RANK
from ._backends import Backend
from ._problem_protocol import _Problem


class NotConvergedError(AssertionError):
    """Newton ran out of iterations."""


class NonlinearStrategy(Protocol):
    """Solves a `_Problem` from `x0`: x and b's norm per iteration. Collective on `problem.comm`."""

    def __call__(
        self, problem: _Problem, backend: Backend, x0: _types.FloatArray
    ) -> tuple[_types.FloatArray, list[float]]: ...


class Newton:
    """Newton-Raphson, `x += dx` until b's norm, its ratio to the first or the owned `|dx|` is under its tolerance."""

    absTol = _params.IntervalooScalarParameter(0, 1)
    """tolerance on b's norm"""

    relTol = _params.IntervalooScalarParameter(0, 1)
    """tolerance on b's norm over the first"""

    incTol = _params.IntervalooScalarParameter(0, 1)
    """tolerance on the owned `|dx|`"""

    maxIter = _params.IntAboveParameter(1)
    """iterations before `NotConvergedError`"""

    def __init__(
        self,
        absTol: float = 1.0e-6,
        relTol: float = 1.0e-10,
        incTol: float = 1.0e-11,
        maxIter: int = 20,
        verbosity: bool = True,
    ) -> None:
        self.absTol = absTol
        self.relTol = relTol
        self.incTol = incTol
        self.maxIter = maxIter
        self.verbosity = verbosity

    def __call__(
        self, problem: _Problem, backend: Backend, x0: _types.FloatArray
    ) -> tuple[_types.FloatArray, list[float]]:
        system, comm = problem.system, problem.comm
        x = x0.copy()
        norms: list[float] = []
        for _ in range(self.maxIter):
            A, b = problem.Get_A_b(x)
            dx, rhsNorm = system.Solve(
                A,
                b,
                x,
                problem.Constraints(x),
                backend,
                comm,
            )
            x += dx
            norms.append(rhsNorm)
            relNorm = rhsNorm / norms[0] if norms[0] > 0 else 0.0
            if self.verbosity and MPI_RANK == 0:
                print(
                    f"At Newton iteration {len(norms)} relative norm is {relNorm:14.12e}"
                )
            if (
                rhsNorm < self.absTol
                or relNorm < self.relTol
                or system.Norm(dx, comm) < self.incTol
            ):
                return x, norms
        raise NotConvergedError(f"Newton did not converge in {len(norms)} iterations.")
