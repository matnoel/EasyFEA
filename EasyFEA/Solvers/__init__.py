# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Mesh-free linear algebra: backends, constraints, the system `A x = b` and nonlinear strategies."""

from ._backends import SolverType, Backend
from ._constraints import ResolType, Constraints, LagrangeConstraint
from ._system import System
from ._newton import (
    Newton,
    NonlinearProblem,
    NonlinearStrategy,
    NotConvergedError,
)
