# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Mesh-free linear algebra: backends, constraints and the system `A x = b`."""

from ._backends import SolverType, Get_backend
from ._constraints import ResolType, Constraints, Lagrange
from ._system import System
from ._drivers import Problem, Linear, Newton, NotConvergedError
