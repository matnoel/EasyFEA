# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Fields of a simulation solved as one system: problem types, terms, DOF layout, assembly."""

from ._problem_type import ProblemType
from ._terms import Term, Integrate_terms
from ._problem import Problem
