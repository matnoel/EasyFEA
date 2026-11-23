# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


class ProblemType(str):
    """Type identifying a physical problem solved by a simulation."""


def _As_problemTypes(
    problemType: ProblemType | tuple[ProblemType, ...],
) -> tuple[ProblemType, ...]:
    """`problemType` as a tuple of problem types: one alone becomes a one-element tuple."""
    return problemType if isinstance(problemType, tuple) else (problemType,)
