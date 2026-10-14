# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.Simulations import ProblemType


def test_ProblemType_is_the_string_it_names():
    problemType = ProblemType("elastic")

    assert isinstance(problemType, str)
    assert problemType == "elastic"
