# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import SolverType


def test_SolverType_prints_its_name():
    assert [str(solver) for solver in SolverType] == [s.value for s in SolverType]
