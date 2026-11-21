# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import SolverType
from EasyFEA.Solvers import Newton


def test_SolverType_prints_its_name():
    assert [str(solver) for solver in SolverType] == [s.value for s in SolverType]


def test_Newton_checks_its_parameters():
    with pytest.raises(AssertionError, match="> 1"):
        Newton(maxIter=1)
    with pytest.raises(AssertionError, match=r"\]0, 1\["):
        Newton(absTol=2.0)
