# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import pytest

from EasyFEA.Simulations import _Simu

FlagsReassembly = Callable[[_Simu, Callable[[], None]], None]


def _assert_change_flags_reassembly(simu: _Simu, change: Callable[[], None]) -> None:
    for problemType in simu.Get_problemTypes():
        simu.Get_K_C_M_F(problemType)
    assert not simu.needUpdate

    change()

    assert simu.needUpdate


@pytest.fixture(scope="session")
def assert_change_flags_reassembly() -> FlagsReassembly:
    """Factory: `(simu, change)` assembles every problem of `simu`, then checks `change()` flags it for reassembly."""
    return _assert_change_flags_reassembly
