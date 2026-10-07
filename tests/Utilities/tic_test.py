# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Tic


@pytest.mark.parametrize(
    "time, expected",
    [
        (5e-4, (500.0, "µs")),
        (1e-3, (1.0, "ms")),
        (1.0, (1.0, "s")),
        (60.0, (1.0, "m")),
        (3600.0, (1.0, "h")),
        (86400.0, (1.0, "j")),
    ],
)
def test_Get_time_unity_at_unit_boundaries(time: float, expected: tuple[float, str]):
    value, unity = Tic.Get_time_unity(time)
    assert unity == expected[1]
    assert value == pytest.approx(expected[0])
