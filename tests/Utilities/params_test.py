# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.Utilities import _params


def test_closed_interval_accepts_its_bounds():
    _params._CheckIsInIntervalcc(0.0, 0, 1)
    _params._CheckIsInIntervalcc(1.0, 0, 1)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_open_interval_rejects_its_bounds(value: float):
    with pytest.raises(AssertionError):
        _params._CheckIsInIntervaloo(value, 0, 1)
