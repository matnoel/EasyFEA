# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Models import Thermal


def test_Thermal_holds_k_c_and_thickness():
    model = Thermal(2.0, c=3.0, thickness=4.0)

    assert (model.k, model.c, model.thickness, model.dim) == (2.0, 3.0, 4.0, 1)


def test_Thermal_c_defaults_to_zero():
    assert Thermal(1.0).c == 0.0


def test_Thermal_rejects_negative_coefficients():
    with pytest.raises(AssertionError):
        Thermal(-1.0, 0.0)
    with pytest.raises(AssertionError):
        Thermal(1.0, -1.0)


def test_Thermal_isHeterogeneous_when_k_or_c_is_a_field():
    assert Thermal(1.0, 0.0).isHeterogeneous is False
    assert Thermal(np.ones(3), 0.0).isHeterogeneous is True
    assert Thermal(1.0, np.ones(3)).isHeterogeneous is True
