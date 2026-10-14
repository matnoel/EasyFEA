# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Utilities import (
    Updatable,
    _CheckIsPositive,
    _CheckIsNegative,
    _CheckIsInIntervalcc,
    _CheckIsInIntervaloo,
)


def test_CheckIsPositive_accepts_non_negative_values(subtests):
    for value in [0, 2.0, [0.0, 1.0], np.array([3.0])]:
        with subtests.test(value=value):
            _CheckIsPositive(value)


def test_CheckIsPositive_rejects_a_negative_value(subtests):
    for value in [-1.0, [1.0, -1.0]]:
        with subtests.test(value=value):
            with pytest.raises(AssertionError):
                _CheckIsPositive(value)


def test_CheckIsPositive_rejects_a_non_numeric_type():
    with pytest.raises(TypeError):
        _CheckIsPositive(None)  # type: ignore[arg-type]


def test_CheckIsNegative_accepts_non_positive_values(subtests):
    for value in [0, -2.0, [0.0, -1.0], np.array([-3.0])]:
        with subtests.test(value=value):
            _CheckIsNegative(value)


def test_CheckIsNegative_rejects_a_positive_value(subtests):
    for value in [1.0, [-1.0, 1.0]]:
        with subtests.test(value=value):
            with pytest.raises(AssertionError):
                _CheckIsNegative(value)


def test_CheckIsNegative_rejects_a_non_numeric_type():
    with pytest.raises(TypeError):
        _CheckIsNegative(None)  # type: ignore[arg-type]


def test_CheckIsInIntervaloo_accepts_inner_values(subtests):
    for value in [0.5, [0.1, 0.9]]:
        with subtests.test(value=value):
            _CheckIsInIntervaloo(value, 0, 1)


def test_CheckIsInIntervaloo_rejects_its_bounds(subtests):
    for value in [0.0, 1.0, [0.5, 1.0]]:
        with subtests.test(value=value):
            with pytest.raises(AssertionError):
                _CheckIsInIntervaloo(value, 0, 1)


def test_CheckIsInIntervaloo_rejects_a_non_numeric_type():
    with pytest.raises(TypeError):
        _CheckIsInIntervaloo(None, 0, 1)  # type: ignore[arg-type]


def test_CheckIsInIntervalcc_accepts_its_bounds(subtests):
    for value in [0.0, 1.0, [0.0, 0.5, 1.0]]:
        with subtests.test(value=value):
            _CheckIsInIntervalcc(value, 0, 1)


def test_CheckIsInIntervalcc_rejects_outer_values(subtests):
    for value in [-0.1, [0.5, 1.1]]:
        with subtests.test(value=value):
            with pytest.raises(AssertionError):
                _CheckIsInIntervalcc(value, 0, 1)


def test_CheckIsInIntervalcc_rejects_a_non_numeric_type():
    with pytest.raises(TypeError):
        _CheckIsInIntervalcc(None, 0, 1)  # type: ignore[arg-type]


class Model(Updatable):
    def Need_Update(self, value=True) -> None:
        super().Need_Update(value)


def test_Updatable_cannot_be_instantiated_without_Need_Update():
    with pytest.raises(TypeError):
        Updatable()  # type: ignore[abstract]


def test_Updatable_Need_Update_sets_needUpdate():
    model = Model()
    model.Need_Update(False)
    assert model.needUpdate is False
    model.Need_Update()
    assert model.needUpdate is True


def test_Updatable_needUpdate_is_true_on_a_new_object():
    assert Model().needUpdate is True
