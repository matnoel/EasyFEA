# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Iterable
import numpy as np

from . import _types


def CheckIsPositive(value: Union[_types.Number, _types.Numbers]) -> None:
    """Checks whether the value is positive"""
    errorText = "Must be > 0!"
    if isinstance(value, (int, float)):
        assert value > 0.0, errorText
    elif isinstance(value, Iterable):
        assert np.asarray(value).min() > 0.0, errorText
    else:
        raise TypeError("Unknown type.")


def CheckIsNegative(value: Union[int, float, Iterable]) -> None:
    """Checks whether the value is negative"""
    errorText = "Must be < 0!"
    if isinstance(value, (int, float)):
        assert value > 0.0, errorText
    elif isinstance(value, Iterable):
        assert np.asarray(value).min() > 0.0, errorText
    else:
        raise TypeError("Unknown type.")


def CheckIsInIntervalcc(value: Union[int, float, Iterable], inf, sup) -> None:
    """Checks whether the value is in ]inf, sup["""
    errorText = f"Must be in ]{inf}, {sup}["
    if isinstance(value, (int, float)):
        assert value > inf and value < sup, errorText
    elif isinstance(value, Iterable):
        values = np.asarray(value)
        assert values.min() > inf and values.max() < sup, errorText
    else:
        raise TypeError("Unknown type.")


def CheckIsInIntervaloo(value: Union[int, float, Iterable], inf, sup) -> None:
    """Checks whether the value is in [inf, sup]"""
    errorText = f"Must be in [{inf}, {sup}]"
    if isinstance(value, (int, float)):
        assert value >= inf and value <= sup, errorText
    elif isinstance(value, Iterable):
        values = np.asarray(value)
        assert values.min() >= inf and values.max() <= sup, errorText
    else:
        raise TypeError("Unknown type.")
