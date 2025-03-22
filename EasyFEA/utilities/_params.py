# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union
from collections.abc import Iterable
import numpy as np

def CheckIsPositive(value: Union[float, int, Iterable]) -> None:
    """Checks whether the value is positive"""
    errorText = "Must be > 0!"
    if isinstance(value, (float, int)):
        assert value > 0.0, errorText
    elif isinstance(value,  Iterable):
        assert np.min(value) > 0.0, errorText
    else:
        raise TypeError("Unknown type.")

def CheckIsNegative(value: Union[float, int, Iterable]) -> None:
    """Checks whether the value is negative"""
    errorText = "Must be < 0!"
    if isinstance(value, (float, int)):
        assert value < 0.0, errorText
    elif isinstance(value,  Iterable):
        assert np.max(value) < 0.0, errorText
    else:
        raise TypeError("Unknown type.")

def CheckIsInIntervalcc(value: Union[float, int, Iterable], inf, sup) -> None:
    """Checks whether the value is in ]inf, sup["""
    errorText = f"Must be in ]{inf}, {sup}["
    if isinstance(value, (float, int)):
        assert value > inf and value < sup, errorText
    elif isinstance(value,  Iterable):
        assert np.min(value) > inf and np.max(value) < sup, errorText
    else:
        raise TypeError("Unknown type.")
    
def CheckIsInIntervaloo(value: Union[float, int, Iterable], inf, sup) -> None:
    """Checks whether the value is in [inf, sup]"""
    errorText = f"Must be in [{inf}, {sup}]"
    if isinstance(value, (float, int)):
        assert value >= inf and value <= sup, errorText
    elif isinstance(value, Iterable):
        assert np.min(value) >= inf and np.max(value) <= sup, errorText
    else:
        raise TypeError("Unknown type.")