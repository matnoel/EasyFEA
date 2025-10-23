# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
from typing import Union, Iterable, Callable
from functools import partial, partialmethod
import numpy as np
import copy

from . import _types


def _CheckIsBool(value: bool) -> None:
    """Checks whether the value is a boolean"""
    assert isinstance(value, bool), "Must be a boolean."


def _CheckIsScalar(value: Union[_types.Number, _types.Numbers]) -> None:
    """Checks whether the value is positive"""
    assert isinstance(value, (int, float)), "must be a scalar value"


def _CheckIsPositive(value: Union[_types.Number, _types.Numbers]) -> None:
    """Checks whether the value is positive"""
    errorText = "Must be >= 0!"
    if isinstance(value, (int, float)):
        assert value >= 0.0, errorText
    elif isinstance(value, Iterable):
        assert np.all(value >= 0.0), errorText
    else:
        raise TypeError("Unknown type.")


def _CheckIsNegative(value: Union[int, float, Iterable]) -> None:
    """Checks whether the value is negative"""
    errorText = "Must be <= 0!"
    if isinstance(value, (int, float)):
        assert value <= 0.0, errorText
    elif isinstance(value, Iterable):
        assert np.all(value <= 0.0), errorText
    else:
        raise TypeError("Unknown type.")


def _CheckIsInIntervalcc(value: Union[int, float, Iterable], inf, sup) -> None:
    """Checks whether the value is in ]inf, sup["""
    assert inf < sup
    errorText = f"Must be in ]{inf}, {sup}["
    if isinstance(value, (int, float)):
        assert inf < value < sup, errorText
    elif isinstance(value, Iterable):
        values = np.asarray(value)
        tests = (inf < values) & (values < sup)
        assert np.all(tests), errorText
    else:
        raise TypeError("Unknown type.")


def _CheckIsInIntervaloo(value: Union[int, float, Iterable], inf, sup) -> None:
    """Checks whether the value is in [inf, sup]"""
    assert inf < sup
    errorText = f"Must be in [{inf}, {sup}]"
    if isinstance(value, (int, float)):
        assert inf <= value <= sup, errorText
    elif isinstance(value, Iterable):
        values = np.asarray(value)
        tests = (inf <= values) & (values <= sup)
        assert np.all(tests), errorText
    else:
        raise TypeError("Unknown type.")


def _CheckIsInValues(value, values: Iterable) -> None:
    """Checks whether the value is in [inf, sup]"""
    errorText = f"{value} Must be in {values}"
    assert value in values, errorText


class Updatable(ABC):

    @abstractmethod
    def Need_Update(self, value=True) -> None:
        """Indicates whether the object needs to be updated."""
        self.__needUpdate = value

    @property
    def needUpdate(self) -> bool:
        """The object needs to be updated."""
        try:
            return self.__needUpdate
        except AttributeError:
            self.__needUpdate = True
            return self.__needUpdate


class Parameter:

    def __set_name__(self, owner, name):
        self.__name = name

    def __get__(self, instance, owner):
        return copy.copy(instance.__dict__[self.__name])

    def __init__(self, check_functions: list[Callable] = []):
        error = "check_functions must be a list of function."
        assert isinstance(check_functions, Iterable), error
        for function in check_functions:
            # fmt: off
            assert (
                isinstance(function, Callable) or
                isinstance(function, partialmethod)
            ), error
            # fmt: on
        self.__check_functions = check_functions

    def __set__(self, instance, value):
        for function in self.__check_functions:
            if isinstance(function, partialmethod):
                function = function.__get__(instance, type(instance))
                function(value)
            else:
                function(value)
        instance.__dict__[self.__name] = value
        if isinstance(instance, Updatable):
            instance.Need_Update()


class BoolParameter(Parameter):
    def __init__(self):
        super().__init__(check_functions=[_CheckIsBool])


class PositiveParameter(Parameter):
    def __init__(self):
        super().__init__(check_functions=[_CheckIsPositive])


class PositiveScalarParameter(Parameter):
    def __init__(self):
        super().__init__(
            check_functions=[partial(_CheckIsScalar), partial(_CheckIsPositive)]
        )


class NegativeParameter(Parameter):
    def __init__(self):
        super().__init__(check_functions=[_CheckIsNegative])


class ParameterInValues(Parameter):
    def __init__(self, values):
        super().__init__(check_functions=[partial(_CheckIsInValues, values=values)])


class IntervalccParameter(Parameter):
    def __init__(self, inf: float, sup: float):
        super().__init__(
            check_functions=[partial(_CheckIsInIntervalcc, inf=inf, sup=sup)]
        )


class IntervalooParameter(Parameter):
    def __init__(self, inf: float, sup: float):
        super().__init__(
            check_functions=[partial(_CheckIsInIntervaloo, inf=inf, sup=sup)]
        )
