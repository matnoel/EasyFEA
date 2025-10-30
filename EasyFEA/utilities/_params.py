# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
from typing import Union, Iterable, Callable
from functools import partialmethod
import numpy as np
import copy

from . import _types


def _CheckIsBool(value: bool) -> None:
    """Checks whether the value is a boolean"""
    assert isinstance(value, bool), "Must be a boolean."


def _CheckIsString(value: str) -> None:
    """Checks whether the value is a string"""
    assert isinstance(value, str), "Must be a string."


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


def _CheckIsVector(value) -> None:
    """Checks whether the value is a (..., 3) numpy array."""
    errorText = "value must be a (..., 3) array"
    assert isinstance(value, np.ndarray) and value.shape[-1] == 3, errorText


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


class _Parameter:

    def __set_name__(self, owner, name):
        self.__name = name

    def __get__(self, instance, owner):
        return copy.copy(instance.__dict__[self.__name])

    def __set__(self, instance, value):
        self._checker(value)
        instance.__dict__[self.__name] = value
        if isinstance(instance, Updatable):
            instance.Need_Update()

    def _checker(self, value):
        raise NotImplementedError(f"Checker not implemented on {type(self)}")


class BoolParameter(_Parameter):
    def _checker(self, value):
        _CheckIsBool(value)


class StringParameter(_Parameter):
    def _checker(self, value):
        _CheckIsString(value)


class ScalarParameter(_Parameter):
    def _checker(self, value):
        _CheckIsScalar(value)


class PositiveParameter(_Parameter):
    def _checker(self, value):
        _CheckIsPositive(value)


class PositiveScalarParameter(_Parameter):
    def _checker(self, value):
        _CheckIsScalar(value)
        _CheckIsPositive(value)


class NegativeParameter(_Parameter):
    def _checker(self, value):
        _CheckIsNegative(value)


class ParameterInValues(_Parameter):
    def __init__(self, values):
        self.__values = values

    def _checker(self, value):
        _CheckIsInValues(value, self.__values)


class IntervalccParameter(_Parameter):
    def __init__(self, inf: float, sup: float):
        self.__inf = inf
        self.__sup = sup

    def _checker(self, value):
        _CheckIsInIntervalcc(value, inf=self.__inf, sup=self.__sup)


class IntervalooParameter(_Parameter):
    def __init__(self, inf: float, sup: float):
        self.__inf = inf
        self.__sup = sup

    def _checker(self, value):
        _CheckIsInIntervaloo(value, inf=self.__inf, sup=self.__sup)


class VectorParameter(_Parameter):
    def _checker(self, value):
        _CheckIsVector(value)


class InstanceParameter(_Parameter):

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
        self.__instance = instance
        return super().__set__(instance, value)

    def _checker(self, value):
        instance = self.__instance
        for function in self.__check_functions:
            if isinstance(function, partialmethod):
                function = function.__get__(instance, type(instance))
                function(value)
            else:
                function(value)
