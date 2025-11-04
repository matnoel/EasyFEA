# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the cache_computed_values decorator."""

from functools import wraps
import copy

CACH_NAME = "__cachedComputedValues"


def cache_computed_values(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):

        __check_class(self)

        if not hasattr(self, CACH_NAME):
            setattr(self, CACH_NAME, {})

        key = (func.__name__, args, frozenset(kwargs.items()))
        # frozenset is used here to make kwargs arguments hashable.
        cachedComputedValues = getattr(self, CACH_NAME)

        if key not in cachedComputedValues:
            result = func(self, *args, **kwargs)
            cachedComputedValues[key] = result

        return copy.copy(cachedComputedValues[key])

    return wrapper


def __check_class(self):
    if not hasattr(self, "__class__"):
        raise TypeError(
            "The `@cache_computed_values` decorator only works for class methods."
        )


def clear_cached_computed_values(self):
    "Clears the cached computed values"
    __check_class(self)
    if hasattr(self, CACH_NAME):
        getattr(self, CACH_NAME).clear()
