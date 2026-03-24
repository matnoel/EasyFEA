# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the cache_computed_values decorator."""

from functools import wraps
import numpy as np

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
            # Mark numpy arrays read-only so the cached buffer is protected
            # without duplicating it on every access (copy.copy was a full
            # data copy, doubling peak memory on every Assembly call).
            if isinstance(result, np.ndarray) and result.flags.writeable:
                result.flags.writeable = False
            cachedComputedValues[key] = result

        return cachedComputedValues[key]

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
