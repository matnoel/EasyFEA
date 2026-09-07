# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Helpers shared by the operator modules: the contraction they all integrate with, and the element restriction they all honour."""

from typing import Optional

import numpy as np

from ...Utilities import _types


def einsum(*args) -> np.ndarray:
    """``np.einsum`` with path optimization, returned as a plain array."""
    return np.asarray(np.einsum(*args, optimize=True))


def Restrict(elements: Optional[_types.IntArray], *arrays: np.ndarray):
    """Restricts per-element arrays to `elements`, or returns them untouched when it is None.

    Every array argument of an operator carries **full-group** shape — row ``i`` is element ``i`` of this group — so an index array is only ever meaningful against the group being integrated. That is what makes ``elements`` safe when several groups share a dimension (PRISM18 + HEXA27): the indices are resolved against the one group being integrated, never hoisted across groups.
    """
    if elements is None:
        return arrays if len(arrays) > 1 else arrays[0]
    # a scalar coefficient is uniform across elements — `FeArray.broadcast` leaves it a float — so it has no element axis to restrict
    restricted = tuple(
        array if np.ndim(array) == 0 else array[elements] for array in arrays
    )
    return restricted if len(restricted) > 1 else restricted[0]


def Scatter(
    values_e: np.ndarray, Ne: int, elements: Optional[_types.IntArray]
) -> np.ndarray:
    """Places `values_e` at `elements` inside a full-group array of zeros, so a restricted operator still scatters uniformly through the group's connectivity."""
    if elements is None:
        return values_e
    out = np.zeros((Ne, *values_e.shape[1:]), dtype=values_e.dtype)
    out[elements] = values_e
    return out
