# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA.FEM._linalg import FeArray


def one_point_field(vec) -> FeArray.FeArrayALike:
    """A (1, 1, ...) field holding one value."""
    return FeArray.asfearray(np.asarray(vec, dtype=float)[np.newaxis, np.newaxis])


def point_value(field) -> np.ndarray:
    """The value a (1, 1, ...) field holds."""
    return np.asarray(field)[0, 0]
