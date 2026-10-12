# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np


def rel_error(actual, expected, scale: float | None = None) -> float:
    """Largest |actual - expected| over `scale` (default: largest |expected|); pass `scale` where `expected` may vanish. Shapes must match unless `expected` is a scalar."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert (
        expected.ndim == 0 or actual.shape == expected.shape
    ), f"shape {actual.shape} != expected {expected.shape}"
    if scale is None:
        scale = float(np.abs(expected).max())
    return float(np.abs(actual - expected).max() / scale)
