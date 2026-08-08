# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Kelvin-Mandel algebra on ``(..., 6)`` vectors ordered ``[xx, yy, zz, yz, xz, xy]``.

The √2 carried by the shear entries makes the double contraction ``A:B`` the plain dot product.

``Trace`` here takes a Kelvin vector, whereas ``FEM._linalg.Trace`` takes a ``(..., dim, dim)`` matrix — import this module by name so the call site stays unambiguous.
"""

import numpy as np

from ..FEM._linalg import FeArray

ONE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
"""Identity tensor ``I`` in Kelvin form."""

IDEV = np.eye(6) - np.outer(ONE, ONE) / 3.0
"""Fourth-order deviatoric projector ``I_dev`` (6, 6)."""


def Trace(A: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """``A_xx + A_yy + A_zz``."""
    return A[..., 0] + A[..., 1] + A[..., 2]


def Spherical(A: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Hydrostatic part ``(tr A / 3)·I``."""
    return Trace(A) / 3 * ONE


def Deviator(A: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    """Deviatoric part ``A − (tr A / 3)·I``."""
    return A - Spherical(A)
