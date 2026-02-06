# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Collection, Any

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

Number = Union[int, float]
"""Number"""

Numbers = Collection[Number]
"""Numbers"""

# --------------------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------------------

FloatArray = NDArray[np.floating]
"""Float array"""

IntArray = NDArray[np.int_]
"""Int array"""

NumberArray = Union[FloatArray, IntArray]
"""Number array"""

AnyArray = NDArray[Any]
"""Any array"""

# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------

Coords = Union[AnyArray, Numbers]
"""Coords"""
