# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Iterable, Any

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore [import-untyped]

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

Number = Union[int, float]

Numbers = Iterable[Number]


# --------------------------------------------------------------------------------------
# Matplotlib
# --------------------------------------------------------------------------------------

Axes = Union[plt.Axes, Axes3D]

# --------------------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------------------

StrArray = NDArray[np.str_]

FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]

NumberArray = Union[FloatArray, IntArray]

Array = NDArray[Any]

# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------

Coords = Union[NumberArray, Numbers]
