# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Iterable, Any

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

Number = Union[int, float]
NumberOrNone = Union[Number, None]

Numbers = Iterable[Number]

StrOrNone = Union[str, None]

# --------------------------------------------------------------------------------------
# Matplotlib
# --------------------------------------------------------------------------------------

Axes = Union[plt.Axes, Axes3D]
AxesOrNone = Union[Axes, None]

# --------------------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------------------

StrArray = NDArray[np.str_]

FloatArray = NDArray[np.float64]

IntArray = NDArray[np.int64]

NumberArray = Union[FloatArray, IntArray]

AnyArray = NDArray[Any]

AnyArrayOrNone = Union[AnyArray, None]

# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------

Coords = Union[NumberArray, Numbers]
CoordsOrNone = Union[Coords, None]
