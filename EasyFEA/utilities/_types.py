# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Collection, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

Number = Union[int, float]

Numbers = Collection[Number]


# --------------------------------------------------------------------------------------
# Matplotlib
# --------------------------------------------------------------------------------------

Axes = Union[plt.Axes, Axes3D]  # type: ignore

# --------------------------------------------------------------------------------------
# Array
# --------------------------------------------------------------------------------------

FloatArray = NDArray[np.floating]

IntType = TypeVar(
    "IntType",
    bound=Union[np.integer, int],
)
IntArray = NDArray[IntType]  # type: ignore

NumberArray = Union[FloatArray, IntArray]

AnyArray = NDArray[Any]

# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------

Coords = Union[AnyArray, Numbers]
