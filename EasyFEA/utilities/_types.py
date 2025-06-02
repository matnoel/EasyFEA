# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from typing import Union, Iterable, Any
import numpy as np

Number = Union[int, float]
Numbers = Iterable[Number]

ArrayOrNone = Union[np.ndarray[Any, np.dtype[Any]], None]
