# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Point element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class POINT(_GroupElem):
    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.empty((0), dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return self.surfaces

    def Get_Local_Coords(self):
        return np.array([0]).reshape(1, 1)

    def _N(self) -> _types.FloatArray:
        return None  # type: ignore [return-value]

    def _dN(self) -> _types.FloatArray:
        return None  # type: ignore [return-value]

    def _ddN(self) -> _types.FloatArray:
        return None  # type: ignore [return-value]

    def _dddN(self) -> _types.FloatArray:
        return None  # type: ignore [return-value]

    def _ddddN(self) -> _types.FloatArray:
        return None  # type: ignore [return-value]
