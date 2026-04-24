# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Seg element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class SEG2(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----+----1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordinates: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordinates)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 1]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.empty((0), dtype=int)

    def Get_Local_Coords(self):
        list_x = [-1, 1]
        local_coords = np.array([list_x]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r: -(r - 1) / 2
        N2 = lambda r: (r + 1) / 2

        N = np.array([N1, N2]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r: -1 / 2]
        dN2 = [lambda r: 1 / 2]

        dN = np.array([dN1, dN2]).reshape(-1, 1)

        return dN

    def _ddN(self) -> _types.FloatArray:
        return super()._ddN()

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class SEG3(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----2----1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordinates: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordinates)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 2, 1]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.empty((0), dtype=int)

    def Get_Local_Coords(self):
        list_x = [-1, 1, 0]
        local_coords = np.array([list_x]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r: r * (r - 1) / 2
        N2 = lambda r: r * (r + 1) / 2
        N3 = lambda r: -(r - 1) * (r + 1)

        N = np.array([N1, N2, N3]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r: r - 1 / 2]
        dN2 = [lambda r: r + 1 / 2]
        dN3 = [lambda r: -2 * r]

        dN = np.array([dN1, dN2, dN3]).reshape(-1, 1)

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r: 1]
        ddN2 = [lambda r: 1]
        ddN3 = [lambda r: -2]

        ddN = np.array([ddN1, ddN2, ddN3])

        return ddN

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class SEG4(_GroupElem):
    #       v
    #       ^
    #       |
    #       |
    # 0---2-+-3---1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordinates: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordinates)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 2, 3, 1]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.empty((0), dtype=int)

    def Get_Local_Coords(self):
        list_x = [-1, 1, -1 / 3, 1 / 3]
        local_coords = np.array([list_x]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r: -9 * r**3 / 16 + 9 * r**2 / 16 + r / 16 - 1 / 16
        N2 = lambda r: 9 * r**3 / 16 + 9 * r**2 / 16 - r / 16 - 1 / 16
        N3 = lambda r: 27 * r**3 / 16 - 9 * r**2 / 16 - 27 * r / 16 + 9 / 16
        N4 = lambda r: -27 * r**3 / 16 - 9 * r**2 / 16 + 27 * r / 16 + 9 / 16

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r: -27 * r**2 / 16 + 9 * r / 8 + 1 / 16]
        dN2 = [lambda r: 27 * r**2 / 16 + 9 * r / 8 - 1 / 16]
        dN3 = [lambda r: 81 * r**2 / 16 - 9 * r / 8 - 27 / 16]
        dN4 = [lambda r: -81 * r**2 / 16 - 9 * r / 8 + 27 / 16]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r: 9 / 8 - 27 * r / 8]
        ddN2 = [lambda r: 27 * r / 8 + 9 / 8]
        ddN3 = [lambda r: 81 * r / 8 - 9 / 8]
        ddN4 = [lambda r: -81 * r / 8 - 9 / 8]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4])

        return ddN

    def _dddN(self) -> _types.FloatArray:
        dddN1 = [lambda r: -27 / 8]
        dddN2 = [lambda r: 27 / 8]
        dddN3 = [lambda r: 81 / 8]
        dddN4 = [lambda r: -81 / 8]

        dddN = np.array([dddN1, dddN2, dddN3, dddN4])

        return dddN

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class SEG5(_GroupElem):
    #        v
    #        ^
    #        |
    #        |
    #  0--2--3--4--1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordinates: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordinates)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 2, 3, 4, 1]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.empty((0), dtype=int)

    def Get_Local_Coords(self):
        list_x = [-1, 1, -1 / 2, 0, 1 / 2]
        local_coords = np.array([list_x]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r: 2 * r**4 / 3 - 2 * r**3 / 3 - r**2 / 6 + r / 6
        N2 = lambda r: 2 * r**4 / 3 + 2 * r**3 / 3 - r**2 / 6 - r / 6
        N3 = lambda r: -8 * r**4 / 3 + 4 * r**3 / 3 + 8 * r**2 / 3 - 4 * r / 3
        N4 = lambda r: 4 * r**4 - 5 * r**2 + 1
        N5 = lambda r: -8 * r**4 / 3 - 4 * r**3 / 3 + 8 * r**2 / 3 + 4 * r / 3

        N = np.array([N1, N2, N3, N4, N5]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r: 8 * r**3 / 3 - 2 * r**2 - r / 3 + 1 / 6]
        dN2 = [lambda r: 8 * r**3 / 3 + 2 * r**2 - r / 3 - 1 / 6]
        dN3 = [lambda r: -32 * r**3 / 3 + 4 * r**2 + 16 * r / 3 - 4 / 3]
        dN4 = [lambda r: 16 * r**3 - 10 * r]
        dN5 = [lambda r: -32 * r**3 / 3 - 4 * r**2 + 16 * r / 3 + 4 / 3]

        dN = np.array([dN1, dN2, dN3, dN4, dN5])

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r: 8 * r**2 - 4 * r - 1 / 3]
        ddN2 = [lambda r: 8 * r**2 + 4 * r - 1 / 3]
        ddN3 = [lambda r: -32 * r**2 + 8 * r + 16 / 3]
        ddN4 = [lambda r: 48 * r**2 - 10]
        ddN5 = [lambda r: -32 * r**2 - 8 * r + 16 / 3]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5])

        return ddN

    def _dddN(self) -> _types.FloatArray:
        dddN1 = [lambda r: 16 * r - 4]
        dddN2 = [lambda r: 16 * r + 4]
        dddN3 = [lambda r: 8 - 64 * r]
        dddN4 = [lambda r: 96 * r]
        dddN5 = [lambda r: -64 * r - 8]

        dddN = np.array([dddN1, dddN2, dddN3, dddN4, dddN5])

        return dddN

    def _ddddN(self) -> _types.FloatArray:
        ddddN1 = [lambda r: 16]
        ddddN2 = [lambda r: 16]
        ddddN3 = [lambda r: -64]
        ddddN4 = [lambda r: 96]
        ddddN5 = [lambda r: -64]

        ddddN = np.array([ddddN1, ddddN2, ddddN3, ddddN4, ddddN5])

        return ddddN
