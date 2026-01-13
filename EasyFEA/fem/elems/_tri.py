# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Tri element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class TRI3(_GroupElem):
    # v
    # ^
    # |
    # 2
    # |`\
    # |  `\
    # |    `\
    # |      `\
    # |        `\
    # 0----------1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0, 1, 2]

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 1, 2, 0]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.arange(self.nPe, dtype=int)

    def Get_Local_Coords(self):
        list_x = [0, 1, 0]
        list_y = [0, 0, 1]
        local_coords = np.array([list_x, list_y]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s: -r - s + 1
        N2 = lambda r, s: r
        N3 = lambda r, s: s

        N = np.array([N1, N2, N3]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r, s: -1, lambda r, s: -1]
        dN2 = [lambda r, s: 1, lambda r, s: 0]
        dN3 = [lambda r, s: 0, lambda r, s: 1]

        dN = np.array([dN1, dN2, dN3])

        return dN

    def _ddN(self) -> _types.FloatArray:
        return super()._ddN()

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class TRI6(_GroupElem):
    # v
    # ^
    # |
    # 2
    # |`\
    # |  `\
    # 5    `4
    # |      `\
    # |        `\
    # 0----3-----1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        # fmt: off
        return [0,3,5,
                3,1,4,
                5,4,2,
                3,4,5]
        # fmt: on

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 3, 1, 4, 2, 5, 0]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.arange(self.nPe, dtype=int)

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0.5, 0.5, 0]
        list_y = [0, 0, 1, 0, 0.5, 0.5]
        local_coords = np.array([list_x, list_y]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s: (r + s - 1) * (2 * r + 2 * s - 1)
        N2 = lambda r, s: r * (2 * r - 1)
        N3 = lambda r, s: s * (2 * s - 1)
        N4 = lambda r, s: -4 * r * (r + s - 1)
        N5 = lambda r, s: 4 * r * s
        N6 = lambda r, s: -4 * s * (r + s - 1)

        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r, s: 4 * r + 4 * s - 3, lambda r, s: 4 * r + 4 * s - 3]
        dN2 = [lambda r, s: 4 * r - 1, lambda r, s: 0]
        dN3 = [lambda r, s: 0, lambda r, s: 4 * s - 1]
        dN4 = [lambda r, s: -8 * r - 4 * s + 4, lambda r, s: -4 * r]
        dN5 = [lambda r, s: 4 * s, lambda r, s: 4 * r]
        dN6 = [lambda r, s: -4 * s, lambda r, s: -4 * r - 8 * s + 4]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r, s: 4, lambda r, s: 4]
        ddN2 = [lambda r, s: 4, lambda r, s: 0]
        ddN3 = [lambda r, s: 0, lambda r, s: 4]
        ddN4 = [lambda r, s: -8, lambda r, s: 0]
        ddN5 = [lambda r, s: 0, lambda r, s: 0]
        ddN6 = [lambda r, s: 0, lambda r, s: -8]

        ddNtild = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])

        return ddNtild

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class TRI10(_GroupElem):
    # v
    # ^
    # |
    # 2
    # | \
    # 7   6
    # |     \
    # 8  (9)  5
    # |         \
    # 0---3---4---1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        # fmt: off
        return [0,3,8,
                3,4,9,
                4,1,5,
                4,5,9,
                3,9,8,
                8,9,7,
                9,5,6,
                9,6,7,
                7,6,2]
        # fmt: on

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 3, 4, 1, 5, 6, 2, 7, 8, 0]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.arange(self.nPe, dtype=int)

    def Get_Local_Coords(self):
        # fmt: off
        list_x = [
            0,1,0,
            1/3,2/3,
            2/3, 1/3,
            0,0,
            1/3
        ]
        list_y = [
            0,0,1,
            0,0,
            1/3, 2/3,
            2/3,1/3,
            1/3
        ]
        # fmt: off
        local_coords = np.array([list_x, list_y]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = (
            lambda r, s: -9 * r**3 / 2
            - 27 * r**2 * s / 2
            + 9 * r**2
            - 27 * r * s**2 / 2
            + 18 * r * s
            - 11 * r / 2
            - 9 * s**3 / 2
            + 9 * s**2
            - 11 * s / 2
            + 1
        )
        N2 = lambda r, s: 9 * r**3 / 2 - 9 * r**2 / 2 + r
        N3 = lambda r, s: 9 * s**3 / 2 - 9 * s**2 / 2 + s
        N4 = (
            lambda r, s: 27 * r**3 / 2
            + 27 * r**2 * s
            - 45 * r**2 / 2
            + 27 * r * s**2 / 2
            - 45 * r * s / 2
            + 9 * r
        )
        N5 = (
            lambda r, s: -27 * r**3 / 2
            - 27 * r**2 * s / 2
            + 18 * r**2
            + 9 * r * s / 2
            - 9 * r / 2
        )
        N6 = lambda r, s: 27 * r**2 * s / 2 - 9 * r * s / 2
        N7 = lambda r, s: 27 * r * s**2 / 2 - 9 * r * s / 2
        N8 = (
            lambda r, s: -27 * r * s**2 / 2
            + 9 * r * s / 2
            - 27 * s**3 / 2
            + 18 * s**2
            - 9 * s / 2
        )
        N9 = (
            lambda r, s: 27 * r**2 * s / 2
            + 27 * r * s**2
            - 45 * r * s / 2
            + 27 * s**3 / 2
            - 45 * s**2 / 2
            + 9 * s
        )
        N10 = lambda r, s: -27 * r**2 * s - 27 * r * s**2 + 27 * r * s

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s: -27 * r**2 / 2
            - 27 * r * s
            + 18 * r
            - 27 * s**2 / 2
            + 18 * s
            - 11 / 2,
            lambda r, s: -27 * r**2 / 2
            - 27 * r * s
            + 18 * r
            - 27 * s**2 / 2
            + 18 * s
            - 11 / 2,
        ]
        dN2 = [lambda r, s: 27 * r**2 / 2 - 9 * r + 1, lambda r, s: 0]
        dN3 = [lambda r, s: 0, lambda r, s: 27 * s**2 / 2 - 9 * s + 1]
        dN4 = [
            lambda r, s: 81 * r**2 / 2
            + 54 * r * s
            - 45 * r
            + 27 * s**2 / 2
            - 45 * s / 2
            + 9,
            lambda r, s: 27 * r**2 + 27 * r * s - 45 * r / 2,
        ]
        dN5 = [
            lambda r, s: -81 * r**2 / 2 - 27 * r * s + 36 * r + 9 * s / 2 - 9 / 2,
            lambda r, s: -27 * r**2 / 2 + 9 * r / 2,
        ]
        dN6 = [
            lambda r, s: 27 * r * s - 9 * s / 2,
            lambda r, s: 27 * r**2 / 2 - 9 * r / 2,
        ]
        dN7 = [
            lambda r, s: 27 * s**2 / 2 - 9 * s / 2,
            lambda r, s: 27 * r * s - 9 * r / 2,
        ]
        dN8 = [
            lambda r, s: -27 * s**2 / 2 + 9 * s / 2,
            lambda r, s: -27 * r * s + 9 * r / 2 - 81 * s**2 / 2 + 36 * s - 9 / 2,
        ]
        dN9 = [
            lambda r, s: 27 * r * s + 27 * s**2 - 45 * s / 2,
            lambda r, s: 27 * r**2 / 2
            + 54 * r * s
            - 45 * r / 2
            + 81 * s**2 / 2
            - 45 * s
            + 9,
        ]
        dN10 = [
            lambda r, s: -54 * r * s - 27 * s**2 + 27 * s,
            lambda r, s: -27 * r**2 - 54 * r * s + 27 * r,
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r, s: -27 * r - 27 * s + 18, lambda r, s: -27 * r - 27 * s + 18]
        ddN2 = [lambda r, s: 27 * r - 9, lambda r, s: 0]
        ddN3 = [lambda r, s: 0, lambda r, s: 27 * s - 9]
        ddN4 = [lambda r, s: 81 * r + 54 * s - 45, lambda r, s: 27 * r]
        ddN5 = [lambda r, s: -81 * r - 27 * s + 36, lambda r, s: 0]
        ddN6 = [lambda r, s: 27 * s, lambda r, s: 0]
        ddN7 = [lambda r, s: 0, lambda r, s: 27 * r]
        ddN8 = [lambda r, s: 0, lambda r, s: -27 * r - 81 * s + 36]
        ddN9 = [lambda r, s: 27 * s, lambda r, s: 54 * r + 81 * s - 45]
        ddN10 = [lambda r, s: -54 * s, lambda r, s: -54 * r]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN

    def _dddN(self) -> _types.FloatArray:
        dddN1 = [lambda r, s: -27, lambda r, s: -27]
        dddN2 = [lambda r, s: 27, lambda r, s: 0]
        dddN3 = [lambda r, s: 0, lambda r, s: 27]
        dddN4 = [lambda r, s: 81, lambda r, s: 0]
        dddN5 = [lambda r, s: -81, lambda r, s: 0]
        dddN6 = [lambda r, s: 0, lambda r, s: 0]
        dddN7 = [lambda r, s: 0, lambda r, s: 0]
        dddN8 = [lambda r, s: 0, lambda r, s: -81]
        dddN9 = [lambda r, s: 0, lambda r, s: 81]
        dddN10 = [lambda r, s: 0, lambda r, s: 0]

        dddN = np.array(
            [dddN1, dddN2, dddN3, dddN4, dddN5, dddN6, dddN7, dddN8, dddN9, dddN10]
        )

        return dddN

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class TRI15(_GroupElem):
    # v
    # 2
    # | \
    # 9   8
    # |     \
    # 10 (14)  7
    # |         \
    # 11 (12) (13) 6
    # |             \
    # 0---3---4---5---1

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        # fmt: off
        return [0,3,11,
                3,4,12,
                4,5,13,
                5,1,6,
                5,6,13,
                4,13,12,
                3,12,11,
                11,12,10,
                12,13,14,
                13,6,7,
                13,7,14,
                12,14,10,
                10,14,9,
                14,7,8,
                14,8,9,
                10,14,9,
                9,8,2]
        # fmt: on

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array([[0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11, 0]], dtype=int)

    @property
    def faces(self) -> _types.IntArray:
        return np.arange(self.nPe, dtype=int)

    def Get_Local_Coords(self):
        # fmt: off
        list_x = [
            0,1,0,
            1/4,1/2,3/4,
            3/4,1/2,1/4,
            0,0,0,
            1/4,1/2,1/4
        ]
        list_y = [
            0,0,1,
            0,0,0,
            1/4,1/2,3/4,
            3/4,1/2,1/4,
            1/4,1/4,1/2
        ]
        # fmt: on
        local_coords = np.array([list_x, list_y]).T
        return local_coords

    def _N(self) -> _types.FloatArray:
        N1 = (
            lambda r, s: (r + s - 1)
            * (2 * r + 2 * s - 1)
            * (4 * r + 4 * s - 3)
            * (4 * r + 4 * s - 1)
            / 3
        )
        N2 = lambda r, s: r * (2 * r - 1) * (4 * r - 3) * (4 * r - 1) / 3
        N3 = lambda r, s: s * (2 * s - 1) * (4 * s - 3) * (4 * s - 1) / 3
        N4 = (
            lambda r, s: -16
            * r
            * (r + s - 1)
            * (2 * r + 2 * s - 1)
            * (4 * r + 4 * s - 3)
            / 3
        )
        N5 = lambda r, s: 4 * r * (4 * r - 1) * (r + s - 1) * (4 * r + 4 * s - 3)
        N6 = lambda r, s: -16 * r * (2 * r - 1) * (4 * r - 1) * (r + s - 1) / 3
        N7 = lambda r, s: 16 * r * s * (2 * r - 1) * (4 * r - 1) / 3
        N8 = lambda r, s: 4 * r * s * (4 * r - 1) * (4 * s - 1)
        N9 = lambda r, s: 16 * r * s * (2 * s - 1) * (4 * s - 1) / 3
        N10 = lambda r, s: -16 * s * (2 * s - 1) * (4 * s - 1) * (r + s - 1) / 3
        N11 = lambda r, s: 4 * s * (4 * s - 1) * (r + s - 1) * (4 * r + 4 * s - 3)
        N12 = (
            lambda r, s: -16
            * s
            * (r + s - 1)
            * (2 * r + 2 * s - 1)
            * (4 * r + 4 * s - 3)
            / 3
        )
        N13 = lambda r, s: 32 * r * s * (r + s - 1) * (4 * r + 4 * s - 3)
        N14 = lambda r, s: -32 * r * s * (4 * r - 1) * (r + s - 1)
        N15 = lambda r, s: -32 * r * s * (4 * s - 1) * (r + s - 1)

        N = np.array(
            [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15]
        ).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s: 128 * r**3 / 3
            + 128 * r**2 * s
            - 80 * r**2
            + 128 * r * s**2
            - 160 * r * s
            + 140 * r / 3
            + 128 * s**3 / 3
            - 80 * s**2
            + 140 * s / 3
            - 25 / 3,
            lambda r, s: 128 * r**3 / 3
            + 128 * r**2 * s
            - 80 * r**2
            + 128 * r * s**2
            - 160 * r * s
            + 140 * r / 3
            + 128 * s**3 / 3
            - 80 * s**2
            + 140 * s / 3
            - 25 / 3,
        ]
        dN2 = [lambda r, s: 128 * r**3 / 3 - 48 * r**2 + 44 * r / 3 - 1, lambda r, s: 0]
        dN3 = [lambda r, s: 0, lambda r, s: 128 * s**3 / 3 - 48 * s**2 + 44 * s / 3 - 1]
        dN4 = [
            lambda r, s: -512 * r**3 / 3
            - 384 * r**2 * s
            + 288 * r**2
            - 256 * r * s**2
            + 384 * r * s
            - 416 * r / 3
            - 128 * s**3 / 3
            + 96 * s**2
            - 208 * s / 3
            + 16,
            lambda r, s: -128 * r**3
            - 256 * r**2 * s
            + 192 * r**2
            - 128 * r * s**2
            + 192 * r * s
            - 208 * r / 3,
        ]
        dN5 = [
            lambda r, s: 256 * r**3
            + 384 * r**2 * s
            - 384 * r**2
            + 128 * r * s**2
            - 288 * r * s
            + 152 * r
            - 16 * s**2
            + 28 * s
            - 12,
            lambda r, s: 128 * r**3 + 128 * r**2 * s - 144 * r**2 - 32 * r * s + 28 * r,
        ]
        dN6 = [
            lambda r, s: -512 * r**3 / 3
            - 128 * r**2 * s
            + 224 * r**2
            + 64 * r * s
            - 224 * r / 3
            - 16 * s / 3
            + 16 / 3,
            lambda r, s: -128 * r**3 / 3 + 32 * r**2 - 16 * r / 3,
        ]
        dN7 = [
            lambda r, s: 128 * r**2 * s - 64 * r * s + 16 * s / 3,
            lambda r, s: 128 * r**3 / 3 - 32 * r**2 + 16 * r / 3,
        ]
        dN8 = [
            lambda r, s: 128 * r * s**2 - 32 * r * s - 16 * s**2 + 4 * s,
            lambda r, s: 128 * r**2 * s - 16 * r**2 - 32 * r * s + 4 * r,
        ]
        dN9 = [
            lambda r, s: 128 * s**3 / 3 - 32 * s**2 + 16 * s / 3,
            lambda r, s: 128 * r * s**2 - 64 * r * s + 16 * r / 3,
        ]
        dN10 = [
            lambda r, s: -128 * s**3 / 3 + 32 * s**2 - 16 * s / 3,
            lambda r, s: -128 * r * s**2
            + 64 * r * s
            - 16 * r / 3
            - 512 * s**3 / 3
            + 224 * s**2
            - 224 * s / 3
            + 16 / 3,
        ]
        dN11 = [
            lambda r, s: 128 * r * s**2 - 32 * r * s + 128 * s**3 - 144 * s**2 + 28 * s,
            lambda r, s: 128 * r**2 * s
            - 16 * r**2
            + 384 * r * s**2
            - 288 * r * s
            + 28 * r
            + 256 * s**3
            - 384 * s**2
            + 152 * s
            - 12,
        ]
        dN12 = [
            lambda r, s: -128 * r**2 * s
            - 256 * r * s**2
            + 192 * r * s
            - 128 * s**3
            + 192 * s**2
            - 208 * s / 3,
            lambda r, s: -128 * r**3 / 3
            - 256 * r**2 * s
            + 96 * r**2
            - 384 * r * s**2
            + 384 * r * s
            - 208 * r / 3
            - 512 * s**3 / 3
            + 288 * s**2
            - 416 * s / 3
            + 16,
        ]
        dN13 = [
            lambda r, s: 384 * r**2 * s
            + 512 * r * s**2
            - 448 * r * s
            + 128 * s**3
            - 224 * s**2
            + 96 * s,
            lambda r, s: 128 * r**3
            + 512 * r**2 * s
            - 224 * r**2
            + 384 * r * s**2
            - 448 * r * s
            + 96 * r,
        ]
        dN14 = [
            lambda r, s: -384 * r**2 * s
            - 256 * r * s**2
            + 320 * r * s
            + 32 * s**2
            - 32 * s,
            lambda r, s: -128 * r**3
            - 256 * r**2 * s
            + 160 * r**2
            + 64 * r * s
            - 32 * r,
        ]
        dN15 = [
            lambda r, s: -256 * r * s**2
            + 64 * r * s
            - 128 * s**3
            + 160 * s**2
            - 32 * s,
            lambda r, s: -256 * r**2 * s
            + 32 * r**2
            - 384 * r * s**2
            + 320 * r * s
            - 32 * r,
        ]

        dN = np.array(
            [
                dN1,
                dN2,
                dN3,
                dN4,
                dN5,
                dN6,
                dN7,
                dN8,
                dN9,
                dN10,
                dN11,
                dN12,
                dN13,
                dN14,
                dN15,
            ]
        )

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [
            lambda r, s: 128 * r**2
            + 256 * r * s
            - 160 * r
            + 128 * s**2
            - 160 * s
            + 140 / 3,
            lambda r, s: 128 * r**2
            + 256 * r * s
            - 160 * r
            + 128 * s**2
            - 160 * s
            + 140 / 3,
        ]
        ddN2 = [lambda r, s: 128 * r**2 - 96 * r + 44 / 3, lambda r, s: 0]
        ddN3 = [lambda r, s: 0, lambda r, s: 128 * s**2 - 96 * s + 44 / 3]
        ddN4 = [
            lambda r, s: -512 * r**2
            - 768 * r * s
            + 576 * r
            - 256 * s**2
            + 384 * s
            - 416 / 3,
            lambda r, s: -256 * r**2 - 256 * r * s + 192 * r,
        ]
        ddN5 = [
            lambda r, s: 768 * r**2
            + 768 * r * s
            - 768 * r
            + 128 * s**2
            - 288 * s
            + 152,
            lambda r, s: 128 * r**2 - 32 * r,
        ]
        ddN6 = [
            lambda r, s: -512 * r**2 - 256 * r * s + 448 * r + 64 * s - 224 / 3,
            lambda r, s: 0,
        ]
        ddN7 = [lambda r, s: 256 * r * s - 64 * s, lambda r, s: 0]
        ddN8 = [lambda r, s: 128 * s**2 - 32 * s, lambda r, s: 128 * r**2 - 32 * r]
        ddN9 = [lambda r, s: 0, lambda r, s: 256 * r * s - 64 * r]
        ddN10 = [
            lambda r, s: 0,
            lambda r, s: -256 * r * s + 64 * r - 512 * s**2 + 448 * s - 224 / 3,
        ]
        ddN11 = [
            lambda r, s: 128 * s**2 - 32 * s,
            lambda r, s: 128 * r**2
            + 768 * r * s
            - 288 * r
            + 768 * s**2
            - 768 * s
            + 152,
        ]
        ddN12 = [
            lambda r, s: -256 * r * s - 256 * s**2 + 192 * s,
            lambda r, s: -256 * r**2
            - 768 * r * s
            + 384 * r
            - 512 * s**2
            + 576 * s
            - 416 / 3,
        ]
        ddN13 = [
            lambda r, s: 768 * r * s + 512 * s**2 - 448 * s,
            lambda r, s: 512 * r**2 + 768 * r * s - 448 * r,
        ]
        ddN14 = [
            lambda r, s: -768 * r * s - 256 * s**2 + 320 * s,
            lambda r, s: -256 * r**2 + 64 * r,
        ]
        ddN15 = [
            lambda r, s: -256 * s**2 + 64 * s,
            lambda r, s: -256 * r**2 - 768 * r * s + 320 * r,
        ]

        ddN = np.array(
            [
                ddN1,
                ddN2,
                ddN3,
                ddN4,
                ddN5,
                ddN6,
                ddN7,
                ddN8,
                ddN9,
                ddN10,
                ddN11,
                ddN12,
                ddN13,
                ddN14,
                ddN15,
            ]
        )

        return ddN

    def _dddN(self) -> _types.FloatArray:
        dddN1 = [
            lambda r, s: 256 * r + 256 * s - 160,
            lambda r, s: 256 * r + 256 * s - 160,
        ]
        dddN2 = [lambda r, s: 256 * r - 96, lambda r, s: 0]
        dddN3 = [lambda r, s: 0, lambda r, s: 256 * s - 96]
        dddN4 = [lambda r, s: -1024 * r - 768 * s + 576, lambda r, s: -256 * r]
        dddN5 = [lambda r, s: 1536 * r + 768 * s - 768, lambda r, s: 0]
        dddN6 = [lambda r, s: -1024 * r - 256 * s + 448, lambda r, s: 0]
        dddN7 = [lambda r, s: 256 * s, lambda r, s: 0]
        dddN8 = [lambda r, s: 0, lambda r, s: 0]
        dddN9 = [lambda r, s: 0, lambda r, s: 256 * r]
        dddN10 = [lambda r, s: 0, lambda r, s: -256 * r - 1024 * s + 448]
        dddN11 = [lambda r, s: 0, lambda r, s: 768 * r + 1536 * s - 768]
        dddN12 = [lambda r, s: -256 * s, lambda r, s: -768 * r - 1024 * s + 576]
        dddN13 = [lambda r, s: 768 * s, lambda r, s: 768 * r]
        dddN14 = [lambda r, s: -768 * s, lambda r, s: 0]
        dddN15 = [lambda r, s: 0, lambda r, s: -768 * r]

        dddN = np.array(
            [
                dddN1,
                dddN2,
                dddN3,
                dddN4,
                dddN5,
                dddN6,
                dddN7,
                dddN8,
                dddN9,
                dddN10,
                dddN11,
                dddN12,
                dddN13,
                dddN14,
                dddN15,
            ]
        )

        return dddN

    def _ddddN(self) -> _types.FloatArray:
        ddddN1 = [lambda r, s: 256, lambda r, s: 256]
        ddddN2 = [lambda r, s: 256, lambda r, s: 0]
        ddddN3 = [lambda r, s: 0, lambda r, s: 256]
        ddddN4 = [lambda r, s: -1024, lambda r, s: 0]
        ddddN5 = [lambda r, s: 1536, lambda r, s: 0]
        ddddN6 = [lambda r, s: -1024, lambda r, s: 0]
        ddddN7 = [lambda r, s: 0, lambda r, s: 0]
        ddddN8 = [lambda r, s: 0, lambda r, s: 0]
        ddddN9 = [lambda r, s: 0, lambda r, s: 0]
        ddddN10 = [lambda r, s: 0, lambda r, s: -1024]
        ddddN11 = [lambda r, s: 0, lambda r, s: 1536]
        ddddN12 = [lambda r, s: 0, lambda r, s: -1024]
        ddddN13 = [lambda r, s: 0, lambda r, s: 0]
        ddddN14 = [lambda r, s: 0, lambda r, s: 0]
        ddddN15 = [lambda r, s: 0, lambda r, s: 0]

        ddddN = np.array(
            [
                ddddN1,
                ddddN2,
                ddddN3,
                ddddN4,
                ddddN5,
                ddddN6,
                ddddN7,
                ddddN8,
                ddddN9,
                ddddN10,
                ddddN11,
                ddddN12,
                ddddN13,
                ddddN14,
                ddddN15,
            ]
        )

        return ddddN
