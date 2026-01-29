# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Hexa element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class HEXA8(_GroupElem):
    #        v
    # 3----------2
    # |\     ^   |\
    # | \    |   | \
    # |  \   |   |  \
    # |   7------+---6
    # |   |  +-- |-- | -> u
    # 0---+---\--1   |
    #  \  |    \  \  |
    #   \ |     \  \ |
    #    \|      w  \|
    #     4----------5

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 3, 2, 1],
                [0, 1, 5, 4],
                [0, 4, 7, 3],
                [6, 2, 3, 7],
                [6, 5, 1, 2],
                [6, 7, 4, 5],
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return self.surfaces

    def Get_Local_Coords(self) -> _types.FloatArray:
        list_x = [-1, 1, 1, -1, -1, 1, 1, -1]
        list_y = [-1, -1, 1, 1, -1, -1, 1, 1]
        list_z = [-1, -1, -1, -1, 1, 1, 1, 1]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [0, 3],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 5],
                [5, 6],
                [6, 7],
                [4, 7],
            ],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: -(r - 1) * (s - 1) * (t - 1) / 8
        N2 = lambda r, s, t: (r + 1) * (s - 1) * (t - 1) / 8
        N3 = lambda r, s, t: -(r + 1) * (s + 1) * (t - 1) / 8
        N4 = lambda r, s, t: (r - 1) * (s + 1) * (t - 1) / 8
        N5 = lambda r, s, t: (r - 1) * (s - 1) * (t + 1) / 8
        N6 = lambda r, s, t: -(r + 1) * (s - 1) * (t + 1) / 8
        N7 = lambda r, s, t: (r + 1) * (s + 1) * (t + 1) / 8
        N8 = lambda r, s, t: -(r - 1) * (s + 1) * (t + 1) / 8

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: -(s - 1) * (t - 1) / 8,
            lambda r, s, t: -(r - 1) * (t - 1) / 8,
            lambda r, s, t: -(r - 1) * (s - 1) / 8,
        ]
        dN2 = [
            lambda r, s, t: (s - 1) * (t - 1) / 8,
            lambda r, s, t: (r + 1) * (t - 1) / 8,
            lambda r, s, t: (r + 1) * (s - 1) / 8,
        ]
        dN3 = [
            lambda r, s, t: -(s + 1) * (t - 1) / 8,
            lambda r, s, t: -(r + 1) * (t - 1) / 8,
            lambda r, s, t: -(r + 1) * (s + 1) / 8,
        ]
        dN4 = [
            lambda r, s, t: (s + 1) * (t - 1) / 8,
            lambda r, s, t: (r - 1) * (t - 1) / 8,
            lambda r, s, t: (r - 1) * (s + 1) / 8,
        ]
        dN5 = [
            lambda r, s, t: (s - 1) * (t + 1) / 8,
            lambda r, s, t: (r - 1) * (t + 1) / 8,
            lambda r, s, t: (r - 1) * (s - 1) / 8,
        ]
        dN6 = [
            lambda r, s, t: -(s - 1) * (t + 1) / 8,
            lambda r, s, t: -(r + 1) * (t + 1) / 8,
            lambda r, s, t: -(r + 1) * (s - 1) / 8,
        ]
        dN7 = [
            lambda r, s, t: (s + 1) * (t + 1) / 8,
            lambda r, s, t: (r + 1) * (t + 1) / 8,
            lambda r, s, t: (r + 1) * (s + 1) / 8,
        ]
        dN8 = [
            lambda r, s, t: -(s + 1) * (t + 1) / 8,
            lambda r, s, t: -(r - 1) * (t + 1) / 8,
            lambda r, s, t: -(r - 1) * (s + 1) / 8,
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _ddN(self) -> _types.FloatArray:
        return super()._ddN()

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class HEXA20(_GroupElem):
    #        v
    # 3----13----2
    # |\     ^   |\
    # | 15   |   | 14
    # 9  \   |   11 \
    # |   7----19+---6
    # |   |  +-- |-- | -> u
    # 0---+-8-\--1   |
    #  \  17   \  \  18
    #  10 |     \  12|
    #    \|      w  \|
    #     4----16----5

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 9, 3, 13, 2, 11, 1, 8],  # 20
                [0, 8, 1, 12, 5, 16, 4, 10],  # 21
                [0, 10, 4, 17, 7, 15, 3, 9],  # 22
                [6, 18, 5, 12, 1, 11, 2, 14],  # 23
                [6, 14, 2, 13, 3, 15, 7, 19],  # 24
                [6, 19, 7, 17, 4, 16, 5, 18],  # 25
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 3, 2, 1, 9, 13, 11, 8],  # 20
                [0, 1, 5, 4, 8, 12, 16, 10],  # 21
                [0, 4, 7, 3, 10, 17, 15, 9],  # 22
                [6, 5, 1, 2, 18, 12, 11, 14],  # 23
                [6, 2, 3, 7, 14, 13, 15, 19],  # 24
                [6, 7, 4, 5, 19, 17, 16, 18],  # 25
            ],
            dtype=int,
        )

    def Get_Local_Coords(self):
        # fmt: off
        list_x = [
            -1,1,1,-1,
            -1,1,1,-1,
            0,-1,-1,1,
            1,0,1,-1,
            0,-1,1,0
        ]
        list_y = [
            -1,-1,1,1,
            -1,-1,1,1,
            -1,0,-1,0,
            -1,1,1,1,
            -1,0,0,1
        ]
        list_z = [
            -1,-1,-1,-1,
            1,1,1,1,
            -1,-1,0,-1,
            0,-1,0,0,
            1,1,1,1
        ]
        # fmt: on
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 8, 1],
                [0, 9, 3],
                [4, 10, 0],
                [1, 11, 2],
                [1, 12, 5],
                [3, 13, 2],
                [2, 14, 6],
                [7, 15, 3],
                [5, 16, 4],
                [4, 17, 7],
                [5, 18, 6],
                [6, 19, 7],
            ]
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: (r - 1) * (s - 1) * (t - 1) * (r + s + t + 2) / 8
        N2 = lambda r, s, t: (r + 1) * (s - 1) * (t - 1) * (r - s - t - 2) / 8
        N3 = lambda r, s, t: -(r + 1) * (s + 1) * (t - 1) * (r + s - t - 2) / 8
        N4 = lambda r, s, t: -(r - 1) * (s + 1) * (t - 1) * (r - s + t + 2) / 8
        N5 = lambda r, s, t: -(r - 1) * (s - 1) * (t + 1) * (r + s - t + 2) / 8
        N6 = lambda r, s, t: -(r + 1) * (s - 1) * (t + 1) * (r - s + t - 2) / 8
        N7 = lambda r, s, t: (r + 1) * (s + 1) * (t + 1) * (r + s + t - 2) / 8
        N8 = lambda r, s, t: (r - 1) * (s + 1) * (t + 1) * (r - s - t + 2) / 8
        N9 = lambda r, s, t: -(r - 1) * (r + 1) * (s - 1) * (t - 1) / 4
        N10 = lambda r, s, t: -(r - 1) * (s - 1) * (s + 1) * (t - 1) / 4
        N11 = lambda r, s, t: -(r - 1) * (s - 1) * (t - 1) * (t + 1) / 4
        N12 = lambda r, s, t: (r + 1) * (s - 1) * (s + 1) * (t - 1) / 4
        N13 = lambda r, s, t: (r + 1) * (s - 1) * (t - 1) * (t + 1) / 4
        N14 = lambda r, s, t: (r - 1) * (r + 1) * (s + 1) * (t - 1) / 4
        N15 = lambda r, s, t: -(r + 1) * (s + 1) * (t - 1) * (t + 1) / 4
        N16 = lambda r, s, t: (r - 1) * (s + 1) * (t - 1) * (t + 1) / 4
        N17 = lambda r, s, t: (r - 1) * (r + 1) * (s - 1) * (t + 1) / 4
        N18 = lambda r, s, t: (r - 1) * (s - 1) * (s + 1) * (t + 1) / 4
        N19 = lambda r, s, t: -(r + 1) * (s - 1) * (s + 1) * (t + 1) / 4
        N20 = lambda r, s, t: -(r - 1) * (r + 1) * (s + 1) * (t + 1) / 4

        N = np.array(
            [
                N1,
                N2,
                N3,
                N4,
                N5,
                N6,
                N7,
                N8,
                N9,
                N10,
                N11,
                N12,
                N13,
                N14,
                N15,
                N16,
                N17,
                N18,
                N19,
                N20,
            ]
        ).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: r * s * t / 4
            - r * s / 4
            - r * t / 4
            + r / 4
            + s**2 * t / 8
            - s**2 / 8
            + s * t**2 / 8
            - s * t / 8
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: r**2 * t / 8
            - r**2 / 8
            + r * s * t / 4
            - r * s / 4
            + r * t**2 / 8
            - r * t / 8
            - s * t / 4
            + s / 4
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: r**2 * s / 8
            - r**2 / 8
            + r * s**2 / 8
            + r * s * t / 4
            - r * s / 8
            - r * t / 4
            - s**2 / 8
            - s * t / 4
            + t / 4
            + 1 / 8,
        ]
        dN2 = [
            lambda r, s, t: r * s * t / 4
            - r * s / 4
            - r * t / 4
            + r / 4
            - s**2 * t / 8
            + s**2 / 8
            - s * t**2 / 8
            + s * t / 8
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: r**2 * t / 8
            - r**2 / 8
            - r * s * t / 4
            + r * s / 4
            - r * t**2 / 8
            + r * t / 8
            - s * t / 4
            + s / 4
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: r**2 * s / 8
            - r**2 / 8
            - r * s**2 / 8
            - r * s * t / 4
            + r * s / 8
            + r * t / 4
            - s**2 / 8
            - s * t / 4
            + t / 4
            + 1 / 8,
        ]
        dN3 = [
            lambda r, s, t: -r * s * t / 4
            + r * s / 4
            - r * t / 4
            + r / 4
            - s**2 * t / 8
            + s**2 / 8
            + s * t**2 / 8
            - s * t / 8
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: -(r**2) * t / 8
            + r**2 / 8
            - r * s * t / 4
            + r * s / 4
            + r * t**2 / 8
            - r * t / 8
            - s * t / 4
            + s / 4
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: -(r**2) * s / 8
            - r**2 / 8
            - r * s**2 / 8
            + r * s * t / 4
            - r * s / 8
            + r * t / 4
            - s**2 / 8
            + s * t / 4
            + t / 4
            + 1 / 8,
        ]
        dN4 = [
            lambda r, s, t: -r * s * t / 4
            + r * s / 4
            - r * t / 4
            + r / 4
            + s**2 * t / 8
            - s**2 / 8
            - s * t**2 / 8
            + s * t / 8
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: -(r**2) * t / 8
            + r**2 / 8
            + r * s * t / 4
            - r * s / 4
            - r * t**2 / 8
            + r * t / 8
            - s * t / 4
            + s / 4
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: -(r**2) * s / 8
            - r**2 / 8
            + r * s**2 / 8
            - r * s * t / 4
            + r * s / 8
            - r * t / 4
            - s**2 / 8
            + s * t / 4
            + t / 4
            + 1 / 8,
        ]
        dN5 = [
            lambda r, s, t: -r * s * t / 4
            - r * s / 4
            + r * t / 4
            + r / 4
            - s**2 * t / 8
            - s**2 / 8
            + s * t**2 / 8
            + s * t / 8
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: -(r**2) * t / 8
            - r**2 / 8
            - r * s * t / 4
            - r * s / 4
            + r * t**2 / 8
            + r * t / 8
            + s * t / 4
            + s / 4
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: -(r**2) * s / 8
            + r**2 / 8
            - r * s**2 / 8
            + r * s * t / 4
            + r * s / 8
            - r * t / 4
            + s**2 / 8
            - s * t / 4
            + t / 4
            - 1 / 8,
        ]
        dN6 = [
            lambda r, s, t: -r * s * t / 4
            - r * s / 4
            + r * t / 4
            + r / 4
            + s**2 * t / 8
            + s**2 / 8
            - s * t**2 / 8
            - s * t / 8
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: -(r**2) * t / 8
            - r**2 / 8
            + r * s * t / 4
            + r * s / 4
            - r * t**2 / 8
            - r * t / 8
            + s * t / 4
            + s / 4
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: -(r**2) * s / 8
            + r**2 / 8
            + r * s**2 / 8
            - r * s * t / 4
            - r * s / 8
            + r * t / 4
            + s**2 / 8
            - s * t / 4
            + t / 4
            - 1 / 8,
        ]
        dN7 = [
            lambda r, s, t: r * s * t / 4
            + r * s / 4
            + r * t / 4
            + r / 4
            + s**2 * t / 8
            + s**2 / 8
            + s * t**2 / 8
            + s * t / 8
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: r**2 * t / 8
            + r**2 / 8
            + r * s * t / 4
            + r * s / 4
            + r * t**2 / 8
            + r * t / 8
            + s * t / 4
            + s / 4
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: r**2 * s / 8
            + r**2 / 8
            + r * s**2 / 8
            + r * s * t / 4
            + r * s / 8
            + r * t / 4
            + s**2 / 8
            + s * t / 4
            + t / 4
            - 1 / 8,
        ]
        dN8 = [
            lambda r, s, t: r * s * t / 4
            + r * s / 4
            + r * t / 4
            + r / 4
            - s**2 * t / 8
            - s**2 / 8
            - s * t**2 / 8
            - s * t / 8
            - t**2 / 8
            + 1 / 8,
            lambda r, s, t: r**2 * t / 8
            + r**2 / 8
            - r * s * t / 4
            - r * s / 4
            - r * t**2 / 8
            - r * t / 8
            + s * t / 4
            + s / 4
            + t**2 / 8
            - 1 / 8,
            lambda r, s, t: r**2 * s / 8
            + r**2 / 8
            - r * s**2 / 8
            - r * s * t / 4
            - r * s / 8
            - r * t / 4
            + s**2 / 8
            + s * t / 4
            + t / 4
            - 1 / 8,
        ]
        dN9 = [
            lambda r, s, t: -r * s * t / 2 + r * s / 2 + r * t / 2 - r / 2,
            lambda r, s, t: -(r**2) * t / 4 + r**2 / 4 + t / 4 - 1 / 4,
            lambda r, s, t: -(r**2) * s / 4 + r**2 / 4 + s / 4 - 1 / 4,
        ]
        dN10 = [
            lambda r, s, t: -(s**2) * t / 4 + s**2 / 4 + t / 4 - 1 / 4,
            lambda r, s, t: -r * s * t / 2 + r * s / 2 + s * t / 2 - s / 2,
            lambda r, s, t: -r * s**2 / 4 + r / 4 + s**2 / 4 - 1 / 4,
        ]
        dN11 = [
            lambda r, s, t: -s * t**2 / 4 + s / 4 + t**2 / 4 - 1 / 4,
            lambda r, s, t: -r * t**2 / 4 + r / 4 + t**2 / 4 - 1 / 4,
            lambda r, s, t: -r * s * t / 2 + r * t / 2 + s * t / 2 - t / 2,
        ]
        dN12 = [
            lambda r, s, t: s**2 * t / 4 - s**2 / 4 - t / 4 + 1 / 4,
            lambda r, s, t: r * s * t / 2 - r * s / 2 + s * t / 2 - s / 2,
            lambda r, s, t: r * s**2 / 4 - r / 4 + s**2 / 4 - 1 / 4,
        ]
        dN13 = [
            lambda r, s, t: s * t**2 / 4 - s / 4 - t**2 / 4 + 1 / 4,
            lambda r, s, t: r * t**2 / 4 - r / 4 + t**2 / 4 - 1 / 4,
            lambda r, s, t: r * s * t / 2 - r * t / 2 + s * t / 2 - t / 2,
        ]
        dN14 = [
            lambda r, s, t: r * s * t / 2 - r * s / 2 + r * t / 2 - r / 2,
            lambda r, s, t: r**2 * t / 4 - r**2 / 4 - t / 4 + 1 / 4,
            lambda r, s, t: r**2 * s / 4 + r**2 / 4 - s / 4 - 1 / 4,
        ]
        dN15 = [
            lambda r, s, t: -s * t**2 / 4 + s / 4 - t**2 / 4 + 1 / 4,
            lambda r, s, t: -r * t**2 / 4 + r / 4 - t**2 / 4 + 1 / 4,
            lambda r, s, t: -r * s * t / 2 - r * t / 2 - s * t / 2 - t / 2,
        ]
        dN16 = [
            lambda r, s, t: s * t**2 / 4 - s / 4 + t**2 / 4 - 1 / 4,
            lambda r, s, t: r * t**2 / 4 - r / 4 - t**2 / 4 + 1 / 4,
            lambda r, s, t: r * s * t / 2 + r * t / 2 - s * t / 2 - t / 2,
        ]
        dN17 = [
            lambda r, s, t: r * s * t / 2 + r * s / 2 - r * t / 2 - r / 2,
            lambda r, s, t: r**2 * t / 4 + r**2 / 4 - t / 4 - 1 / 4,
            lambda r, s, t: r**2 * s / 4 - r**2 / 4 - s / 4 + 1 / 4,
        ]
        dN18 = [
            lambda r, s, t: s**2 * t / 4 + s**2 / 4 - t / 4 - 1 / 4,
            lambda r, s, t: r * s * t / 2 + r * s / 2 - s * t / 2 - s / 2,
            lambda r, s, t: r * s**2 / 4 - r / 4 - s**2 / 4 + 1 / 4,
        ]
        dN19 = [
            lambda r, s, t: -(s**2) * t / 4 - s**2 / 4 + t / 4 + 1 / 4,
            lambda r, s, t: -r * s * t / 2 - r * s / 2 - s * t / 2 - s / 2,
            lambda r, s, t: -r * s**2 / 4 + r / 4 - s**2 / 4 + 1 / 4,
        ]
        dN20 = [
            lambda r, s, t: -r * s * t / 2 - r * s / 2 - r * t / 2 - r / 2,
            lambda r, s, t: -(r**2) * t / 4 - r**2 / 4 + t / 4 + 1 / 4,
            lambda r, s, t: -(r**2) * s / 4 - r**2 / 4 + s / 4 + 1 / 4,
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
                dN16,
                dN17,
                dN18,
                dN19,
                dN20,
            ]
        )

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [
            lambda r, s, t: (s - 1) * (t - 1) / 4,
            lambda r, s, t: (r - 1) * (t - 1) / 4,
            lambda r, s, t: (r - 1) * (s - 1) / 4,
        ]
        ddN2 = [
            lambda r, s, t: (s - 1) * (t - 1) / 4,
            lambda r, s, t: -(r + 1) * (t - 1) / 4,
            lambda r, s, t: -(r + 1) * (s - 1) / 4,
        ]
        ddN3 = [
            lambda r, s, t: -(s + 1) * (t - 1) / 4,
            lambda r, s, t: -(r + 1) * (t - 1) / 4,
            lambda r, s, t: (r + 1) * (s + 1) / 4,
        ]
        ddN4 = [
            lambda r, s, t: -(s + 1) * (t - 1) / 4,
            lambda r, s, t: (r - 1) * (t - 1) / 4,
            lambda r, s, t: -(r - 1) * (s + 1) / 4,
        ]
        ddN5 = [
            lambda r, s, t: -(s - 1) * (t + 1) / 4,
            lambda r, s, t: -(r - 1) * (t + 1) / 4,
            lambda r, s, t: (r - 1) * (s - 1) / 4,
        ]
        ddN6 = [
            lambda r, s, t: -(s - 1) * (t + 1) / 4,
            lambda r, s, t: (r + 1) * (t + 1) / 4,
            lambda r, s, t: -(r + 1) * (s - 1) / 4,
        ]
        ddN7 = [
            lambda r, s, t: (s + 1) * (t + 1) / 4,
            lambda r, s, t: (r + 1) * (t + 1) / 4,
            lambda r, s, t: (r + 1) * (s + 1) / 4,
        ]
        ddN8 = [
            lambda r, s, t: (s + 1) * (t + 1) / 4,
            lambda r, s, t: -(r - 1) * (t + 1) / 4,
            lambda r, s, t: -(r - 1) * (s + 1) / 4,
        ]
        ddN9 = [
            lambda r, s, t: -(s - 1) * (t - 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: 0,
        ]
        ddN10 = [
            lambda r, s, t: 0,
            lambda r, s, t: -(r - 1) * (t - 1) / 2,
            lambda r, s, t: 0,
        ]
        ddN11 = [
            lambda r, s, t: 0,
            lambda r, s, t: 0,
            lambda r, s, t: -(r - 1) * (s - 1) / 2,
        ]
        ddN12 = [
            lambda r, s, t: 0,
            lambda r, s, t: (r + 1) * (t - 1) / 2,
            lambda r, s, t: 0,
        ]
        ddN13 = [
            lambda r, s, t: 0,
            lambda r, s, t: 0,
            lambda r, s, t: (r + 1) * (s - 1) / 2,
        ]
        ddN14 = [
            lambda r, s, t: (s + 1) * (t - 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: 0,
        ]
        ddN15 = [
            lambda r, s, t: 0,
            lambda r, s, t: 0,
            lambda r, s, t: -(r + 1) * (s + 1) / 2,
        ]
        ddN16 = [
            lambda r, s, t: 0,
            lambda r, s, t: 0,
            lambda r, s, t: (r - 1) * (s + 1) / 2,
        ]
        ddN17 = [
            lambda r, s, t: (s - 1) * (t + 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: 0,
        ]
        ddN18 = [
            lambda r, s, t: 0,
            lambda r, s, t: (r - 1) * (t + 1) / 2,
            lambda r, s, t: 0,
        ]
        ddN19 = [
            lambda r, s, t: 0,
            lambda r, s, t: -(r + 1) * (t + 1) / 2,
            lambda r, s, t: 0,
        ]
        ddN20 = [
            lambda r, s, t: -(s + 1) * (t + 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: 0,
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
                ddN16,
                ddN17,
                ddN18,
                ddN19,
                ddN20,
            ]
        )

        return ddN

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class HEXA27(_GroupElem):
    #
    # 3----13----2
    # |\         |\
    # |15    24  | 14
    # 9  \ 20    11 \
    # |   7----19+---6
    # |22 |  26  | 23|
    # 0---+-8----1   |
    #  \ 17    25 \  18
    #  10 |  21    12|
    #    \|         \|
    #     4----16----5

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 9, 3, 13, 2, 11, 1, 8],  # 20
                [0, 8, 1, 12, 5, 16, 4, 10],  # 21
                [0, 10, 4, 17, 7, 15, 3, 9],  # 22
                [6, 18, 5, 12, 1, 11, 2, 14],  # 23
                [6, 14, 2, 13, 3, 15, 7, 19],  # 24
                [6, 19, 7, 17, 4, 16, 5, 18],  # 25
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 3, 2, 1, 9, 13, 11, 8, 20],
                [0, 1, 5, 4, 8, 12, 16, 10, 21],
                [0, 4, 7, 3, 10, 17, 15, 9, 22],
                [6, 5, 1, 2, 18, 12, 11, 14, 23],
                [6, 2, 3, 7, 14, 13, 15, 19, 24],
                [6, 7, 4, 5, 19, 17, 16, 18, 25],
            ],
            dtype=int,
        )

    def Get_Local_Coords(self):
        # fmt: off
        list_x = [
            -1,1,1,-1,
            -1,1,1,-1,
            0,-1,-1,1,
            1,0,1,-1,
            0,-1,1,0,
            0,0,-1,1,
            0,0,0
        ]
        list_y = [
            -1,-1,1,1,
            -1,-1,1,1,
            -1,0,-1,0,
            -1,1,1,1,
            -1,0,0,1,
            0,-1,0,0,
            1,0,0
        ]
        list_z = [
            -1,-1,-1,-1,
            1,1,1,1,
            -1,-1,0,-1,
            0,-1,0,0,
            1,1,1,1,
            -1,0,0,0,
            0,1,0
        ]
        # fmt: on
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 8, 1],
                [0, 9, 3],
                [4, 10, 0],
                [1, 11, 2],
                [1, 12, 5],
                [3, 13, 2],
                [2, 14, 6],
                [7, 15, 3],
                [5, 16, 4],
                [4, 17, 7],
                [5, 18, 6],
                [6, 19, 7],
            ],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: r * s * t * (r - 1) * (s - 1) * (t - 1) / 8
        N2 = lambda r, s, t: r * s * t * (r + 1) * (s - 1) * (t - 1) / 8
        N3 = lambda r, s, t: r * s * t * (r + 1) * (s + 1) * (t - 1) / 8
        N4 = lambda r, s, t: r * s * t * (r - 1) * (s + 1) * (t - 1) / 8
        N5 = lambda r, s, t: r * s * t * (r - 1) * (s - 1) * (t + 1) / 8
        N6 = lambda r, s, t: r * s * t * (r + 1) * (s - 1) * (t + 1) / 8
        N7 = lambda r, s, t: r * s * t * (r + 1) * (s + 1) * (t + 1) / 8
        N8 = lambda r, s, t: r * s * t * (r - 1) * (s + 1) * (t + 1) / 8
        N9 = lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s - 1) * (t - 1) / 4
        N10 = lambda r, s, t: -r * t * (r - 1) * (s - 1) * (s + 1) * (t - 1) / 4
        N11 = lambda r, s, t: -r * s * (r - 1) * (s - 1) * (t - 1) * (t + 1) / 4
        N12 = lambda r, s, t: -r * t * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 4
        N13 = lambda r, s, t: -r * s * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 4
        N14 = lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s + 1) * (t - 1) / 4
        N15 = lambda r, s, t: -r * s * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 4
        N16 = lambda r, s, t: -r * s * (r - 1) * (s + 1) * (t - 1) * (t + 1) / 4
        N17 = lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s - 1) * (t + 1) / 4
        N18 = lambda r, s, t: -r * t * (r - 1) * (s - 1) * (s + 1) * (t + 1) / 4
        N19 = lambda r, s, t: -r * t * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 4
        N20 = lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s + 1) * (t + 1) / 4
        N21 = lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 2
        N22 = lambda r, s, t: s * (r - 1) * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 2
        N23 = lambda r, s, t: r * (r - 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2
        N24 = lambda r, s, t: r * (r + 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2
        N25 = lambda r, s, t: s * (r - 1) * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 2
        N26 = lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 2
        N27 = lambda r, s, t: -(r - 1) * (r + 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1)

        N = np.array(
            [
                N1,
                N2,
                N3,
                N4,
                N5,
                N6,
                N7,
                N8,
                N9,
                N10,
                N11,
                N12,
                N13,
                N14,
                N15,
                N16,
                N17,
                N18,
                N19,
                N20,
                N21,
                N22,
                N23,
                N24,
                N25,
                N26,
                N27,
            ]
        ).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: r * s * t * (s - 1) * (t - 1) / 8
            + s * t * (r - 1) * (s - 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (t - 1) / 8
            + r * t * (r - 1) * (s - 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (s - 1) / 8
            + r * s * (r - 1) * (s - 1) * (t - 1) / 8,
        ]
        dN2 = [
            lambda r, s, t: r * s * t * (s - 1) * (t - 1) / 8
            + s * t * (r + 1) * (s - 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (t - 1) / 8
            + r * t * (r + 1) * (s - 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (s - 1) / 8
            + r * s * (r + 1) * (s - 1) * (t - 1) / 8,
        ]
        dN3 = [
            lambda r, s, t: r * s * t * (s + 1) * (t - 1) / 8
            + s * t * (r + 1) * (s + 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (t - 1) / 8
            + r * t * (r + 1) * (s + 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (s + 1) / 8
            + r * s * (r + 1) * (s + 1) * (t - 1) / 8,
        ]
        dN4 = [
            lambda r, s, t: r * s * t * (s + 1) * (t - 1) / 8
            + s * t * (r - 1) * (s + 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (t - 1) / 8
            + r * t * (r - 1) * (s + 1) * (t - 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (s + 1) / 8
            + r * s * (r - 1) * (s + 1) * (t - 1) / 8,
        ]
        dN5 = [
            lambda r, s, t: r * s * t * (s - 1) * (t + 1) / 8
            + s * t * (r - 1) * (s - 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (t + 1) / 8
            + r * t * (r - 1) * (s - 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (s - 1) / 8
            + r * s * (r - 1) * (s - 1) * (t + 1) / 8,
        ]
        dN6 = [
            lambda r, s, t: r * s * t * (s - 1) * (t + 1) / 8
            + s * t * (r + 1) * (s - 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (t + 1) / 8
            + r * t * (r + 1) * (s - 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (s - 1) / 8
            + r * s * (r + 1) * (s - 1) * (t + 1) / 8,
        ]
        dN7 = [
            lambda r, s, t: r * s * t * (s + 1) * (t + 1) / 8
            + s * t * (r + 1) * (s + 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (t + 1) / 8
            + r * t * (r + 1) * (s + 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r + 1) * (s + 1) / 8
            + r * s * (r + 1) * (s + 1) * (t + 1) / 8,
        ]
        dN8 = [
            lambda r, s, t: r * s * t * (s + 1) * (t + 1) / 8
            + s * t * (r - 1) * (s + 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (t + 1) / 8
            + r * t * (r - 1) * (s + 1) * (t + 1) / 8,
            lambda r, s, t: r * s * t * (r - 1) * (s + 1) / 8
            + r * s * (r - 1) * (s + 1) * (t + 1) / 8,
        ]
        dN9 = [
            lambda r, s, t: -s * t * (r - 1) * (s - 1) * (t - 1) / 4
            - s * t * (r + 1) * (s - 1) * (t - 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (t - 1) / 4
            - t * (r - 1) * (r + 1) * (s - 1) * (t - 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s - 1) / 4
            - s * (r - 1) * (r + 1) * (s - 1) * (t - 1) / 4,
        ]
        dN10 = [
            lambda r, s, t: -r * t * (s - 1) * (s + 1) * (t - 1) / 4
            - t * (r - 1) * (s - 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -r * t * (r - 1) * (s - 1) * (t - 1) / 4
            - r * t * (r - 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -r * t * (r - 1) * (s - 1) * (s + 1) / 4
            - r * (r - 1) * (s - 1) * (s + 1) * (t - 1) / 4,
        ]
        dN11 = [
            lambda r, s, t: -r * s * (s - 1) * (t - 1) * (t + 1) / 4
            - s * (r - 1) * (s - 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r - 1) * (t - 1) * (t + 1) / 4
            - r * (r - 1) * (s - 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r - 1) * (s - 1) * (t - 1) / 4
            - r * s * (r - 1) * (s - 1) * (t + 1) / 4,
        ]
        dN12 = [
            lambda r, s, t: -r * t * (s - 1) * (s + 1) * (t - 1) / 4
            - t * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -r * t * (r + 1) * (s - 1) * (t - 1) / 4
            - r * t * (r + 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -r * t * (r + 1) * (s - 1) * (s + 1) / 4
            - r * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 4,
        ]
        dN13 = [
            lambda r, s, t: -r * s * (s - 1) * (t - 1) * (t + 1) / 4
            - s * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r + 1) * (t - 1) * (t + 1) / 4
            - r * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r + 1) * (s - 1) * (t - 1) / 4
            - r * s * (r + 1) * (s - 1) * (t + 1) / 4,
        ]
        dN14 = [
            lambda r, s, t: -s * t * (r - 1) * (s + 1) * (t - 1) / 4
            - s * t * (r + 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (t - 1) / 4
            - t * (r - 1) * (r + 1) * (s + 1) * (t - 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s + 1) / 4
            - s * (r - 1) * (r + 1) * (s + 1) * (t - 1) / 4,
        ]
        dN15 = [
            lambda r, s, t: -r * s * (s + 1) * (t - 1) * (t + 1) / 4
            - s * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r + 1) * (t - 1) * (t + 1) / 4
            - r * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r + 1) * (s + 1) * (t - 1) / 4
            - r * s * (r + 1) * (s + 1) * (t + 1) / 4,
        ]
        dN16 = [
            lambda r, s, t: -r * s * (s + 1) * (t - 1) * (t + 1) / 4
            - s * (r - 1) * (s + 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r - 1) * (t - 1) * (t + 1) / 4
            - r * (r - 1) * (s + 1) * (t - 1) * (t + 1) / 4,
            lambda r, s, t: -r * s * (r - 1) * (s + 1) * (t - 1) / 4
            - r * s * (r - 1) * (s + 1) * (t + 1) / 4,
        ]
        dN17 = [
            lambda r, s, t: -s * t * (r - 1) * (s - 1) * (t + 1) / 4
            - s * t * (r + 1) * (s - 1) * (t + 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (t + 1) / 4
            - t * (r - 1) * (r + 1) * (s - 1) * (t + 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s - 1) / 4
            - s * (r - 1) * (r + 1) * (s - 1) * (t + 1) / 4,
        ]
        dN18 = [
            lambda r, s, t: -r * t * (s - 1) * (s + 1) * (t + 1) / 4
            - t * (r - 1) * (s - 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -r * t * (r - 1) * (s - 1) * (t + 1) / 4
            - r * t * (r - 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -r * t * (r - 1) * (s - 1) * (s + 1) / 4
            - r * (r - 1) * (s - 1) * (s + 1) * (t + 1) / 4,
        ]
        dN19 = [
            lambda r, s, t: -r * t * (s - 1) * (s + 1) * (t + 1) / 4
            - t * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -r * t * (r + 1) * (s - 1) * (t + 1) / 4
            - r * t * (r + 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -r * t * (r + 1) * (s - 1) * (s + 1) / 4
            - r * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 4,
        ]
        dN20 = [
            lambda r, s, t: -s * t * (r - 1) * (s + 1) * (t + 1) / 4
            - s * t * (r + 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (t + 1) / 4
            - t * (r - 1) * (r + 1) * (s + 1) * (t + 1) / 4,
            lambda r, s, t: -s * t * (r - 1) * (r + 1) * (s + 1) / 4
            - s * (r - 1) * (r + 1) * (s + 1) * (t + 1) / 4,
        ]
        dN21 = [
            lambda r, s, t: t * (r - 1) * (s - 1) * (s + 1) * (t - 1) / 2
            + t * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 2,
            lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (t - 1) / 2
            + t * (r - 1) * (r + 1) * (s + 1) * (t - 1) / 2,
            lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (s + 1) / 2
            + (r - 1) * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 2,
        ]
        dN22 = [
            lambda r, s, t: s * (r - 1) * (s - 1) * (t - 1) * (t + 1) / 2
            + s * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: s * (r - 1) * (r + 1) * (t - 1) * (t + 1) / 2
            + (r - 1) * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: s * (r - 1) * (r + 1) * (s - 1) * (t - 1) / 2
            + s * (r - 1) * (r + 1) * (s - 1) * (t + 1) / 2,
        ]
        dN23 = [
            lambda r, s, t: r * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2
            + (r - 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: r * (r - 1) * (s - 1) * (t - 1) * (t + 1) / 2
            + r * (r - 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: r * (r - 1) * (s - 1) * (s + 1) * (t - 1) / 2
            + r * (r - 1) * (s - 1) * (s + 1) * (t + 1) / 2,
        ]
        dN24 = [
            lambda r, s, t: r * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2
            + (r + 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: r * (r + 1) * (s - 1) * (t - 1) * (t + 1) / 2
            + r * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: r * (r + 1) * (s - 1) * (s + 1) * (t - 1) / 2
            + r * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 2,
        ]
        dN25 = [
            lambda r, s, t: s * (r - 1) * (s + 1) * (t - 1) * (t + 1) / 2
            + s * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: s * (r - 1) * (r + 1) * (t - 1) * (t + 1) / 2
            + (r - 1) * (r + 1) * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: s * (r - 1) * (r + 1) * (s + 1) * (t - 1) / 2
            + s * (r - 1) * (r + 1) * (s + 1) * (t + 1) / 2,
        ]
        dN26 = [
            lambda r, s, t: t * (r - 1) * (s - 1) * (s + 1) * (t + 1) / 2
            + t * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 2,
            lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (t + 1) / 2
            + t * (r - 1) * (r + 1) * (s + 1) * (t + 1) / 2,
            lambda r, s, t: t * (r - 1) * (r + 1) * (s - 1) * (s + 1) / 2
            + (r - 1) * (r + 1) * (s - 1) * (s + 1) * (t + 1) / 2,
        ]
        dN27 = [
            lambda r, s, t: -(r - 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1)
            - (r + 1) * (s - 1) * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: -(r - 1) * (r + 1) * (s - 1) * (t - 1) * (t + 1)
            - (r - 1) * (r + 1) * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: -(r - 1) * (r + 1) * (s - 1) * (s + 1) * (t - 1)
            - (r - 1) * (r + 1) * (s - 1) * (s + 1) * (t + 1),
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
                dN16,
                dN17,
                dN18,
                dN19,
                dN20,
                dN21,
                dN22,
                dN23,
                dN24,
                dN25,
                dN26,
                dN27,
            ]
        )

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [
            lambda r, s, t: s * t * (s - 1) * (t - 1) / 4,
            lambda r, s, t: r * t * (r - 1) * (t - 1) / 4,
            lambda r, s, t: r * s * (r - 1) * (s - 1) / 4,
        ]
        ddN2 = [
            lambda r, s, t: s * t * (s - 1) * (t - 1) / 4,
            lambda r, s, t: r * t * (r + 1) * (t - 1) / 4,
            lambda r, s, t: r * s * (r + 1) * (s - 1) / 4,
        ]
        ddN3 = [
            lambda r, s, t: s * t * (s + 1) * (t - 1) / 4,
            lambda r, s, t: r * t * (r + 1) * (t - 1) / 4,
            lambda r, s, t: r * s * (r + 1) * (s + 1) / 4,
        ]
        ddN4 = [
            lambda r, s, t: s * t * (s + 1) * (t - 1) / 4,
            lambda r, s, t: r * t * (r - 1) * (t - 1) / 4,
            lambda r, s, t: r * s * (r - 1) * (s + 1) / 4,
        ]
        ddN5 = [
            lambda r, s, t: s * t * (s - 1) * (t + 1) / 4,
            lambda r, s, t: r * t * (r - 1) * (t + 1) / 4,
            lambda r, s, t: r * s * (r - 1) * (s - 1) / 4,
        ]
        ddN6 = [
            lambda r, s, t: s * t * (s - 1) * (t + 1) / 4,
            lambda r, s, t: r * t * (r + 1) * (t + 1) / 4,
            lambda r, s, t: r * s * (r + 1) * (s - 1) / 4,
        ]
        ddN7 = [
            lambda r, s, t: s * t * (s + 1) * (t + 1) / 4,
            lambda r, s, t: r * t * (r + 1) * (t + 1) / 4,
            lambda r, s, t: r * s * (r + 1) * (s + 1) / 4,
        ]
        ddN8 = [
            lambda r, s, t: s * t * (s + 1) * (t + 1) / 4,
            lambda r, s, t: r * t * (r - 1) * (t + 1) / 4,
            lambda r, s, t: r * s * (r - 1) * (s + 1) / 4,
        ]
        ddN9 = [
            lambda r, s, t: -s * t * (s - 1) * (t - 1) / 2,
            lambda r, s, t: -t * (r - 1) * (r + 1) * (t - 1) / 2,
            lambda r, s, t: -s * (r - 1) * (r + 1) * (s - 1) / 2,
        ]
        ddN10 = [
            lambda r, s, t: -t * (s - 1) * (s + 1) * (t - 1) / 2,
            lambda r, s, t: -r * t * (r - 1) * (t - 1) / 2,
            lambda r, s, t: -r * (r - 1) * (s - 1) * (s + 1) / 2,
        ]
        ddN11 = [
            lambda r, s, t: -s * (s - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * s * (r - 1) * (s - 1) / 2,
        ]
        ddN12 = [
            lambda r, s, t: -t * (s - 1) * (s + 1) * (t - 1) / 2,
            lambda r, s, t: -r * t * (r + 1) * (t - 1) / 2,
            lambda r, s, t: -r * (r + 1) * (s - 1) * (s + 1) / 2,
        ]
        ddN13 = [
            lambda r, s, t: -s * (s - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * s * (r + 1) * (s - 1) / 2,
        ]
        ddN14 = [
            lambda r, s, t: -s * t * (s + 1) * (t - 1) / 2,
            lambda r, s, t: -t * (r - 1) * (r + 1) * (t - 1) / 2,
            lambda r, s, t: -s * (r - 1) * (r + 1) * (s + 1) / 2,
        ]
        ddN15 = [
            lambda r, s, t: -s * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * s * (r + 1) * (s + 1) / 2,
        ]
        ddN16 = [
            lambda r, s, t: -s * (s + 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r - 1) * (t - 1) * (t + 1) / 2,
            lambda r, s, t: -r * s * (r - 1) * (s + 1) / 2,
        ]
        ddN17 = [
            lambda r, s, t: -s * t * (s - 1) * (t + 1) / 2,
            lambda r, s, t: -t * (r - 1) * (r + 1) * (t + 1) / 2,
            lambda r, s, t: -s * (r - 1) * (r + 1) * (s - 1) / 2,
        ]
        ddN18 = [
            lambda r, s, t: -t * (s - 1) * (s + 1) * (t + 1) / 2,
            lambda r, s, t: -r * t * (r - 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r - 1) * (s - 1) * (s + 1) / 2,
        ]
        ddN19 = [
            lambda r, s, t: -t * (s - 1) * (s + 1) * (t + 1) / 2,
            lambda r, s, t: -r * t * (r + 1) * (t + 1) / 2,
            lambda r, s, t: -r * (r + 1) * (s - 1) * (s + 1) / 2,
        ]
        ddN20 = [
            lambda r, s, t: -s * t * (s + 1) * (t + 1) / 2,
            lambda r, s, t: -t * (r - 1) * (r + 1) * (t + 1) / 2,
            lambda r, s, t: -s * (r - 1) * (r + 1) * (s + 1) / 2,
        ]
        ddN21 = [
            lambda r, s, t: t * (s - 1) * (s + 1) * (t - 1),
            lambda r, s, t: t * (r - 1) * (r + 1) * (t - 1),
            lambda r, s, t: (r - 1) * (r + 1) * (s - 1) * (s + 1),
        ]
        ddN22 = [
            lambda r, s, t: s * (s - 1) * (t - 1) * (t + 1),
            lambda r, s, t: (r - 1) * (r + 1) * (t - 1) * (t + 1),
            lambda r, s, t: s * (r - 1) * (r + 1) * (s - 1),
        ]
        ddN23 = [
            lambda r, s, t: (s - 1) * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: r * (r - 1) * (t - 1) * (t + 1),
            lambda r, s, t: r * (r - 1) * (s - 1) * (s + 1),
        ]
        ddN24 = [
            lambda r, s, t: (s - 1) * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: r * (r + 1) * (t - 1) * (t + 1),
            lambda r, s, t: r * (r + 1) * (s - 1) * (s + 1),
        ]
        ddN25 = [
            lambda r, s, t: s * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: (r - 1) * (r + 1) * (t - 1) * (t + 1),
            lambda r, s, t: s * (r - 1) * (r + 1) * (s + 1),
        ]
        ddN26 = [
            lambda r, s, t: t * (s - 1) * (s + 1) * (t + 1),
            lambda r, s, t: t * (r - 1) * (r + 1) * (t + 1),
            lambda r, s, t: (r - 1) * (r + 1) * (s - 1) * (s + 1),
        ]
        ddN27 = [
            lambda r, s, t: -2 * (s - 1) * (s + 1) * (t - 1) * (t + 1),
            lambda r, s, t: -2 * (r - 1) * (r + 1) * (t - 1) * (t + 1),
            lambda r, s, t: -2 * (r - 1) * (r + 1) * (s - 1) * (s + 1),
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
                ddN16,
                ddN17,
                ddN18,
                ddN19,
                ddN20,
                ddN21,
                ddN22,
                ddN23,
                ddN24,
                ddN25,
                ddN26,
                ddN27,
            ]
        )

        return ddN

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()
