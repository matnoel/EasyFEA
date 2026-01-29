# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Tetra element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class TETRA4(_GroupElem):
    #                    v
    #                  .
    #                ,/
    #               /
    #            2
    #          ,/|`\
    #        ,/  |  `\
    #      ,/    '.   `\
    #    ,/       |     `\
    #  ,/         |       `\
    # 0-----------'.--------1 --> u
    #  `\.         |      ,/
    #     `\.      |    ,/
    #        `\.   '. ,/
    #           `\. |/
    #              `3
    #                 `\.
    #                    ` w

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
        return np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [0, 1, 3],
                [1, 2, 3],
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return self.surfaces

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0]
        list_y = [0, 0, 1, 0]
        list_z = [0, 0, 0, 1]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [[0, 1], [2, 1], [2, 0], [0, 3], [2, 3], [3, 1]],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: -r - s - t + 1
        N2 = lambda r, s, t: r
        N3 = lambda r, s, t: s
        N4 = lambda r, s, t: t

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [lambda r, s, t: -1, lambda r, s, t: -1, lambda r, s, t: -1]
        dN2 = [lambda r, s, t: 1, lambda r, s, t: 0, lambda r, s, t: 0]
        dN3 = [lambda r, s, t: 0, lambda r, s, t: 1, lambda r, s, t: 0]
        dN4 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 1]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _ddN(self) -> _types.FloatArray:
        return super()._ddN()

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class TETRA10(_GroupElem):
    #                    v
    #                  .
    #                ,/
    #               /
    #            2
    #          ,/|`\
    #        ,/  |  `\
    #      ,6    '.   `5
    #    ,/       8     `\
    #  ,/         |       `\
    # 0--------4--'.--------1 --> u
    #  `\.         |      ,/
    #     `\.      |    ,9
    #        `7.   '. ,/
    #           `\. |/
    #              `3
    #                 `\.
    #                    ` w

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
        return np.array(
            [
                [0, 6, 2, 5, 1, 4],
                [0, 7, 3, 8, 2, 6],
                [0, 4, 1, 9, 3, 7],
                [1, 5, 2, 8, 3, 9],
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 2, 1, 6, 5, 4],
                [0, 3, 2, 7, 8, 6],
                [0, 1, 3, 4, 9, 7],
                [1, 2, 3, 5, 8, 9],
            ],
            dtype=int,
        )

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0, 0.5, 0.5, 0, 0, 0, 0.5]
        list_y = [0, 0, 1, 0, 0, 0.5, 0.5, 0, 0.5, 0]
        list_z = [0, 0, 0, 1, 0, 0, 0, 0.5, 0.5, 0.5]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [[0, 4, 1], [2, 5, 1], [2, 6, 0], [0, 7, 3], [2, 8, 3], [3, 9, 1]],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: (r + s + t - 1) * (2 * r + 2 * s + 2 * t - 1)
        N2 = lambda r, s, t: r * (2 * r - 1)
        N3 = lambda r, s, t: s * (2 * s - 1)
        N4 = lambda r, s, t: t * (2 * t - 1)
        N5 = lambda r, s, t: -4 * r * (r + s + t - 1)
        N6 = lambda r, s, t: 4 * r * s
        N7 = lambda r, s, t: -4 * s * (r + s + t - 1)
        N8 = lambda r, s, t: -4 * t * (r + s + t - 1)
        N9 = lambda r, s, t: 4 * s * t
        N10 = lambda r, s, t: 4 * r * t

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: 4 * r + 4 * s + 4 * t - 3,
            lambda r, s, t: 4 * r + 4 * s + 4 * t - 3,
            lambda r, s, t: 4 * r + 4 * s + 4 * t - 3,
        ]
        dN2 = [lambda r, s, t: 4 * r - 1, lambda r, s, t: 0, lambda r, s, t: 0]
        dN3 = [lambda r, s, t: 0, lambda r, s, t: 4 * s - 1, lambda r, s, t: 0]
        dN4 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 4 * t - 1]
        dN5 = [
            lambda r, s, t: -8 * r - 4 * s - 4 * t + 4,
            lambda r, s, t: -4 * r,
            lambda r, s, t: -4 * r,
        ]
        dN6 = [lambda r, s, t: 4 * s, lambda r, s, t: 4 * r, lambda r, s, t: 0]
        dN7 = [
            lambda r, s, t: -4 * s,
            lambda r, s, t: -4 * r - 8 * s - 4 * t + 4,
            lambda r, s, t: -4 * s,
        ]
        dN8 = [
            lambda r, s, t: -4 * t,
            lambda r, s, t: -4 * t,
            lambda r, s, t: -4 * r - 4 * s - 8 * t + 4,
        ]
        dN9 = [lambda r, s, t: 0, lambda r, s, t: 4 * t, lambda r, s, t: 4 * s]
        dN10 = [lambda r, s, t: 4 * t, lambda r, s, t: 0, lambda r, s, t: 4 * r]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [lambda r, s, t: 4, lambda r, s, t: 4, lambda r, s, t: 4]
        ddN2 = [lambda r, s, t: 4, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN3 = [lambda r, s, t: 0, lambda r, s, t: 4, lambda r, s, t: 0]
        ddN4 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 4]
        ddN5 = [lambda r, s, t: -8, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN6 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN7 = [lambda r, s, t: 0, lambda r, s, t: -8, lambda r, s, t: 0]
        ddN8 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: -8]
        ddN9 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN10 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 0]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()
