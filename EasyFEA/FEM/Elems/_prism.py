# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Prism element module."""

import numpy as np

from .._group_elem import _GroupElem
from ...Utilities import _types


class PRISM6(_GroupElem):
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        ,/  |  `\
    #      ,/    |    `\
    #     4------+------5
    #     |      |      |
    #     |    ,/|`\    |
    #     |  ,/  |  `\  |
    #     |,/    |    `\|
    #    ,|      |      |\
    #  ,/ |      0      | `\
    # u   |    ,/ `\    |    v
    #     |  ,/     `\  |
    #     |,/         `\|
    #     1-------------2

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1, 4, 3],
                [0, 3, 5, 2],
                [1, 2, 5, 4],
                [3, 4, 5, 3],  # tri z=1
                [0, 2, 1, 0],  # tri z=-1
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1, 4, 3],
                [0, 3, 5, 2],
                [1, 2, 5, 4],
                [3, 4, 5],  # tri z=1
                [0, 2, 1],  # tri z=-1
            ],
            dtype=object,
        )

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0, 1, 0]
        list_y = [0, 0, 1, 0, 0, 1]
        list_z = [-1, -1, -1, 1, 1, 1]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1],
                [2, 0],
                [0, 3],
                [1, 2],
                [1, 4],
                [2, 5],
                [3, 4],
                [5, 3],
                [4, 5],
            ],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: (t - 1) * (r + s - 1) / 2
        N2 = lambda r, s, t: -r * (t - 1) / 2
        N3 = lambda r, s, t: -s * (t - 1) / 2
        N4 = lambda r, s, t: -(t + 1) * (r + s - 1) / 2
        N5 = lambda r, s, t: r * (t + 1) / 2
        N6 = lambda r, s, t: s * (t + 1) / 2

        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: t / 2 - 1 / 2,
            lambda r, s, t: t / 2 - 1 / 2,
            lambda r, s, t: r / 2 + s / 2 - 1 / 2,
        ]
        dN2 = [lambda r, s, t: 1 / 2 - t / 2, lambda r, s, t: 0, lambda r, s, t: -r / 2]
        dN3 = [lambda r, s, t: 0, lambda r, s, t: 1 / 2 - t / 2, lambda r, s, t: -s / 2]
        dN4 = [
            lambda r, s, t: -t / 2 - 1 / 2,
            lambda r, s, t: -t / 2 - 1 / 2,
            lambda r, s, t: -r / 2 - s / 2 + 1 / 2,
        ]
        dN5 = [lambda r, s, t: t / 2 + 1 / 2, lambda r, s, t: 0, lambda r, s, t: r / 2]
        dN6 = [lambda r, s, t: 0, lambda r, s, t: t / 2 + 1 / 2, lambda r, s, t: s / 2]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _ddN(self) -> _types.FloatArray:
        return super()._ddN()

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class PRISM15(_GroupElem):
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        12  |  13
    #      ,/    |    `\
    #     4------14-----5
    #     |      8      |
    #     |    ,/|`\    |
    #     |  ,/  |  `\  |
    #     |,/    |    `\|
    #    ,10      |     11
    #  ,/ |      0      | \
    # u   |    ,/ `\    |   v
    #     |  ,6     `7  |
    #     |,/         `\|
    #     1------9------2

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 6, 1, 10, 4, 12, 3, 8],  # 15
                [0, 8, 3, 13, 5, 11, 2, 7],  # 16
                [1, 9, 2, 11, 5, 14, 4, 10],  # 17
                [3, 12, 4, 14, 5, 13, 3, 3],  # tri z=1
                [0, 7, 2, 9, 1, 6, 0, 0],  # tri z=-1
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1, 4, 3, 6, 10, 12, 8],  # 15
                [0, 3, 5, 2, 8, 13, 11, 7],  # 16
                [1, 2, 5, 4, 9, 11, 14, 10],  # 17
                [3, 4, 5, 12, 14, 13],  # tri z=1
                [0, 2, 1, 7, 9, 6],  # tri z=-1
            ],
            dtype=object,
        )

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 0.5]
        list_y = [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0.5]
        list_z = [-1, -1, -1, 1, 1, 1, -1, -1, 0, -1, 0, 0, 1, 1, 1]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 6, 1],
                [2, 7, 0],
                [0, 8, 3],
                [1, 9, 2],
                [1, 10, 4],
                [2, 11, 5],
                [3, 12, 4],
                [5, 13, 3],
                [4, 14, 5],
            ],
            dtype=int,
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: -(t - 1) * (r + s - 1) * (2 * r + 2 * s + t) / 2
        N2 = lambda r, s, t: -r * (t - 1) * (2 * r - t - 2) / 2
        N3 = lambda r, s, t: -s * (t - 1) * (2 * s - t - 2) / 2
        N4 = lambda r, s, t: (t + 1) * (r + s - 1) * (2 * r + 2 * s - t) / 2
        N5 = lambda r, s, t: r * (t + 1) * (2 * r + t - 2) / 2
        N6 = lambda r, s, t: s * (t + 1) * (2 * s + t - 2) / 2
        N7 = lambda r, s, t: 2 * r * (t - 1) * (r + s - 1)
        N8 = lambda r, s, t: 2 * s * (t - 1) * (r + s - 1)
        N9 = lambda r, s, t: (t - 1) * (t + 1) * (r + s - 1)
        N10 = lambda r, s, t: -2 * r * s * (t - 1)
        N11 = lambda r, s, t: -r * (t - 1) * (t + 1)
        N12 = lambda r, s, t: -s * (t - 1) * (t + 1)
        N13 = lambda r, s, t: -2 * r * (t + 1) * (r + s - 1)
        N14 = lambda r, s, t: -2 * s * (t + 1) * (r + s - 1)
        N15 = lambda r, s, t: 2 * r * s * (t + 1)

        N = np.array(
            [N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15]
        ).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: -(t - 1) * (r + s - 1) - (t - 1) * (2 * r + 2 * s + t) / 2,
            lambda r, s, t: -(t - 1) * (r + s - 1) - (t - 1) * (2 * r + 2 * s + t) / 2,
            lambda r, s, t: -(t - 1) * (r + s - 1) / 2
            - (r + s - 1) * (2 * r + 2 * s + t) / 2,
        ]
        dN2 = [
            lambda r, s, t: -r * (t - 1) - (t - 1) * (2 * r - t - 2) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: r * (t - 1) / 2 - r * (2 * r - t - 2) / 2,
        ]
        dN3 = [
            lambda r, s, t: 0,
            lambda r, s, t: -s * (t - 1) - (t - 1) * (2 * s - t - 2) / 2,
            lambda r, s, t: s * (t - 1) / 2 - s * (2 * s - t - 2) / 2,
        ]
        dN4 = [
            lambda r, s, t: (t + 1) * (r + s - 1) + (t + 1) * (2 * r + 2 * s - t) / 2,
            lambda r, s, t: (t + 1) * (r + s - 1) + (t + 1) * (2 * r + 2 * s - t) / 2,
            lambda r, s, t: -(t + 1) * (r + s - 1) / 2
            + (r + s - 1) * (2 * r + 2 * s - t) / 2,
        ]
        dN5 = [
            lambda r, s, t: r * (t + 1) + (t + 1) * (2 * r + t - 2) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: r * (t + 1) / 2 + r * (2 * r + t - 2) / 2,
        ]
        dN6 = [
            lambda r, s, t: 0,
            lambda r, s, t: s * (t + 1) + (t + 1) * (2 * s + t - 2) / 2,
            lambda r, s, t: s * (t + 1) / 2 + s * (2 * s + t - 2) / 2,
        ]
        dN7 = [
            lambda r, s, t: 2 * r * (t - 1) + 2 * (t - 1) * (r + s - 1),
            lambda r, s, t: 2 * r * (t - 1),
            lambda r, s, t: 2 * r * (r + s - 1),
        ]
        dN8 = [
            lambda r, s, t: 2 * s * (t - 1),
            lambda r, s, t: 2 * s * (t - 1) + 2 * (t - 1) * (r + s - 1),
            lambda r, s, t: 2 * s * (r + s - 1),
        ]
        dN9 = [
            lambda r, s, t: (t - 1) * (t + 1),
            lambda r, s, t: (t - 1) * (t + 1),
            lambda r, s, t: (t - 1) * (r + s - 1) + (t + 1) * (r + s - 1),
        ]
        dN10 = [
            lambda r, s, t: -2 * s * (t - 1),
            lambda r, s, t: -2 * r * (t - 1),
            lambda r, s, t: -2 * r * s,
        ]
        dN11 = [
            lambda r, s, t: -(t - 1) * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: -r * (t - 1) - r * (t + 1),
        ]
        dN12 = [
            lambda r, s, t: 0,
            lambda r, s, t: -(t - 1) * (t + 1),
            lambda r, s, t: -s * (t - 1) - s * (t + 1),
        ]
        dN13 = [
            lambda r, s, t: -2 * r * (t + 1) - 2 * (t + 1) * (r + s - 1),
            lambda r, s, t: -2 * r * (t + 1),
            lambda r, s, t: -2 * r * (r + s - 1),
        ]
        dN14 = [
            lambda r, s, t: -2 * s * (t + 1),
            lambda r, s, t: -2 * s * (t + 1) - 2 * (t + 1) * (r + s - 1),
            lambda r, s, t: -2 * s * (r + s - 1),
        ]
        dN15 = [
            lambda r, s, t: 2 * s * (t + 1),
            lambda r, s, t: 2 * r * (t + 1),
            lambda r, s, t: 2 * r * s,
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
            lambda r, s, t: 2 - 2 * t,
            lambda r, s, t: 2 - 2 * t,
            lambda r, s, t: -r - s + 1,
        ]
        ddN2 = [lambda r, s, t: 2 - 2 * t, lambda r, s, t: 0, lambda r, s, t: r]
        ddN3 = [lambda r, s, t: 0, lambda r, s, t: 2 - 2 * t, lambda r, s, t: s]
        ddN4 = [
            lambda r, s, t: 2 * t + 2,
            lambda r, s, t: 2 * t + 2,
            lambda r, s, t: -r - s + 1,
        ]
        ddN5 = [lambda r, s, t: 2 * t + 2, lambda r, s, t: 0, lambda r, s, t: r]
        ddN6 = [lambda r, s, t: 0, lambda r, s, t: 2 * t + 2, lambda r, s, t: s]
        ddN7 = [lambda r, s, t: 4 * t - 4, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN8 = [lambda r, s, t: 0, lambda r, s, t: 4 * t - 4, lambda r, s, t: 0]
        ddN9 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 2 * r + 2 * s - 2]
        ddN10 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN11 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: -2 * r]
        ddN12 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: -2 * s]
        ddN13 = [lambda r, s, t: -4 * t - 4, lambda r, s, t: 0, lambda r, s, t: 0]
        ddN14 = [lambda r, s, t: 0, lambda r, s, t: -4 * t - 4, lambda r, s, t: 0]
        ddN15 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 0]

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
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()


class PRISM18(_GroupElem):
    #            w
    #            ^
    #            |
    #            3
    #          ,/|`\
    #        12  |  13
    #      ,/    |    `\
    #     4------14-----5
    #     |      8      |
    #     |    ,/|`\    |
    #     |  15  |  16  |
    #     |,/    |    `\|
    #    ,10-----17-----11
    #  ,/ |      0      | \
    # u   |    ,/ `\    |   v
    #     |  ,6     `7  |
    #     |,/         `\|
    #     1------9------2

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

    @property
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def surfaces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 6, 1, 10, 4, 12, 3, 8],  # 15
                [0, 8, 3, 13, 5, 11, 2, 7],  # 16
                [1, 9, 2, 11, 5, 14, 4, 10],  # 17
                [3, 12, 4, 14, 5, 13, 3, 3],  # tri z=1
                [0, 7, 2, 9, 1, 6, 0, 0],  # tri z=-1
            ],
            dtype=int,
        )

    @property
    def faces(self) -> _types.IntArray:
        return np.array(
            [
                [0, 1, 4, 3, 6, 10, 12, 8, 15],
                [0, 3, 5, 2, 8, 13, 11, 7, 16],
                [1, 2, 5, 4, 9, 11, 14, 10, 17],
                [3, 4, 5, 12, 14, 13],  # tri z=1
                [0, 2, 1, 7, 9, 6],  # tri z=-1
            ],
            dtype=object,
        )

    def Get_Local_Coords(self):
        list_x = [0, 1, 0, 0, 1, 0, 0.5, 0, 0, 0.5, 1, 0, 0.5, 0, 0.5, 0.5, 0, 0.5]
        list_y = [0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5, 0, 1, 0, 0.5, 0.5, 0, 0.5, 0.5]
        list_z = [-1, -1, -1, 1, 1, 1, -1, -1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0]
        local_coords = np.array([list_x, list_y, list_z]).T
        return local_coords

    @property
    def segments(self) -> _types.IntArray:
        return np.array(
            [
                [0, 6, 1],
                [2, 7, 0],
                [0, 8, 3],
                [1, 9, 2],
                [1, 10, 4],
                [2, 11, 5],
                [3, 12, 4],
                [5, 13, 3],
                [4, 14, 5],
            ]
        )

    def _N(self) -> _types.FloatArray:
        N1 = lambda r, s, t: t * (t - 1) * (r + s - 1) * (2 * r + 2 * s - 1) / 2
        N2 = lambda r, s, t: r * t * (2 * r - 1) * (t - 1) / 2
        N3 = lambda r, s, t: s * t * (2 * s - 1) * (t - 1) / 2
        N4 = lambda r, s, t: t * (t + 1) * (r + s - 1) * (2 * r + 2 * s - 1) / 2
        N5 = lambda r, s, t: r * t * (2 * r - 1) * (t + 1) / 2
        N6 = lambda r, s, t: s * t * (2 * s - 1) * (t + 1) / 2
        N7 = lambda r, s, t: -2 * r * t * (t - 1) * (r + s - 1)
        N8 = lambda r, s, t: -2 * s * t * (t - 1) * (r + s - 1)
        N9 = lambda r, s, t: -(t - 1) * (t + 1) * (r + s - 1) * (2 * r + 2 * s - 1)
        N10 = lambda r, s, t: 2 * r * s * t * (t - 1)
        N11 = lambda r, s, t: -r * (2 * r - 1) * (t - 1) * (t + 1)
        N12 = lambda r, s, t: -s * (2 * s - 1) * (t - 1) * (t + 1)
        N13 = lambda r, s, t: -2 * r * t * (t + 1) * (r + s - 1)
        N14 = lambda r, s, t: -2 * s * t * (t + 1) * (r + s - 1)
        N15 = lambda r, s, t: 2 * r * s * t * (t + 1)
        N16 = lambda r, s, t: 4 * r * (t - 1) * (t + 1) * (r + s - 1)
        N17 = lambda r, s, t: 4 * s * (t - 1) * (t + 1) * (r + s - 1)
        N18 = lambda r, s, t: -4 * r * s * (t - 1) * (t + 1)

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
            ]
        ).reshape(-1, 1)

        return N

    def _dN(self) -> _types.FloatArray:
        dN1 = [
            lambda r, s, t: t * (t - 1) * (r + s - 1)
            + t * (t - 1) * (2 * r + 2 * s - 1) / 2,
            lambda r, s, t: t * (t - 1) * (r + s - 1)
            + t * (t - 1) * (2 * r + 2 * s - 1) / 2,
            lambda r, s, t: t * (r + s - 1) * (2 * r + 2 * s - 1) / 2
            + (t - 1) * (r + s - 1) * (2 * r + 2 * s - 1) / 2,
        ]
        dN2 = [
            lambda r, s, t: r * t * (t - 1) + t * (2 * r - 1) * (t - 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: r * t * (2 * r - 1) / 2 + r * (2 * r - 1) * (t - 1) / 2,
        ]
        dN3 = [
            lambda r, s, t: 0,
            lambda r, s, t: s * t * (t - 1) + t * (2 * s - 1) * (t - 1) / 2,
            lambda r, s, t: s * t * (2 * s - 1) / 2 + s * (2 * s - 1) * (t - 1) / 2,
        ]
        dN4 = [
            lambda r, s, t: t * (t + 1) * (r + s - 1)
            + t * (t + 1) * (2 * r + 2 * s - 1) / 2,
            lambda r, s, t: t * (t + 1) * (r + s - 1)
            + t * (t + 1) * (2 * r + 2 * s - 1) / 2,
            lambda r, s, t: t * (r + s - 1) * (2 * r + 2 * s - 1) / 2
            + (t + 1) * (r + s - 1) * (2 * r + 2 * s - 1) / 2,
        ]
        dN5 = [
            lambda r, s, t: r * t * (t + 1) + t * (2 * r - 1) * (t + 1) / 2,
            lambda r, s, t: 0,
            lambda r, s, t: r * t * (2 * r - 1) / 2 + r * (2 * r - 1) * (t + 1) / 2,
        ]
        dN6 = [
            lambda r, s, t: 0,
            lambda r, s, t: s * t * (t + 1) + t * (2 * s - 1) * (t + 1) / 2,
            lambda r, s, t: s * t * (2 * s - 1) / 2 + s * (2 * s - 1) * (t + 1) / 2,
        ]
        dN7 = [
            lambda r, s, t: -2 * r * t * (t - 1) - 2 * t * (t - 1) * (r + s - 1),
            lambda r, s, t: -2 * r * t * (t - 1),
            lambda r, s, t: -2 * r * t * (r + s - 1) - 2 * r * (t - 1) * (r + s - 1),
        ]
        dN8 = [
            lambda r, s, t: -2 * s * t * (t - 1),
            lambda r, s, t: -2 * s * t * (t - 1) - 2 * t * (t - 1) * (r + s - 1),
            lambda r, s, t: -2 * s * t * (r + s - 1) - 2 * s * (t - 1) * (r + s - 1),
        ]
        dN9 = [
            lambda r, s, t: -2 * (t - 1) * (t + 1) * (r + s - 1)
            - (t - 1) * (t + 1) * (2 * r + 2 * s - 1),
            lambda r, s, t: -2 * (t - 1) * (t + 1) * (r + s - 1)
            - (t - 1) * (t + 1) * (2 * r + 2 * s - 1),
            lambda r, s, t: -(t - 1) * (r + s - 1) * (2 * r + 2 * s - 1)
            - (t + 1) * (r + s - 1) * (2 * r + 2 * s - 1),
        ]
        dN10 = [
            lambda r, s, t: 2 * s * t * (t - 1),
            lambda r, s, t: 2 * r * t * (t - 1),
            lambda r, s, t: 2 * r * s * t + 2 * r * s * (t - 1),
        ]
        dN11 = [
            lambda r, s, t: -2 * r * (t - 1) * (t + 1)
            - (2 * r - 1) * (t - 1) * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: -r * (2 * r - 1) * (t - 1) - r * (2 * r - 1) * (t + 1),
        ]
        dN12 = [
            lambda r, s, t: 0,
            lambda r, s, t: -2 * s * (t - 1) * (t + 1)
            - (2 * s - 1) * (t - 1) * (t + 1),
            lambda r, s, t: -s * (2 * s - 1) * (t - 1) - s * (2 * s - 1) * (t + 1),
        ]
        dN13 = [
            lambda r, s, t: -2 * r * t * (t + 1) - 2 * t * (t + 1) * (r + s - 1),
            lambda r, s, t: -2 * r * t * (t + 1),
            lambda r, s, t: -2 * r * t * (r + s - 1) - 2 * r * (t + 1) * (r + s - 1),
        ]
        dN14 = [
            lambda r, s, t: -2 * s * t * (t + 1),
            lambda r, s, t: -2 * s * t * (t + 1) - 2 * t * (t + 1) * (r + s - 1),
            lambda r, s, t: -2 * s * t * (r + s - 1) - 2 * s * (t + 1) * (r + s - 1),
        ]
        dN15 = [
            lambda r, s, t: 2 * s * t * (t + 1),
            lambda r, s, t: 2 * r * t * (t + 1),
            lambda r, s, t: 2 * r * s * t + 2 * r * s * (t + 1),
        ]
        dN16 = [
            lambda r, s, t: 4 * r * (t - 1) * (t + 1)
            + 4 * (t - 1) * (t + 1) * (r + s - 1),
            lambda r, s, t: 4 * r * (t - 1) * (t + 1),
            lambda r, s, t: 4 * r * (t - 1) * (r + s - 1)
            + 4 * r * (t + 1) * (r + s - 1),
        ]
        dN17 = [
            lambda r, s, t: 4 * s * (t - 1) * (t + 1),
            lambda r, s, t: 4 * s * (t - 1) * (t + 1)
            + 4 * (t - 1) * (t + 1) * (r + s - 1),
            lambda r, s, t: 4 * s * (t - 1) * (r + s - 1)
            + 4 * s * (t + 1) * (r + s - 1),
        ]
        dN18 = [
            lambda r, s, t: -4 * s * (t - 1) * (t + 1),
            lambda r, s, t: -4 * r * (t - 1) * (t + 1),
            lambda r, s, t: -4 * r * s * (t - 1) - 4 * r * s * (t + 1),
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
            ]
        )

        return dN

    def _ddN(self) -> _types.FloatArray:
        ddN1 = [
            lambda r, s, t: 2 * t * (t - 1),
            lambda r, s, t: 2 * t * (t - 1),
            lambda r, s, t: (r + s - 1) * (2 * r + 2 * s - 1),
        ]
        ddN2 = [
            lambda r, s, t: 2 * t * (t - 1),
            lambda r, s, t: 0,
            lambda r, s, t: r * (2 * r - 1),
        ]
        ddN3 = [
            lambda r, s, t: 0,
            lambda r, s, t: 2 * t * (t - 1),
            lambda r, s, t: s * (2 * s - 1),
        ]
        ddN4 = [
            lambda r, s, t: 2 * t * (t + 1),
            lambda r, s, t: 2 * t * (t + 1),
            lambda r, s, t: (r + s - 1) * (2 * r + 2 * s - 1),
        ]
        ddN5 = [
            lambda r, s, t: 2 * t * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: r * (2 * r - 1),
        ]
        ddN6 = [
            lambda r, s, t: 0,
            lambda r, s, t: 2 * t * (t + 1),
            lambda r, s, t: s * (2 * s - 1),
        ]
        ddN7 = [
            lambda r, s, t: -4 * t * (t - 1),
            lambda r, s, t: 0,
            lambda r, s, t: -4 * r * (r + s - 1),
        ]
        ddN8 = [
            lambda r, s, t: 0,
            lambda r, s, t: -4 * t * (t - 1),
            lambda r, s, t: -4 * s * (r + s - 1),
        ]
        ddN9 = [
            lambda r, s, t: -4 * (t - 1) * (t + 1),
            lambda r, s, t: -4 * (t - 1) * (t + 1),
            lambda r, s, t: -2 * (r + s - 1) * (2 * r + 2 * s - 1),
        ]
        ddN10 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 4 * r * s]
        ddN11 = [
            lambda r, s, t: -4 * (t - 1) * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: -2 * r * (2 * r - 1),
        ]
        ddN12 = [
            lambda r, s, t: 0,
            lambda r, s, t: -4 * (t - 1) * (t + 1),
            lambda r, s, t: -2 * s * (2 * s - 1),
        ]
        ddN13 = [
            lambda r, s, t: -4 * t * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: -4 * r * (r + s - 1),
        ]
        ddN14 = [
            lambda r, s, t: 0,
            lambda r, s, t: -4 * t * (t + 1),
            lambda r, s, t: -4 * s * (r + s - 1),
        ]
        ddN15 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: 4 * r * s]
        ddN16 = [
            lambda r, s, t: 8 * (t - 1) * (t + 1),
            lambda r, s, t: 0,
            lambda r, s, t: 8 * r * (r + s - 1),
        ]
        ddN17 = [
            lambda r, s, t: 0,
            lambda r, s, t: 8 * (t - 1) * (t + 1),
            lambda r, s, t: 8 * s * (r + s - 1),
        ]
        ddN18 = [lambda r, s, t: 0, lambda r, s, t: 0, lambda r, s, t: -8 * r * s]

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
            ]
        )

        return ddN

    def _dddN(self) -> _types.FloatArray:
        return super()._dddN()

    def _ddddN(self) -> _types.FloatArray:
        return super()._ddddN()
