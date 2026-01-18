# Copyright (C) 2021-2025 Université Gustave Eiffel.
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
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

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

    def _EulerBernoulli_N(self) -> _types.AnyArray:
        N1 = lambda r: (r - 1) ** 2 * (r + 2) / 4
        N2 = lambda r: (r - 1) ** 2 * (r + 1) / 8
        N3 = lambda r: -(r - 2) * (r + 1) ** 2 / 4
        N4 = lambda r: (r - 1) * (r + 1) ** 2 / 8

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _EulerBernoulli_dN(self) -> _types.AnyArray:
        dN1 = [lambda r: (r - 1) ** 2 / 4 + (r + 2) * (2 * r - 2) / 4]
        dN2 = [lambda r: (r - 1) ** 2 / 8 + (r + 1) * (2 * r - 2) / 8]
        dN3 = [lambda r: -(r - 2) * (2 * r + 2) / 4 - (r + 1) ** 2 / 4]
        dN4 = [lambda r: (r - 1) * (2 * r + 2) / 8 + (r + 1) ** 2 / 8]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _EulerBernoulli_ddN(self) -> _types.AnyArray:
        ddN1 = [lambda r: 3 * r / 2]
        ddN2 = [lambda r: 3 * r / 4 - 1 / 4]
        ddN3 = [lambda r: -3 * r / 2]
        ddN4 = [lambda r: 3 * r / 4 + 1 / 4]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4])

        return ddN


class SEG3(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----2----1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

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

    def _EulerBernoulli_N(self) -> _types.AnyArray:
        N1 = lambda r: r**2 * (r - 1) ** 2 * (3 * r + 4) / 4
        N2 = lambda r: r**2 * (r - 1) ** 2 * (r + 1) / 8
        N3 = lambda r: -(r**2) * (r + 1) ** 2 * (3 * r - 4) / 4
        N4 = lambda r: r**2 * (r - 1) * (r + 1) ** 2 / 8
        N5 = lambda r: (r - 1) ** 2 * (r + 1) ** 2
        N6 = lambda r: r * (r - 1) ** 2 * (r + 1) ** 2 / 2

        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _EulerBernoulli_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 3 * r**2 * (r - 1) ** 2 / 4
            + r**2 * (2 * r - 2) * (3 * r + 4) / 4
            + r * (r - 1) ** 2 * (3 * r + 4) / 2
        ]
        dN2 = [
            lambda r: r**2 * (r - 1) ** 2 / 8
            + r**2 * (r + 1) * (2 * r - 2) / 8
            + r * (r - 1) ** 2 * (r + 1) / 4
        ]
        dN3 = [
            lambda r: -3 * r**2 * (r + 1) ** 2 / 4
            - r**2 * (2 * r + 2) * (3 * r - 4) / 4
            - r * (r + 1) ** 2 * (3 * r - 4) / 2
        ]
        dN4 = [
            lambda r: r**2 * (r - 1) * (2 * r + 2) / 8
            + r**2 * (r + 1) ** 2 / 8
            + r * (r - 1) * (r + 1) ** 2 / 4
        ]
        dN5 = [lambda r: (r - 1) ** 2 * (2 * r + 2) + (r + 1) ** 2 * (2 * r - 2)]
        dN6 = [
            lambda r: r * (r - 1) ** 2 * (2 * r + 2) / 2
            + r * (r + 1) ** 2 * (2 * r - 2) / 2
            + (r - 1) ** 2 * (r + 1) ** 2 / 2
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _EulerBernoulli_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 3 * r**2 * (2 * r - 2) / 2
            + r**2 * (3 * r + 4) / 2
            + 3 * r * (r - 1) ** 2
            + r * (2 * r - 2) * (3 * r + 4)
            + (r - 1) ** 2 * (3 * r + 4) / 2
        ]
        ddN2 = [
            lambda r: r**2 * (r + 1) / 4
            + r**2 * (2 * r - 2) / 4
            + r * (r - 1) ** 2 / 2
            + r * (r + 1) * (2 * r - 2) / 2
            + (r - 1) ** 2 * (r + 1) / 4
        ]
        ddN3 = [
            lambda r: -3 * r**2 * (2 * r + 2) / 2
            - r**2 * (3 * r - 4) / 2
            - 3 * r * (r + 1) ** 2
            - r * (2 * r + 2) * (3 * r - 4)
            - (r + 1) ** 2 * (3 * r - 4) / 2
        ]
        ddN4 = [
            lambda r: r**2 * (r - 1) / 4
            + r**2 * (2 * r + 2) / 4
            + r * (r - 1) * (2 * r + 2) / 2
            + r * (r + 1) ** 2 / 2
            + (r - 1) * (r + 1) ** 2 / 4
        ]
        ddN5 = [
            lambda r: 2 * (r - 1) ** 2
            + 2 * (r + 1) ** 2
            + 2 * (2 * r - 2) * (2 * r + 2)
        ]
        ddN6 = [
            lambda r: r * (r - 1) ** 2
            + r * (r + 1) ** 2
            + r * (2 * r - 2) * (2 * r + 2)
            + (r - 1) ** 2 * (2 * r + 2)
            + (r + 1) ** 2 * (2 * r - 2)
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])

        return ddN


class SEG4(_GroupElem):
    #       v
    #       ^
    #       |
    #       |
    # 0---2-+-3---1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

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

    def _EulerBernoulli_N(self) -> _types.AnyArray:
        N1 = (
            lambda r: 891 * r**7 / 512
            - 729 * r**6 / 512
            - 275976562500001 * r**5 / 100000000000000
            + 1215 * r**4 / 512
            + 137207031250001 * r**3 / 250000000000000
            - 237304687500001 * r**2 / 500000000000000
            - 292968750000009 * r / 10000000000000000
            + 63476562500001 / 2500000000000000
        )
        N2 = (
            lambda r: 81 * r**7 / 512
            - 81 * r**6 / 512
            - 99 * r**5 / 512
            + 99 * r**4 / 512
            + 185546875000001 * r**3 / 5000000000000000
            - 185546875000001 * r**2 / 5000000000000000
            - r / 512
            + 1 / 512
        )
        N3 = (
            lambda r: -891 * r**7 / 512
            - 729 * r**6 / 512
            + 275976562500001 * r**5 / 100000000000000
            + 1215 * r**4 / 512
            - 274414062500003 * r**3 / 500000000000000
            - 243 * r**2 / 512
            + 292968750000013 * r / 10000000000000000
            + 126953125000001 / 5000000000000000
        )
        N4 = (
            lambda r: 81 * r**7 / 512
            + 81 * r**6 / 512
            - 99 * r**5 / 512
            - 99 * r**4 / 512
            + 92773437500001 * r**3 / 2500000000000000
            + 371093749999999 * r**2 / 10000000000000000
            - r / 512
            - 1 / 512
        )
        N5 = (
            lambda r: 2187 * r**7 / 512
            + 729 * r**6 / 512
            - 5589 * r**5 / 512
            - 237304687499999 * r**4 / 100000000000000
            + 901757812500001 * r**3 / 100000000000000
            + 118652343749999 * r**2 / 250000000000000
            - 1215 * r / 512
            + 243 / 512
        )
        N6 = (
            lambda r: 729 * r**7 / 512
            - 243 * r**6 / 512
            - 1539 * r**5 / 512
            + 513 * r**4 / 512
            + 891 * r**3 / 512
            - 290039062500001 * r**2 / 500000000000000
            - 81 * r / 512
            + 131835937500001 / 2500000000000000
        )
        N7 = (
            lambda r: -2187 * r**7 / 512
            + 729 * r**6 / 512
            + 5589 * r**5 / 512
            - 237304687500001 * r**4 / 100000000000000
            - 901757812500001 * r**3 / 100000000000000
            + 118652343750001 * r**2 / 250000000000000
            + 1215 * r / 512
            + 243 / 512
        )
        N8 = (
            lambda r: 729 * r**7 / 512
            + 237304687499999 * r**6 / 500000000000000
            - 300585937500001 * r**5 / 100000000000000
            - 513 * r**4 / 512
            + 891 * r**3 / 512
            + 290039062499999 * r**2 / 500000000000000
            - 81 * r / 512
            - 131835937499999 / 2500000000000000
        )

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N

    def _EulerBernoulli_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 6237 * r**6 / 512
            - 2187 * r**5 / 256
            - 275976562500001 * r**4 / 20000000000000
            + 1215 * r**3 / 128
            + 411621093750003 * r**2 / 250000000000000
            - 237304687500001 * r / 250000000000000
            - 292968750000009 / 10000000000000000
        ]
        dN2 = [
            lambda r: 567 * r**6 / 512
            - 243 * r**5 / 256
            - 495 * r**4 / 512
            + 99 * r**3 / 128
            + 556640625000003 * r**2 / 5000000000000000
            - 185546875000001 * r / 2500000000000000
            - 1 / 512
        ]
        dN3 = [
            lambda r: -6237 * r**6 / 512
            - 2187 * r**5 / 256
            + 275976562500001 * r**4 / 20000000000000
            + 1215 * r**3 / 128
            - 823242187500009 * r**2 / 500000000000000
            - 243 * r / 256
            + 292968750000013 / 10000000000000000
        ]
        dN4 = [
            lambda r: 567 * r**6 / 512
            + 243 * r**5 / 256
            - 495 * r**4 / 512
            - 99 * r**3 / 128
            + 278320312500003 * r**2 / 2500000000000000
            + 371093749999999 * r / 5000000000000000
            - 1 / 512
        ]
        dN5 = [
            lambda r: 15309 * r**6 / 512
            + 2187 * r**5 / 256
            - 27945 * r**4 / 512
            - 237304687499999 * r**3 / 25000000000000
            + 2705273437500003 * r**2 / 100000000000000
            + 118652343749999 * r / 125000000000000
            - 1215 / 512
        ]
        dN6 = [
            lambda r: 5103 * r**6 / 512
            - 729 * r**5 / 256
            - 7695 * r**4 / 512
            + 513 * r**3 / 128
            + 2673 * r**2 / 512
            - 290039062500001 * r / 250000000000000
            - 81 / 512
        ]
        dN7 = [
            lambda r: -15309 * r**6 / 512
            + 2187 * r**5 / 256
            + 27945 * r**4 / 512
            - 237304687500001 * r**3 / 25000000000000
            - 2705273437500003 * r**2 / 100000000000000
            + 118652343750001 * r / 125000000000000
            + 1215 / 512
        ]
        dN8 = [
            lambda r: 5103 * r**6 / 512
            + 711914062499997 * r**5 / 250000000000000
            - 300585937500001 * r**4 / 20000000000000
            - 513 * r**3 / 128
            + 2673 * r**2 / 512
            + 290039062499999 * r / 250000000000000
            - 81 / 512
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _EulerBernoulli_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 18711 * r**5 / 256
            - 10935 * r**4 / 256
            - 275976562500001 * r**3 / 5000000000000
            + 3645 * r**2 / 128
            + 411621093750003 * r / 125000000000000
            - 237304687500001 / 250000000000000
        ]
        ddN2 = [
            lambda r: 1701 * r**5 / 256
            - 1215 * r**4 / 256
            - 495 * r**3 / 128
            + 297 * r**2 / 128
            + 556640625000003 * r / 2500000000000000
            - 185546875000001 / 2500000000000000
        ]
        ddN3 = [
            lambda r: -18711 * r**5 / 256
            - 10935 * r**4 / 256
            + 275976562500001 * r**3 / 5000000000000
            + 3645 * r**2 / 128
            - 823242187500009 * r / 250000000000000
            - 243 / 256
        ]
        ddN4 = [
            lambda r: 1701 * r**5 / 256
            + 1215 * r**4 / 256
            - 495 * r**3 / 128
            - 297 * r**2 / 128
            + 278320312500003 * r / 1250000000000000
            + 371093749999999 / 5000000000000000
        ]
        ddN5 = [
            lambda r: 45927 * r**5 / 256
            + 10935 * r**4 / 256
            - 27945 * r**3 / 128
            - 711914062499997 * r**2 / 25000000000000
            + 2705273437500003 * r / 50000000000000
            + 118652343749999 / 125000000000000
        ]
        ddN6 = [
            lambda r: 15309 * r**5 / 256
            - 3645 * r**4 / 256
            - 7695 * r**3 / 128
            + 1539 * r**2 / 128
            + 2673 * r / 256
            - 290039062500001 / 250000000000000
        ]
        ddN7 = [
            lambda r: -45927 * r**5 / 256
            + 10935 * r**4 / 256
            + 27945 * r**3 / 128
            - 711914062500003 * r**2 / 25000000000000
            - 2705273437500003 * r / 50000000000000
            + 118652343750001 / 125000000000000
        ]
        ddN8 = [
            lambda r: 15309 * r**5 / 256
            + 711914062499997 * r**4 / 50000000000000
            - 300585937500001 * r**3 / 5000000000000
            - 1539 * r**2 / 128
            + 2673 * r / 256
            + 290039062499999 / 250000000000000
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8])

        return ddN


class SEG5(_GroupElem):
    #        v
    #        ^
    #        |
    #        |
    #  0--2--3--4--1 --> u

    def __init__(
        self, gmshId: int, connect: _types.IntArray, coordGlob: _types.FloatArray
    ):
        super().__init__(gmshId, connect, coordGlob)

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

    def _EulerBernoulli_N(self) -> _types.AnyArray:
        N1 = (
            lambda r: 100 * r**9 / 27
            - 162962962962963 * r**8 / 50000000000000
            - 58 * r**7 / 9
            + 52 * r**6 / 9
            + 91 * r**5 / 36
            - 41 * r**4 / 18
            - 143518518518519 * r**3 / 500000000000000
            + 7 * r**2 / 27
        )
        N2 = (
            lambda r: 2 * r**9 / 9
            - 2 * r**8 / 9
            - r**7 / 3
            + r**6 / 3
            + r**5 / 8
            - r**4 / 8
            - r**3 / 72
            + r**2 / 72
        )
        N3 = (
            lambda r: -100 * r**9 / 27
            - 162962962962963 * r**8 / 50000000000000
            + 58 * r**7 / 9
            + 52 * r**6 / 9
            - 91 * r**5 / 36
            - 41 * r**4 / 18
            + 143518518518519 * r**3 / 500000000000000
            + 7 * r**2 / 27
        )
        N4 = (
            lambda r: 2 * r**9 / 9
            + 2 * r**8 / 9
            - r**7 / 3
            - r**6 / 3
            + r**5 / 8
            + r**4 / 8
            - r**3 / 72
            - r**2 / 72
        )
        N5 = (
            lambda r: 640 * r**9 / 27
            - 128 * r**8 / 27
            - 544 * r**7 / 9
            + 128 * r**6 / 9
            + 448 * r**5 / 9
            - 128 * r**4 / 9
            - 352 * r**3 / 27
            + 128 * r**2 / 27
        )
        N6 = (
            lambda r: 32 * r**9 / 9
            - 16 * r**8 / 9
            - 8 * r**7
            + 4 * r**6
            + 16 * r**5 / 3
            - 8 * r**4 / 3
            - 8 * r**3 / 9
            + 4 * r**2 / 9
        )
        N7 = lambda r: 16 * r**8 - 40 * r**6 + 33 * r**4 - 10 * r**2 + 1
        N8 = lambda r: 8 * r**9 - 20 * r**7 + 33 * r**5 / 2 - 5 * r**3 + r / 2
        N9 = (
            lambda r: -640 * r**9 / 27
            - 128 * r**8 / 27
            + 544 * r**7 / 9
            + 128 * r**6 / 9
            - 448 * r**5 / 9
            - 128 * r**4 / 9
            + 352 * r**3 / 27
            + 128 * r**2 / 27
        )
        N10 = (
            lambda r: 32 * r**9 / 9
            + 16 * r**8 / 9
            - 8 * r**7
            - 4 * r**6
            + 16 * r**5 / 3
            + 8 * r**4 / 3
            - 8 * r**3 / 9
            - 4 * r**2 / 9
        )

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _EulerBernoulli_dN(self) -> _types.AnyArray:
        dN1 = [
            lambda r: 100 * r**8 / 3
            - 162962962962963 * r**7 / 6250000000000
            - 406 * r**6 / 9
            + 104 * r**5 / 3
            + 455 * r**4 / 36
            - 82 * r**3 / 9
            - 430555555555557 * r**2 / 500000000000000
            + 14 * r / 27
        ]
        dN2 = [
            lambda r: 2 * r**8
            - 16 * r**7 / 9
            - 7 * r**6 / 3
            + 2 * r**5
            + 5 * r**4 / 8
            - r**3 / 2
            - r**2 / 24
            + r / 36
        ]
        dN3 = [
            lambda r: -100 * r**8 / 3
            - 162962962962963 * r**7 / 6250000000000
            + 406 * r**6 / 9
            + 104 * r**5 / 3
            - 455 * r**4 / 36
            - 82 * r**3 / 9
            + 430555555555557 * r**2 / 500000000000000
            + 14 * r / 27
        ]
        dN4 = [
            lambda r: 2 * r**8
            + 16 * r**7 / 9
            - 7 * r**6 / 3
            - 2 * r**5
            + 5 * r**4 / 8
            + r**3 / 2
            - r**2 / 24
            - r / 36
        ]
        dN5 = [
            lambda r: 640 * r**8 / 3
            - 1024 * r**7 / 27
            - 3808 * r**6 / 9
            + 256 * r**5 / 3
            + 2240 * r**4 / 9
            - 512 * r**3 / 9
            - 352 * r**2 / 9
            + 256 * r / 27
        ]
        dN6 = [
            lambda r: 32 * r**8
            - 128 * r**7 / 9
            - 56 * r**6
            + 24 * r**5
            + 80 * r**4 / 3
            - 32 * r**3 / 3
            - 8 * r**2 / 3
            + 8 * r / 9
        ]
        dN7 = [lambda r: 128 * r**7 - 240 * r**5 + 132 * r**3 - 20 * r]
        dN8 = [lambda r: 72 * r**8 - 140 * r**6 + 165 * r**4 / 2 - 15 * r**2 + 1 / 2]
        dN9 = [
            lambda r: -640 * r**8 / 3
            - 1024 * r**7 / 27
            + 3808 * r**6 / 9
            + 256 * r**5 / 3
            - 2240 * r**4 / 9
            - 512 * r**3 / 9
            + 352 * r**2 / 9
            + 256 * r / 27
        ]
        dN10 = [
            lambda r: 32 * r**8
            + 128 * r**7 / 9
            - 56 * r**6
            - 24 * r**5
            + 80 * r**4 / 3
            + 32 * r**3 / 3
            - 8 * r**2 / 3
            - 8 * r / 9
        ]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _EulerBernoulli_ddN(self) -> _types.AnyArray:
        ddN1 = [
            lambda r: 800 * r**7 / 3
            - 1140740740740741 * r**6 / 6250000000000
            - 812 * r**5 / 3
            + 520 * r**4 / 3
            + 455 * r**3 / 9
            - 82 * r**2 / 3
            - 430555555555557 * r / 250000000000000
            + 14 / 27
        ]
        ddN2 = [
            lambda r: 16 * r**7
            - 112 * r**6 / 9
            - 14 * r**5
            + 10 * r**4
            + 5 * r**3 / 2
            - 3 * r**2 / 2
            - r / 12
            + 1 / 36
        ]
        ddN3 = [
            lambda r: -800 * r**7 / 3
            - 1140740740740741 * r**6 / 6250000000000
            + 812 * r**5 / 3
            + 520 * r**4 / 3
            - 455 * r**3 / 9
            - 82 * r**2 / 3
            + 430555555555557 * r / 250000000000000
            + 14 / 27
        ]
        ddN4 = [
            lambda r: 16 * r**7
            + 112 * r**6 / 9
            - 14 * r**5
            - 10 * r**4
            + 5 * r**3 / 2
            + 3 * r**2 / 2
            - r / 12
            - 1 / 36
        ]
        ddN5 = [
            lambda r: 5120 * r**7 / 3
            - 7168 * r**6 / 27
            - 7616 * r**5 / 3
            + 1280 * r**4 / 3
            + 8960 * r**3 / 9
            - 512 * r**2 / 3
            - 704 * r / 9
            + 256 / 27
        ]
        ddN6 = [
            lambda r: 256 * r**7
            - 896 * r**6 / 9
            - 336 * r**5
            + 120 * r**4
            + 320 * r**3 / 3
            - 32 * r**2
            - 16 * r / 3
            + 8 / 9
        ]
        ddN7 = [lambda r: 896 * r**6 - 1200 * r**4 + 396 * r**2 - 20]
        ddN8 = [lambda r: 576 * r**7 - 840 * r**5 + 330 * r**3 - 30 * r]
        ddN9 = [
            lambda r: -5120 * r**7 / 3
            - 7168 * r**6 / 27
            + 7616 * r**5 / 3
            + 1280 * r**4 / 3
            - 8960 * r**3 / 9
            - 512 * r**2 / 3
            + 704 * r / 9
            + 256 / 27
        ]
        ddN10 = [
            lambda r: 256 * r**7
            + 896 * r**6 / 9
            - 336 * r**5
            - 120 * r**4
            + 320 * r**3 / 3
            + 32 * r**2
            - 16 * r / 3
            - 8 / 9
        ]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN
