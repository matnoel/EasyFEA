# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Tri element module."""

import numpy as np

from .._group_elems import _GroupElem

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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0,1,2]

    @property
    def faces(self) -> list[int]:
        return [0,1,2,0]    

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : -r - s + 1
        N2 = lambda r, s : r
        N3 = lambda r, s : s
        
        N = np.array([N1, N2, N3]).reshape(-1,1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : -1, lambda r, s : -1]
        dN2 = [lambda r, s : 1,  lambda r, s : 0]
        dN3 = [lambda r, s : 0,  lambda r, s : 1]

        dN = np.array([dN1, dN2, dN3])

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0,3,5,
                3,1,4,
                5,4,2,
                3,4,5]

    @property
    def faces(self) -> list[int]:
        return [0,3,1,4,2,5,0]

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : (r + s - 1)*(2*r + 2*s - 1)
        N2 = lambda r, s : r*(2*r - 1)
        N3 = lambda r, s : s*(2*s - 1)
        N4 = lambda r, s : -4*r*(r + s - 1)
        N5 = lambda r, s : 4*r*s
        N6 = lambda r, s : -4*s*(r + s - 1)
        
        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : 4*r + 4*s - 3,  lambda r, s : 4*r + 4*s - 3]
        dN2 = [lambda r, s : 4*r - 1,        lambda r, s : 0]
        dN3 = [lambda r, s : 0,              lambda r, s : 4*s - 1]
        dN4 = [lambda r, s : -8*r - 4*s + 4, lambda r, s : -4*r]
        dN5 = [lambda r, s : 4*s,            lambda r, s : 4*r]
        dN6 = [lambda r, s : -4*s,           lambda r, s : -4*r - 8*s + 4]
        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _ddN(self) -> np.ndarray:
        ddN1 = [lambda r, s : 4,  lambda r, s : 4]
        ddN2 = [lambda r, s : 4,  lambda r, s : 0]
        ddN3 = [lambda r, s : 0,  lambda r, s : 4]
        ddN4 = [lambda r, s : -8, lambda r, s : 0]
        ddN5 = [lambda r, s : 0,  lambda r, s : 0]
        ddN6 = [lambda r, s : 0,  lambda r, s : -8]
        
        ddNtild = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])

        return ddNtild

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return [0,3,8,
                3,4,9,
                4,1,5,
                4,5,9,
                3,9,8,
                8,9,7,
                9,5,6,
                9,6,7,
                7,6,2]
    
    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,5,6,2,7,8,0]

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : -9*r**3/2 - 27*r**2*s/2 + 9*r**2 - 27*r*s**2/2 + 18*r*s - 11*r/2 - 9*s**3/2 + 9*s**2 - 11*s/2 + 1
        N2 = lambda r, s : 9*r**3/2 - 9*r**2/2 + r
        N3 = lambda r, s : 9*s**3/2 - 9*s**2/2 + s
        N4 = lambda r, s : 27*r**3/2 + 27*r**2*s - 45*r**2/2 + 27*r*s**2/2 - 45*r*s/2 + 9*r
        N5 = lambda r, s : -27*r**3/2 - 27*r**2*s/2 + 18*r**2 + 9*r*s/2 - 9*r/2
        N6 = lambda r, s : 27*r**2*s/2 - 9*r*s/2
        N7 = lambda r, s : 27*r*s**2/2 - 9*r*s/2
        N8 = lambda r, s : -27*r*s**2/2 + 9*r*s/2 - 27*s**3/2 + 18*s**2 - 9*s/2
        N9 = lambda r, s : 27*r**2*s/2 + 27*r*s**2 - 45*r*s/2 + 27*s**3/2 - 45*s**2/2 + 9*s
        N10 = lambda r, s : -27*r**2*s - 27*r*s**2 + 27*r*s
        
        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : -27*r**2/2 - 27*r*s + 18*r - 27*s**2/2 + 18*s - 11/2,
               lambda r, s : -27*r**2/2 - 27*r*s + 18*r - 27*s**2/2 + 18*s - 11/2]
        dN2 = [lambda r, s : 27*r**2/2 - 9*r + 1,
               lambda r, s : 0]
        dN3 = [lambda r, s : 0,
               lambda r, s : 27*s**2/2 - 9*s + 1]
        dN4 = [lambda r, s : 81*r**2/2 + 54*r*s - 45*r + 27*s**2/2 - 45*s/2 + 9,
               lambda r, s : 27*r**2 + 27*r*s - 45*r/2]
        dN5 = [lambda r, s : -81*r**2/2 - 27*r*s + 36*r + 9*s/2 - 9/2,
               lambda r, s : -27*r**2/2 + 9*r/2]
        dN6 = [lambda r, s : 27*r*s - 9*s/2,
               lambda r, s : 27*r**2/2 - 9*r/2]
        dN7 = [lambda r, s : 27*s**2/2 - 9*s/2,
               lambda r, s : 27*r*s - 9*r/2]
        dN8 = [lambda r, s : -27*s**2/2 + 9*s/2,
               lambda r, s : -27*r*s + 9*r/2 - 81*s**2/2 + 36*s - 9/2]
        dN9 = [lambda r, s : 27*r*s + 27*s**2 - 45*s/2,
               lambda r, s : 27*r**2/2 + 54*r*s - 45*r/2 + 81*s**2/2 - 45*s + 9]
        dN10 = [lambda r, s : -54*r*s - 27*s**2 + 27*s,
               lambda r, s : -27*r**2 - 54*r*s + 27*r]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda r, s : -27*r - 27*s + 18,    lambda r, s : -27*r - 27*s + 18]
        ddN2 = [lambda r, s : 27*r - 9,             lambda r, s : 0]
        ddN3 = [lambda r, s : 0,                    lambda r, s : 27*s - 9]
        ddN4 = [lambda r, s : 81*r + 54*s - 45,     lambda r, s : 27*r]
        ddN5 = [lambda r, s : -81*r - 27*s + 36,    lambda r, s : 0]
        ddN6 = [lambda r, s : 27*s,                 lambda r, s : 0]
        ddN7 = [lambda r, s : 0,                    lambda r, s : 27*r]
        ddN8 = [lambda r, s : 0,                    lambda r, s : -27*r - 81*s + 36]
        ddN9 = [lambda r, s : 27*s,                 lambda r, s : 54*r + 81*s - 45]
        ddN10 = [lambda r, s : -54*s,               lambda r, s : -54*r]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN

    def _dddN(self) -> np.ndarray:
        
        dddN1 = [lambda r, s : -27, lambda r, s : -27]
        dddN2 = [lambda r, s : 27,  lambda r, s : 0]
        dddN3 = [lambda r, s : 0,   lambda r, s : 27]
        dddN4 = [lambda r, s : 81,  lambda r, s : 0]
        dddN5 = [lambda r, s : -81, lambda r, s : 0]
        dddN6 = [lambda r, s : 0,   lambda r, s : 0]
        dddN7 = [lambda r, s : 0,   lambda r, s : 0]
        dddN8 = [lambda r, s : 0,   lambda r, s : -81]
        dddN9 = [lambda r, s : 0,   lambda r, s : 81]
        dddN10 = [lambda r, s : 0,  lambda r, s : 0]

        dddN = np.array([dddN1, dddN2, dddN3, dddN4, dddN5, dddN6, dddN7, dddN8, dddN9, dddN10])

        return dddN

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()