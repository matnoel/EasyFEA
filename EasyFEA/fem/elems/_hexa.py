# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hexa element module."""

import numpy as np

from .._group_elems import _GroupElem

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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,1,2,3,
                0,4,5,1,
                0,3,7,4,
                6,7,3,2,
                6,2,1,5,
                6,5,4,7]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,5],[5,4],[4,0],[3,2],[2,6],[6,7],[7,3],[0,3],[1,2],[5,6],[4,7]])

    def _N(self) -> np.ndarray:

        N1 = lambda r, s, t : -(r - 1)*(s - 1)*(t - 1)/8
        N2 = lambda r, s, t : (r + 1)*(s - 1)*(t - 1)/8
        N3 = lambda r, s, t : -(r + 1)*(s + 1)*(t - 1)/8
        N4 = lambda r, s, t : (r - 1)*(s + 1)*(t - 1)/8
        N5 = lambda r, s, t : (r - 1)*(s - 1)*(t + 1)/8
        N6 = lambda r, s, t : -(r + 1)*(s - 1)*(t + 1)/8
        N7 = lambda r, s, t : (r + 1)*(s + 1)*(t + 1)/8
        N8 = lambda r, s, t : -(r - 1)*(s + 1)*(t + 1)/8

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s, t : -(s - 1)*(t - 1)/8, lambda r, s, t : -(r - 1)*(t - 1)/8, lambda r, s, t : -(r - 1)*(s - 1)/8]
        dN2 = [lambda r, s, t : (s - 1)*(t - 1)/8, lambda r, s, t : (r + 1)*(t - 1)/8, lambda r, s, t : (r + 1)*(s - 1)/8]
        dN3 = [lambda r, s, t : -(s + 1)*(t - 1)/8, lambda r, s, t : -(r + 1)*(t - 1)/8, lambda r, s, t : -(r + 1)*(s + 1)/8]
        dN4 = [lambda r, s, t : (s + 1)*(t - 1)/8, lambda r, s, t : (r - 1)*(t - 1)/8, lambda r, s, t : (r - 1)*(s + 1)/8]
        dN5 = [lambda r, s, t : (s - 1)*(t + 1)/8, lambda r, s, t : (r - 1)*(t + 1)/8, lambda r, s, t : (r - 1)*(s - 1)/8]
        dN6 = [lambda r, s, t : -(s - 1)*(t + 1)/8, lambda r, s, t : -(r + 1)*(t + 1)/8, lambda r, s, t : -(r + 1)*(s - 1)/8]
        dN7 = [lambda r, s, t : (s + 1)*(t + 1)/8, lambda r, s, t : (r + 1)*(t + 1)/8, lambda r, s, t : (r + 1)*(s + 1)/8]
        dN8 = [lambda r, s, t : -(s + 1)*(t + 1)/8, lambda r, s, t : -(r - 1)*(t + 1)/8, lambda r, s, t : -(r - 1)*(s + 1)/8]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1, -1, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,8,1,11,2,13,3,9,
                0,10,4,16,5,12,1,8,
                0,9,3,15,7,17,4,10,
                6,19,7,15,3,13,2,14,
                6,14,2,11,1,12,5,18,
                6,18,5,16,4,17,7,19]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,5],[5,4],[4,0],[3,2],[2,6],[6,7],[7,3],[0,3],[1,2],[5,6],[4,7]])

    def _N(self) -> np.ndarray:

        N1 = lambda r, s, t : (r - 1)*(s - 1)*(t - 1)*(r + s + t + 2)/8
        N2 = lambda r, s, t : (r + 1)*(s - 1)*(t - 1)*(r - s - t - 2)/8
        N3 = lambda r, s, t : -(r + 1)*(s + 1)*(t - 1)*(r + s - t - 2)/8
        N4 = lambda r, s, t : -(r - 1)*(s + 1)*(t - 1)*(r - s + t + 2)/8
        N5 = lambda r, s, t : -(r - 1)*(s - 1)*(t + 1)*(r + s - t + 2)/8
        N6 = lambda r, s, t : -(r + 1)*(s - 1)*(t + 1)*(r - s + t - 2)/8
        N7 = lambda r, s, t : (r + 1)*(s + 1)*(t + 1)*(r + s + t - 2)/8
        N8 = lambda r, s, t : (r - 1)*(s + 1)*(t + 1)*(r - s - t + 2)/8
        N9 = lambda r, s, t : -(r - 1)*(r + 1)*(s - 1)*(t - 1)/4
        N10 = lambda r, s, t : -(r - 1)*(s - 1)*(s + 1)*(t - 1)/4
        N11 = lambda r, s, t : -(r - 1)*(s - 1)*(t - 1)*(t + 1)/4
        N12 = lambda r, s, t : (r + 1)*(s - 1)*(s + 1)*(t - 1)/4
        N13 = lambda r, s, t : (r + 1)*(s - 1)*(t - 1)*(t + 1)/4
        N14 = lambda r, s, t : (r - 1)*(r + 1)*(s + 1)*(t - 1)/4
        N15 = lambda r, s, t : -(r + 1)*(s + 1)*(t - 1)*(t + 1)/4
        N16 = lambda r, s, t : (r - 1)*(s + 1)*(t - 1)*(t + 1)/4
        N17 = lambda r, s, t : (r - 1)*(r + 1)*(s - 1)*(t + 1)/4
        N18 = lambda r, s, t : (r - 1)*(s - 1)*(s + 1)*(t + 1)/4
        N19 = lambda r, s, t : -(r + 1)*(s - 1)*(s + 1)*(t + 1)/4
        N20 = lambda r, s, t : -(r - 1)*(r + 1)*(s + 1)*(t + 1)/4

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17, N18, N19, N20]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s, t : r*s*t/4 - r*s/4 - r*t/4 + r/4 + s**2*t/8 - s**2/8 + s*t**2/8 - s*t/8 - t**2/8 + 1/8,
               lambda r, s, t : r**2*t/8 - r**2/8 + r*s*t/4 - r*s/4 + r*t**2/8 - r*t/8 - s*t/4 + s/4 - t**2/8 + 1/8,
               lambda r, s, t : r**2*s/8 - r**2/8 + r*s**2/8 + r*s*t/4 - r*s/8 - r*t/4 - s**2/8 - s*t/4 + t/4 + 1/8]
        dN2 = [lambda r, s, t : r*s*t/4 - r*s/4 - r*t/4 + r/4 - s**2*t/8 + s**2/8 - s*t**2/8 + s*t/8 + t**2/8 - 1/8,
               lambda r, s, t : r**2*t/8 - r**2/8 - r*s*t/4 + r*s/4 - r*t**2/8 + r*t/8 - s*t/4 + s/4 - t**2/8 + 1/8,
               lambda r, s, t : r**2*s/8 - r**2/8 - r*s**2/8 - r*s*t/4 + r*s/8 + r*t/4 - s**2/8 - s*t/4 + t/4 + 1/8]
        dN3 = [lambda r, s, t : -r*s*t/4 + r*s/4 - r*t/4 + r/4 - s**2*t/8 + s**2/8 + s*t**2/8 - s*t/8 + t**2/8 - 1/8,
               lambda r, s, t : -r**2*t/8 + r**2/8 - r*s*t/4 + r*s/4 + r*t**2/8 - r*t/8 - s*t/4 + s/4 + t**2/8 - 1/8,
               lambda r, s, t : -r**2*s/8 - r**2/8 - r*s**2/8 + r*s*t/4 - r*s/8 + r*t/4 - s**2/8 + s*t/4 + t/4 + 1/8]
        dN4 = [lambda r, s, t : -r*s*t/4 + r*s/4 - r*t/4 + r/4 + s**2*t/8 - s**2/8 - s*t**2/8 + s*t/8 - t**2/8 + 1/8,
               lambda r, s, t : -r**2*t/8 + r**2/8 + r*s*t/4 - r*s/4 - r*t**2/8 + r*t/8 - s*t/4 + s/4 + t**2/8 - 1/8,
               lambda r, s, t : -r**2*s/8 - r**2/8 + r*s**2/8 - r*s*t/4 + r*s/8 - r*t/4 - s**2/8 + s*t/4 + t/4 + 1/8]
        dN5 = [lambda r, s, t : -r*s*t/4 - r*s/4 + r*t/4 + r/4 - s**2*t/8 - s**2/8 + s*t**2/8 + s*t/8 - t**2/8 + 1/8,
               lambda r, s, t : -r**2*t/8 - r**2/8 - r*s*t/4 - r*s/4 + r*t**2/8 + r*t/8 + s*t/4 + s/4 - t**2/8 + 1/8,
               lambda r, s, t : -r**2*s/8 + r**2/8 - r*s**2/8 + r*s*t/4 + r*s/8 - r*t/4 + s**2/8 - s*t/4 + t/4 - 1/8]
        dN6 = [lambda r, s, t : -r*s*t/4 - r*s/4 + r*t/4 + r/4 + s**2*t/8 + s**2/8 - s*t**2/8 - s*t/8 + t**2/8 - 1/8,
               lambda r, s, t : -r**2*t/8 - r**2/8 + r*s*t/4 + r*s/4 - r*t**2/8 - r*t/8 + s*t/4 + s/4 - t**2/8 + 1/8,
               lambda r, s, t : -r**2*s/8 + r**2/8 + r*s**2/8 - r*s*t/4 - r*s/8 + r*t/4 + s**2/8 - s*t/4 + t/4 - 1/8]
        dN7 = [lambda r, s, t : r*s*t/4 + r*s/4 + r*t/4 + r/4 + s**2*t/8 + s**2/8 + s*t**2/8 + s*t/8 + t**2/8 - 1/8,
               lambda r, s, t : r**2*t/8 + r**2/8 + r*s*t/4 + r*s/4 + r*t**2/8 + r*t/8 + s*t/4 + s/4 + t**2/8 - 1/8,
               lambda r, s, t : r**2*s/8 + r**2/8 + r*s**2/8 + r*s*t/4 + r*s/8 + r*t/4 + s**2/8 + s*t/4 + t/4 - 1/8]
        dN8 = [lambda r, s, t : r*s*t/4 + r*s/4 + r*t/4 + r/4 - s**2*t/8 - s**2/8 - s*t**2/8 - s*t/8 - t**2/8 + 1/8,
               lambda r, s, t : r**2*t/8 + r**2/8 - r*s*t/4 - r*s/4 - r*t**2/8 - r*t/8 + s*t/4 + s/4 + t**2/8 - 1/8,
               lambda r, s, t : r**2*s/8 + r**2/8 - r*s**2/8 - r*s*t/4 - r*s/8 - r*t/4 + s**2/8 + s*t/4 + t/4 - 1/8]
        dN9 = [lambda r, s, t : -r*s*t/2 + r*s/2 + r*t/2 - r/2,
               lambda r, s, t : -r**2*t/4 + r**2/4 + t/4 - 1/4,
               lambda r, s, t : -r**2*s/4 + r**2/4 + s/4 - 1/4]
        dN10 = [lambda r, s, t : -s**2*t/4 + s**2/4 + t/4 - 1/4,
               lambda r, s, t : -r*s*t/2 + r*s/2 + s*t/2 - s/2,
               lambda r, s, t : -r*s**2/4 + r/4 + s**2/4 - 1/4]
        dN11 = [lambda r, s, t : -s*t**2/4 + s/4 + t**2/4 - 1/4,
               lambda r, s, t : -r*t**2/4 + r/4 + t**2/4 - 1/4,
               lambda r, s, t : -r*s*t/2 + r*t/2 + s*t/2 - t/2]
        dN12 = [lambda r, s, t : s**2*t/4 - s**2/4 - t/4 + 1/4,
               lambda r, s, t : r*s*t/2 - r*s/2 + s*t/2 - s/2,
               lambda r, s, t : r*s**2/4 - r/4 + s**2/4 - 1/4]
        dN13 = [lambda r, s, t : s*t**2/4 - s/4 - t**2/4 + 1/4,
               lambda r, s, t : r*t**2/4 - r/4 + t**2/4 - 1/4,
               lambda r, s, t : r*s*t/2 - r*t/2 + s*t/2 - t/2]
        dN14 = [lambda r, s, t : r*s*t/2 - r*s/2 + r*t/2 - r/2,
               lambda r, s, t : r**2*t/4 - r**2/4 - t/4 + 1/4,
               lambda r, s, t : r**2*s/4 + r**2/4 - s/4 - 1/4]
        dN15 = [lambda r, s, t : -s*t**2/4 + s/4 - t**2/4 + 1/4,
               lambda r, s, t : -r*t**2/4 + r/4 - t**2/4 + 1/4,
               lambda r, s, t : -r*s*t/2 - r*t/2 - s*t/2 - t/2]
        dN16 = [lambda r, s, t : s*t**2/4 - s/4 + t**2/4 - 1/4,
               lambda r, s, t : r*t**2/4 - r/4 - t**2/4 + 1/4,
               lambda r, s, t : r*s*t/2 + r*t/2 - s*t/2 - t/2]
        dN17 = [lambda r, s, t : r*s*t/2 + r*s/2 - r*t/2 - r/2,
               lambda r, s, t : r**2*t/4 + r**2/4 - t/4 - 1/4,
               lambda r, s, t : r**2*s/4 - r**2/4 - s/4 + 1/4]
        dN18 = [lambda r, s, t : s**2*t/4 + s**2/4 - t/4 - 1/4,
               lambda r, s, t : r*s*t/2 + r*s/2 - s*t/2 - s/2,
               lambda r, s, t : r*s**2/4 - r/4 - s**2/4 + 1/4]
        dN19 = [lambda r, s, t : -s**2*t/4 - s**2/4 + t/4 + 1/4,
               lambda r, s, t : -r*s*t/2 - r*s/2 - s*t/2 - s/2,
               lambda r, s, t : -r*s**2/4 + r/4 - s**2/4 + 1/4]
        dN20 = [lambda r, s, t : -r*s*t/2 - r*s/2 - r*t/2 - r/2,
               lambda r, s, t : -r**2*t/4 - r**2/4 + t/4 + 1/4,
               lambda r, s, t : -r**2*s/4 - r**2/4 + s/4 + 1/4]
        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10, dN11, dN12, dN13, dN14, dN15, dN16, dN17, dN18, dN19, dN20])        

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda r, s, t : (s - 1)*(t - 1)/4, lambda r, s, t : (r - 1)*(t - 1)/4, lambda r, s, t : (r - 1)*(s - 1)/4]
        ddN2 = [lambda r, s, t : (s - 1)*(t - 1)/4, lambda r, s, t : -(r + 1)*(t - 1)/4, lambda r, s, t : -(r + 1)*(s - 1)/4]
        ddN3 = [lambda r, s, t : -(s + 1)*(t - 1)/4, lambda r, s, t : -(r + 1)*(t - 1)/4, lambda r, s, t : (r + 1)*(s + 1)/4]
        ddN4 = [lambda r, s, t : -(s + 1)*(t - 1)/4, lambda r, s, t : (r - 1)*(t - 1)/4, lambda r, s, t : -(r - 1)*(s + 1)/4]
        ddN5 = [lambda r, s, t : -(s - 1)*(t + 1)/4, lambda r, s, t : -(r - 1)*(t + 1)/4, lambda r, s, t : (r - 1)*(s - 1)/4]
        ddN6 = [lambda r, s, t : -(s - 1)*(t + 1)/4, lambda r, s, t : (r + 1)*(t + 1)/4, lambda r, s, t : -(r + 1)*(s - 1)/4]
        ddN7 = [lambda r, s, t : (s + 1)*(t + 1)/4, lambda r, s, t : (r + 1)*(t + 1)/4, lambda r, s, t : (r + 1)*(s + 1)/4]
        ddN8 = [lambda r, s, t : (s + 1)*(t + 1)/4, lambda r, s, t : -(r - 1)*(t + 1)/4, lambda r, s, t : -(r - 1)*(s + 1)/4]
        ddN9 = [lambda r, s, t : -(s - 1)*(t - 1)/2, lambda r, s, t : 0, lambda r, s, t : 0]
        ddN10 = [lambda r, s, t : 0, lambda r, s, t : -(r - 1)*(t - 1)/2, lambda r, s, t : 0]
        ddN11 = [lambda r, s, t : 0, lambda r, s, t : 0, lambda r, s, t : -(r - 1)*(s - 1)/2]
        ddN12 = [lambda r, s, t : 0, lambda r, s, t : (r + 1)*(t - 1)/2, lambda r, s, t : 0]
        ddN13 = [lambda r, s, t : 0, lambda r, s, t : 0, lambda r, s, t : (r + 1)*(s - 1)/2]
        ddN14 = [lambda r, s, t : (s + 1)*(t - 1)/2, lambda r, s, t : 0, lambda r, s, t : 0]
        ddN15 = [lambda r, s, t : 0, lambda r, s, t : 0, lambda r, s, t : -(r + 1)*(s + 1)/2]
        ddN16 = [lambda r, s, t : 0, lambda r, s, t : 0, lambda r, s, t : (r - 1)*(s + 1)/2]
        ddN17 = [lambda r, s, t : (s - 1)*(t + 1)/2, lambda r, s, t : 0, lambda r, s, t : 0]
        ddN18 = [lambda r, s, t : 0, lambda r, s, t : (r - 1)*(t + 1)/2, lambda r, s, t : 0]
        ddN19 = [lambda r, s, t : 0, lambda r, s, t : -(r + 1)*(t + 1)/2, lambda r, s, t : 0]
        ddN20 = [lambda r, s, t : -(s + 1)*(t + 1)/2, lambda r, s, t : 0, lambda r, s, t : 0]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10, ddN11, ddN12, ddN13, ddN14, ddN15, ddN16, ddN17, ddN18, ddN19, ddN20])        

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()