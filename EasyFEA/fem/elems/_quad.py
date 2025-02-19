# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Quad element module."""

import numpy as np

from .._group_elems import _GroupElem

class QUAD4(_GroupElem):
    #       v
    #       ^
    #       |
    # 3-----------2
    # |     |     |
    # |     |     |
    # |     +---- | --> u
    # |           |
    # |           |
    # 0-----------1

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1, -1]

    @property
    def triangles(self) -> list[int]:
        return [0,1,3,
                1,2,3]

    @property
    def faces(self) -> list[int]:
        return [0,1,2,3,0]

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : (r - 1)*(s - 1)/4
        N2 = lambda r, s : -(r + 1)*(s - 1)/4
        N3 = lambda r, s : (r + 1)*(s + 1)/4
        N4 = lambda r, s : -(r - 1)*(s + 1)/4
        
        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : s/4 - 1/4,  lambda r, s : r/4 - 1/4]
        dN2 = [lambda r, s : 1/4 - s/4,  lambda r, s : -r/4 - 1/4]
        dN3 = [lambda r, s : s/4 + 1/4,  lambda r, s : r/4 + 1/4]
        dN4 = [lambda r, s : -s/4 - 1/4, lambda r, s : 1/4 - r/4]
        
        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()
    
    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()

class QUAD8(_GroupElem):
    #       v
    #       ^
    #       |
    # 3-----6-----2
    # |     |     |
    # |     |     |
    # 7     +---- 5 --> u
    # |           |
    # |           |
    # 0-----4-----1

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1, -1]

    @property
    def triangles(self) -> list[int]:
        return [4,5,7,
                5,6,7,
                0,4,7,
                4,1,5,
                5,2,6,
                6,3,7]

    @property
    def faces(self) -> list[int]:
        return [0,4,1,5,2,6,3,7,0]

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : -(r - 1)*(s - 1)*(r + s + 1)/4
        N2 = lambda r, s : -(r + 1)*(s - 1)*(r - s - 1)/4
        N3 = lambda r, s : (r + 1)*(s + 1)*(r + s - 1)/4
        N4 = lambda r, s : (r - 1)*(s + 1)*(r - s + 1)/4
        N5 = lambda r, s : (r - 1)*(r + 1)*(s - 1)/2
        N6 = lambda r, s : -(r + 1)*(s - 1)*(s + 1)/2
        N7 = lambda r, s : -(r - 1)*(r + 1)*(s + 1)/2
        N8 = lambda r, s : (r - 1)*(s - 1)*(s + 1)/2
        
        N =  np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : -(r - 1)*(s - 1)/4 - (s - 1)*(r + s + 1)/4,
               lambda r, s : -(r - 1)*(s - 1)/4 - (r - 1)*(r + s + 1)/4]
        dN2 = [lambda r, s : -(r + 1)*(s - 1)/4 - (s - 1)*(r - s - 1)/4,
               lambda r, s : (r + 1)*(s - 1)/4 - (r + 1)*(r - s - 1)/4]
        dN3 = [lambda r, s : (r + 1)*(s + 1)/4 + (s + 1)*(r + s - 1)/4,
               lambda r, s : (r + 1)*(s + 1)/4 + (r + 1)*(r + s - 1)/4]
        dN4 = [lambda r, s : (r - 1)*(s + 1)/4 + (s + 1)*(r - s + 1)/4,
               lambda r, s : -(r - 1)*(s + 1)/4 + (r - 1)*(r - s + 1)/4]
        dN5 = [lambda r, s : (r - 1)*(s - 1)/2 + (r + 1)*(s - 1)/2,
               lambda r, s : (r - 1)*(r + 1)/2]
        dN6 = [lambda r, s : -(s - 1)*(s + 1)/2,
               lambda r, s : -(r + 1)*(s - 1)/2 - (r + 1)*(s + 1)/2]
        dN7 = [lambda r, s : -(r - 1)*(s + 1)/2 - (r + 1)*(s + 1)/2,
               lambda r, s : -(r - 1)*(r + 1)/2]
        dN8 = [lambda r, s : (s - 1)*(s + 1)/2,
               lambda r, s : (r - 1)*(s - 1)/2 + (r - 1)*(s + 1)/2]
                        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda r, s : 1/2 - s/2, lambda r, s : 1/2 - r/2]
        ddN2 = [lambda r, s : 1/2 - s/2, lambda r, s : r/2 + 1/2]
        ddN3 = [lambda r, s : s/2 + 1/2, lambda r, s : r/2 + 1/2]
        ddN4 = [lambda r, s : s/2 + 1/2, lambda r, s : 1/2 - r/2]
        ddN5 = [lambda r, s : s - 1,     lambda r, s : 0]
        ddN6 = [lambda r, s : 0,         lambda r, s : -r - 1]
        ddN7 = [lambda r, s : -s - 1,    lambda r, s : 0]
        ddN8 = [lambda r, s : 0,         lambda r, s : r - 1]
                        
        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()
    
class QUAD9(_GroupElem):
    #       v
    #       ^
    #       |
    # 3-----6-----2
    # |     |     |
    # |     |     |
    # 7     8---- 5 --> u
    # |           |
    # |           |
    # 0-----4-----1

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1, -1]

    @property
    def triangles(self) -> list[int]:
        return [0, 4, 8,
                4, 1, 5,
                4, 5, 8,
                0, 8, 7,
                7, 8, 6,
                8, 5, 2,
                8, 2, 6,
                7, 6, 3]

    @property
    def faces(self) -> list[int]:
        return [0,4,1,5,2,6,3,7,0]

    def _N(self) -> np.ndarray:

        N1 = lambda r, s : r*s*(r - 1)*(s - 1)/4
        N2 = lambda r, s : r*s*(r + 1)*(s - 1)/4
        N3 = lambda r, s : r*s*(r + 1)*(s + 1)/4
        N4 = lambda r, s : r*s*(r - 1)*(s + 1)/4
        N5 = lambda r, s : -s*(r - 1)*(r + 1)*(s - 1)/2
        N6 = lambda r, s : -r*(r + 1)*(s - 1)*(s + 1)/2
        N7 = lambda r, s : -s*(r - 1)*(r + 1)*(s + 1)/2
        N8 = lambda r, s : -r*(r - 1)*(s - 1)*(s + 1)/2
        N9 = lambda r, s : (r - 1)*(r + 1)*(s - 1)*(s + 1)

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda r, s : r*s*(s - 1)/4 + s*(r - 1)*(s - 1)/4,
               lambda r, s : r*s*(r - 1)/4 + r*(r - 1)*(s - 1)/4]
        dN2 = [lambda r, s : r*s*(s - 1)/4 + s*(r + 1)*(s - 1)/4,
               lambda r, s : r*s*(r + 1)/4 + r*(r + 1)*(s - 1)/4]
        dN3 = [lambda r, s : r*s*(s + 1)/4 + s*(r + 1)*(s + 1)/4,
               lambda r, s : r*s*(r + 1)/4 + r*(r + 1)*(s + 1)/4]
        dN4 = [lambda r, s : r*s*(s + 1)/4 + s*(r - 1)*(s + 1)/4,
               lambda r, s : r*s*(r - 1)/4 + r*(r - 1)*(s + 1)/4]
        dN5 = [lambda r, s : -s*(r - 1)*(s - 1)/2 - s*(r + 1)*(s - 1)/2,
               lambda r, s : -s*(r - 1)*(r + 1)/2 - (r - 1)*(r + 1)*(s - 1)/2]
        dN6 = [lambda r, s : -r*(s - 1)*(s + 1)/2 - (r + 1)*(s - 1)*(s + 1)/2,
               lambda r, s : -r*(r + 1)*(s - 1)/2 - r*(r + 1)*(s + 1)/2]
        dN7 = [lambda r, s : -s*(r - 1)*(s + 1)/2 - s*(r + 1)*(s + 1)/2,
               lambda r, s : -s*(r - 1)*(r + 1)/2 - (r - 1)*(r + 1)*(s + 1)/2]
        dN8 = [lambda r, s : -r*(s - 1)*(s + 1)/2 - (r - 1)*(s - 1)*(s + 1)/2,
               lambda r, s : -r*(r - 1)*(s - 1)/2 - r*(r - 1)*(s + 1)/2]
        dN9 = [lambda r, s : (r - 1)*(s - 1)*(s + 1) + (r + 1)*(s - 1)*(s + 1),
               lambda r, s : (r - 1)*(r + 1)*(s - 1) + (r - 1)*(r + 1)*(s + 1)]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda r, s : s*(s - 1)/2, lambda r, s : r*(r - 1)/2]
        ddN2 = [lambda r, s : s*(s - 1)/2, lambda r, s : r*(r + 1)/2]
        ddN3 = [lambda r, s : s*(s + 1)/2, lambda r, s : r*(r + 1)/2]
        ddN4 = [lambda r, s : s*(s + 1)/2, lambda r, s : r*(r - 1)/2]
        ddN5 = [lambda r, s : -s*(s - 1), lambda r, s : -(r - 1)*(r + 1)]
        ddN6 = [lambda r, s : -(s - 1)*(s + 1), lambda r, s : -r*(r + 1)]
        ddN7 = [lambda r, s : -s*(s + 1), lambda r, s : -(r - 1)*(r + 1)]
        ddN8 = [lambda r, s : -(s - 1)*(s + 1), lambda r, s : -r*(r - 1)]
        ddN9 = [lambda r, s : 2*(s - 1)*(s + 1), lambda r, s : 2*(r - 1)*(r + 1)]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()