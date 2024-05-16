# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Hexa element module."""

from ... import np, plt

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
        return [0,1,2,3,0,4,5,1,0,3,7,4,6,7,3,2,6,2,1,5,6,5,4,7]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,5],[5,4],[4,0],[3,2],[2,6],[6,7],[7,3],[0,3],[1,2],[5,6],[4,7]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1-z)
        N2t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1-z)
        N3t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1-z)
        N4t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1-z)
        N5t = lambda x,y,z: 1/8 * (1-x) * (1-y) * (1+z)
        N6t = lambda x,y,z: 1/8 * (1+x) * (1-y) * (1+z)
        N7t = lambda x,y,z: 1/8 * (1+x) * (1+y) * (1+z)
        N8t = lambda x,y,z: 1/8 * (1-x) * (1+y) * (1+z)

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -1/8 * (1-y) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-z),   lambda x,y,z: -1/8 * (1-x) * (1-y)]
        dN2t = [lambda x,y,z: 1/8 * (1-y) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1-y)]
        dN3t = [lambda x,y,z: 1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1+x) * (1-z),    lambda x,y,z: -1/8 * (1+x) * (1+y)]
        dN4t = [lambda x,y,z: -1/8 * (1+y) * (1-z),    lambda x,y,z: 1/8 * (1-x) * (1-z),    lambda x,y,z: -1/8 * (1-x) * (1+y)]
        dN5t = [lambda x,y,z: -1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1-y)]
        dN6t = [lambda x,y,z: 1/8 * (1-y) * (1+z),    lambda x,y,z: -1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1-y)]
        dN7t = [lambda x,y,z: 1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+z),    lambda x,y,z: 1/8 * (1+x) * (1+y)]
        dN8t = [lambda x,y,z: -1/8 * (1+y) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+z),    lambda x,y,z: 1/8 * (1-x) * (1+y)]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

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

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 0.125*x**2*y*z + 0.125*y**2*x*z + 0.125*z**2*x*y + -0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + 0.125*y + 0.125*z + -0.125*x*y*z + -0.25
        N2t = lambda x,y,z: 0.125*x**2*y*z + -0.125*y**2*x*z + -0.125*z**2*x*y + -0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + 0.125*y + 0.125*z + 0.125*x*y*z + -0.25
        N3t = lambda x,y,z: -0.125*x**2*y*z + -0.125*y**2*x*z + 0.125*z**2*x*y + 0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + -0.125*y + 0.125*z + -0.125*x*y*z + -0.25
        N4t = lambda x,y,z: -0.125*x**2*y*z + 0.125*y**2*x*z + -0.125*z**2*x*y + 0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + -0.125*x**2*z + -0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + -0.125*y + 0.125*z + 0.125*x*y*z + -0.25
        N5t = lambda x,y,z: -0.125*x**2*y*z + -0.125*y**2*x*z + 0.125*z**2*x*y + -0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + 0.125*y + -0.125*z + 0.125*x*y*z + -0.25
        N6t = lambda x,y,z: -0.125*x**2*y*z + 0.125*y**2*x*z + -0.125*z**2*x*y + -0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + -0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + 0.125*y + -0.125*z + -0.125*x*y*z + -0.25
        N7t = lambda x,y,z: 0.125*x**2*y*z + 0.125*y**2*x*z + 0.125*z**2*x*y + 0.125*x**2*y + 0.125*y**2*x + 0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -0.125*x + -0.125*y + -0.125*z + 0.125*x*y*z + -0.25
        N8t = lambda x,y,z: 0.125*x**2*y*z + -0.125*y**2*x*z + -0.125*z**2*x*y + 0.125*x**2*y + -0.125*y**2*x + -0.125*z**2*x + 0.125*x**2*z + 0.125*y**2*z + 0.125*z**2*y + 0.125*x**2 + 0.125*y**2 + 0.125*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.125*x + -0.125*y + -0.125*z + -0.125*x*y*z + -0.25
        N9t = lambda x,y,z: -0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + 0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + 0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.25*y*z + 0.0*x + -0.25*y + -0.25*z + 0.0*x*y*z + 0.25
        N10t = lambda x,y,z: 0.0*x**2*y*z + -0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + 0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + 0.25*x*z + 0.0*y*z + -0.25*x + 0.0*y + -0.25*z + 0.0*x*y*z + 0.25
        N11t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + -0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + 0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + 0.25*x*y + 0.0*x*z + 0.0*y*z + -0.25*x + -0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N12t = lambda x,y,z: 0.0*x**2*y*z + 0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + -0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + -0.25*x*z + 0.0*y*z + 0.25*x + 0.0*y + -0.25*z + 0.0*x*y*z + 0.25
        N13t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + 0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + -0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + -0.25*x*y + 0.0*x*z + 0.0*y*z + 0.25*x + -0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N14t = lambda x,y,z: 0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + -0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + 0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + -0.25*y*z + 0.0*x + 0.25*y + -0.25*z + 0.0*x*y*z + 0.25
        N15t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + -0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + -0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + 0.25*x*y + 0.0*x*z + 0.0*y*z + 0.25*x + 0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N16t = lambda x,y,z: 0.0*x**2*y*z + 0.0*y**2*x*z + 0.25*z**2*x*y + 0.0*x**2*y + 0.0*y**2*x + 0.25*z**2*x + 0.0*x**2*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x**2 + 0.0*y**2 + -0.25*z**2 + -0.25*x*y + 0.0*x*z + 0.0*y*z + -0.25*x + 0.25*y + 0.0*z + 0.0*x*y*z + 0.25
        N17t = lambda x,y,z: 0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + 0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + -0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + -0.25*y*z + 0.0*x + -0.25*y + 0.25*z + 0.0*x*y*z + 0.25
        N18t = lambda x,y,z: 0.0*x**2*y*z + 0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + 0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + -0.25*x*z + 0.0*y*z + -0.25*x + 0.0*y + 0.25*z + 0.0*x*y*z + 0.25
        N19t = lambda x,y,z: 0.0*x**2*y*z + -0.25*y**2*x*z + 0.0*z**2*x*y + 0.0*x**2*y + -0.25*y**2*x + 0.0*z**2*x + 0.0*x**2*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x**2 + -0.25*y**2 + 0.0*z**2 + 0.0*x*y + 0.25*x*z + 0.0*y*z + 0.25*x + 0.0*y + 0.25*z + 0.0*x*y*z + 0.25
        N20t = lambda x,y,z: -0.25*x**2*y*z + 0.0*y**2*x*z + 0.0*z**2*x*y + -0.25*x**2*y + 0.0*y**2*x + 0.0*z**2*x + -0.25*x**2*z + 0.0*y**2*z + 0.0*z**2*y + -0.25*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.25*y*z + 0.0*x + 0.25*y + 0.25*z + 0.0*x*y*z + 0.25        

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t, N16t, N17t, N18t, N19t, N20t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: 0.25*x*y*z + 0.125*y**2*z + 0.125*z**2*y + -0.25*x*y + -0.125*y**2 + -0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + -0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + 0.25*y*x*z + 0.125*z**2*x + -0.125*x**2 + -0.25*y*x + -0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + -0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + 0.125*y**2*x + 0.25*z*x*y + -0.25*z*x + -0.125*x**2 + -0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + -0.125*x*y]
        dN2t = [lambda x,y,z: 0.25*x*y*z + -0.125*y**2*z + -0.125*z**2*y + -0.25*x*y + 0.125*y**2 + 0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + 0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + -0.25*y*x*z + -0.125*z**2*x + -0.125*x**2 + 0.25*y*x + -0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + 0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + -0.125*y**2*x + -0.25*z*x*y + 0.25*z*x + -0.125*x**2 + -0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + 0.125*x*y]
        dN3t = [lambda x,y,z: -0.25*x*y*z + -0.125*y**2*z + 0.125*z**2*y + 0.25*x*y + 0.125*y**2 + 0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + -0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + -0.25*y*x*z + 0.125*z**2*x + 0.125*x**2 + 0.25*y*x + -0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + -0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + -0.125*y**2*x + 0.25*z*x*y + 0.25*z*x + -0.125*x**2 + -0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + -0.125*x*y]
        dN4t = [lambda x,y,z: -0.25*x*y*z + 0.125*y**2*z + -0.125*z**2*y + 0.25*x*y + -0.125*y**2 + -0.125*z**2 + -0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + 0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + 0.25*y*x*z + -0.125*z**2*x + 0.125*x**2 + -0.25*y*x + -0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + 0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + 0.125*y**2*x + -0.25*z*x*y + -0.25*z*x + -0.125*x**2 + -0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + 0.125 + 0.125*x*y]
        dN5t = [lambda x,y,z: -0.25*x*y*z + -0.125*y**2*z + 0.125*z**2*y + -0.25*x*y + -0.125*y**2 + -0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + 0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + -0.25*y*x*z + 0.125*z**2*x + -0.125*x**2 + -0.25*y*x + 0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + 0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + -0.125*y**2*x + 0.25*z*x*y + -0.25*z*x + 0.125*x**2 + 0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + 0.125*x*y]
        dN6t = [lambda x,y,z: -0.25*x*y*z + 0.125*y**2*z + -0.125*z**2*y + -0.25*x*y + 0.125*y**2 + 0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + -0.125*y*z,
            lambda x,y,z: -0.125*x**2*z + 0.25*y*x*z + -0.125*z**2*x + -0.125*x**2 + 0.25*y*x + 0.25*y*z + -0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + 0.125 + -0.125*x*z,
            lambda x,y,z: -0.125*x**2*y + 0.125*y**2*x + -0.25*z*x*y + 0.25*z*x + 0.125*x**2 + 0.125*y**2 + -0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + -0.125*x*y]
        dN7t = [lambda x,y,z: 0.25*x*y*z + 0.125*y**2*z + 0.125*z**2*y + 0.25*x*y + 0.125*y**2 + 0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + -0.125 + 0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + 0.25*y*x*z + 0.125*z**2*x + 0.125*x**2 + 0.25*y*x + 0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + 0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + 0.125*y**2*x + 0.25*z*x*y + 0.25*z*x + 0.125*x**2 + 0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + 0.125*x*y]
        dN8t = [lambda x,y,z: 0.25*x*y*z + -0.125*y**2*z + -0.125*z**2*y + 0.25*x*y + -0.125*y**2 + -0.125*z**2 + 0.25*x*z + 0.25*x + 0.0*y + 0.0*z + 0.125 + -0.125*y*z,
            lambda x,y,z: 0.125*x**2*z + -0.25*y*x*z + -0.125*z**2*x + 0.125*x**2 + -0.25*y*x + 0.25*y*z + 0.125*z**2 + 0.25*y + 0.0*x + 0.0*z + -0.125 + -0.125*x*z,
            lambda x,y,z: 0.125*x**2*y + -0.125*y**2*x + -0.25*z*x*y + -0.25*z*x + 0.125*x**2 + 0.125*y**2 + 0.25*z*y + 0.25*z + 0.0*x + 0.0*y + -0.125 + -0.125*x*y]
        dN9t = [lambda x,y,z: -0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + 0.5*x*y + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: -0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + 0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + 0.25*z + -0.25 + 0.0*x*z,
            lambda x,y,z: -0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + 0.25*y + -0.25 + 0.0*x*y]
        dN10t = [lambda x,y,z: 0.0*x*y*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + 0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + 0.25*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + -0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + 0.5*y*x + 0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + -0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + 0.25*y**2 + 0.0*z*y + 0.0*z + 0.25*x + 0.0*y + -0.25 + 0.0*x*y]
        dN11t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x*y + 0.0*y**2 + 0.25*z**2 + 0.0*x*z + 0.0*x + 0.25*y + 0.0*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + -0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + 0.25*z**2 + 0.0*y + 0.25*x + 0.0*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + -0.5*z*x*y + 0.5*z*x + 0.0*x**2 + 0.0*y**2 + 0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN12t = [lambda x,y,z: 0.0*x*y*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + -0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + -0.25*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + -0.5*y*x + 0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + 0.25*y**2 + 0.0*z*y + 0.0*z + -0.25*x + 0.0*y + -0.25 + 0.0*x*y]
        dN13t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x*y + 0.0*y**2 + -0.25*z**2 + 0.0*x*z + 0.0*x + -0.25*y + 0.0*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + 0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + 0.25*z**2 + 0.0*y + -0.25*x + 0.0*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + 0.5*z*x*y + -0.5*z*x + 0.0*x**2 + 0.0*y**2 + 0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN14t = [lambda x,y,z: 0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + -0.5*x*y + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: 0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + -0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + -0.25*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + -0.25*y + -0.25 + 0.0*x*y]
        dN15t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + -0.25*z**2*y + 0.0*x*y + 0.0*y**2 + -0.25*z**2 + 0.0*x*z + 0.0*x + 0.25*y + 0.0*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + -0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + -0.25*z**2 + 0.0*y + 0.25*x + 0.0*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + -0.5*z*x*y + -0.5*z*x + 0.0*x**2 + 0.0*y**2 + -0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN16t = [lambda x,y,z: 0.0*x*y*z + 0.0*y**2*z + 0.25*z**2*y + 0.0*x*y + 0.0*y**2 + 0.25*z**2 + 0.0*x*z + 0.0*x + -0.25*y + 0.0*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.0*y*x*z + 0.25*z**2*x + 0.0*x**2 + 0.0*y*x + 0.0*y*z + -0.25*z**2 + 0.0*y + -0.25*x + 0.0*z + 0.25 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.0*y**2*x + 0.5*z*x*y + 0.5*z*x + 0.0*x**2 + 0.0*y**2 + -0.5*z*y + -0.5*z + 0.0*x + 0.0*y + 0.0 + 0.0*x*y]
        dN17t = [lambda x,y,z: 0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + 0.5*x*y + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: 0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + 0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + -0.25*z + -0.25 + 0.0*x*z,
            lambda x,y,z: 0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + -0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + -0.25*y + 0.25 + 0.0*x*y]
        dN18t = [lambda x,y,z: 0.0*x*y*z + 0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + 0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + -0.25*z + -0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + 0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + 0.5*y*x + -0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + 0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + -0.25*y**2 + 0.0*z*y + 0.0*z + -0.25*x + 0.0*y + 0.25 + 0.0*x*y]
        dN19t = [lambda x,y,z: 0.0*x*y*z + -0.25*y**2*z + 0.0*z**2*y + 0.0*x*y + -0.25*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*x + 0.0*y + 0.25*z + 0.25 + 0.0*y*z,
            lambda x,y,z: 0.0*x**2*z + -0.5*y*x*z + 0.0*z**2*x + 0.0*x**2 + -0.5*y*x + -0.5*y*z + 0.0*z**2 + -0.5*y + 0.0*x + 0.0*z + 0.0 + 0.0*x*z,
            lambda x,y,z: 0.0*x**2*y + -0.25*y**2*x + 0.0*z*x*y + 0.0*z*x + 0.0*x**2 + -0.25*y**2 + 0.0*z*y + 0.0*z + 0.25*x + 0.0*y + 0.25 + 0.0*x*y]
        dN20t = [lambda x,y,z: -0.5*x*y*z + 0.0*y**2*z + 0.0*z**2*y + -0.5*x*y + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + -0.5*x + 0.0*y + 0.0*z + 0.0 + 0.0*y*z,
            lambda x,y,z: -0.25*x**2*z + 0.0*y*x*z + 0.0*z**2*x + -0.25*x**2 + 0.0*y*x + 0.0*y*z + 0.0*z**2 + 0.0*y + 0.0*x + 0.25*z + 0.25 + 0.0*x*z,
            lambda x,y,z: -0.25*x**2*y + 0.0*y**2*x + 0.0*z*x*y + 0.0*z*x + -0.25*x**2 + 0.0*y**2 + 0.0*z*y + 0.0*z + 0.0*x + 0.25*y + 0.25 + 0.0*x*y]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t, dN16t, dN17t, dN18t, dN19t, dN20t])        

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: 0.25*y*z + -0.25*y + -0.25*z + 0.25, lambda x,y,z: 0.25*x*z + -0.25*x + -0.25*z + 0.25, lambda x,y,z: 0.25*x*y + -0.25*x + -0.25*y + 0.25]
        ddN2t = [lambda x,y,z: 0.25*y*z + -0.25*y + -0.25*z + 0.25,lambda x,y,z: -0.25*x*z + 0.25*x + -0.25*z + 0.25, lambda x,y,z: -0.25*x*y + 0.25*x + -0.25*y + 0.25]
        ddN3t = [lambda x,y,z: -0.25*y*z + 0.25*y + -0.25*z + 0.25, lambda x,y,z: -0.25*x*z + 0.25*x + -0.25*z + 0.25, lambda x,y,z: 0.25*x*y + 0.25*x + 0.25*y + 0.25]
        ddN4t = [lambda x,y,z: -0.25*y*z + 0.25*y + -0.25*z + 0.25, lambda x,y,z: 0.25*x*z + -0.25*x + -0.25*z + 0.25, lambda x,y,z: -0.25*x*y + -0.25*x + 0.25*y + 0.25]
        ddN5t = [lambda x,y,z: -0.25*y*z + -0.25*y + 0.25*z + 0.25, lambda x,y,z: -0.25*x*z + -0.25*x + 0.25*z + 0.25, lambda x,y,z: 0.25*x*y + -0.25*x + -0.25*y + 0.25]
        ddN6t = [lambda x,y,z: -0.25*y*z + -0.25*y + 0.25*z + 0.25, lambda x,y,z: 0.25*x*z + 0.25*x + 0.25*z + 0.25, lambda x,y,z: -0.25*x*y + 0.25*x + -0.25*y + 0.25]
        ddN7t = [lambda x,y,z: 0.25*y*z + 0.25*y + 0.25*z + 0.25, lambda x,y,z: 0.25*x*z + 0.25*x + 0.25*z + 0.25, lambda x,y,z: 0.25*x*y + 0.25*x + 0.25*y + 0.25]
        ddN8t = [lambda x,y,z: 0.25*y*z + 0.25*y + 0.25*z + 0.25, lambda x,y,z: -0.25*x*z + -0.25*x + 0.25*z + 0.25, lambda x,y,z: -0.25*x*y + -0.25*x + 0.25*y + 0.25]
        ddN9t = [lambda x,y,z: -0.5*y*z + 0.5*y + 0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN10t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: -0.5*x*z + 0.5*x + 0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN11t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: -0.5*x*y + 0.5*x + 0.5*y + -0.5]
        ddN12t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.5*x*z + -0.5*x + 0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN13t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.5*x*y + -0.5*x + 0.5*y + -0.5]
        ddN14t = [lambda x,y,z: 0.5*y*z + -0.5*y + 0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN15t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: -0.5*x*y + -0.5*x + -0.5*y + -0.5]
        ddN16t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.5*x*y + 0.5*x + -0.5*y + -0.5]
        ddN17t = [lambda x,y,z: 0.5*y*z + 0.5*y + -0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN18t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: 0.5*x*z + 0.5*x + -0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN19t = [lambda x,y,z: 0.0*y*z + 0.0*y + 0.0*z + 0.0, lambda x,y,z: -0.5*x*z + -0.5*x + -0.5*z + -0.5, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]
        ddN20t = [lambda x,y,z: -0.5*y*z + -0.5*y + -0.5*z + -0.5, lambda x,y,z: 0.0*x*z + 0.0*x + 0.0*z + 0.0, lambda x,y,z: 0.0*x*y + 0.0*x + 0.0*y + 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t, ddN16t, ddN17t, ddN18t, ddN19t, ddN20t])        

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()