# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Prism element module."""

import numpy as np

from .._group_elems import _GroupElem

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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,
                0,2,5,3,
                1,4,5,2,
                3,5,4,3,
                0,1,2,0]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]])

    def _N(self) -> np.ndarray:        

        N1 = lambda x,y,z: 0.5*x*z + 0.5*y*z + -0.5*x + -0.5*y + -0.5*z + 0.5
        N2 = lambda x,y,z: -0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N3 = lambda x,y,z: 0.0*x*z + -0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0
        N4 = lambda x,y,z: -0.5*x*z + -0.5*y*z + -0.5*x + -0.5*y + 0.5*z + 0.5
        N5 = lambda x,y,z: 0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N6 = lambda x,y,z: 0.0*x*z + 0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0        

        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:        

        dN1 = [lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*x + 0.5*y + -0.5]
        dN2 = [lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*x + 0.0*y + 0.0]
        dN3 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*x + -0.5*y + 0.0]
        dN4 = [lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*x + -0.5*y + 0.5]
        dN5 = [lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*x + 0.0*y + 0.0]
        dN6 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*x + 0.5*y + 0.0]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])        

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [0, 0, -1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,8,3,12,4,10,1,6,
                0,7,2,11,5,13,3,8,
                1,10,4,14,5,11,2,9,
                3,13,5,14,4,12,3,3,
                0,6,1,9,2,7,0,0]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[1,2],[2,0],[3,4],[4,5],[5,3],[0,3],[1,4],[2,5]])

    def _N(self) -> np.ndarray:

        N1 = lambda x,y,z: -1.0*x**2*z + -1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + -2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + 1.5*x*z + 1.5*y*z + -1.0*x + -1.0*y + -0.5*z + 2.0*x*y + 0.0
        N2 = lambda x,y,z: -1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N3 = lambda x,y,z: 0.0*x**2*z + -1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N4 = lambda x,y,z: 1.0*x**2*z + 1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + 2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + -1.5*x*z + -1.5*y*z + -1.0*x + -1.0*y + 0.5*z + 2.0*x*y + 0.0
        N5 = lambda x,y,z: 1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N6 = lambda x,y,z: 0.0*x**2*z + 1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + -0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N7 = lambda x,y,z: 2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + -2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N8 = lambda x,y,z: 0.0*x**2*z + 2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + -2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N9 = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 1.0*z**2*x + 1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + -1.0*z**2 + 0.0*x*z + 0.0*y*z + -1.0*x + -1.0*y + 0.0*z + 0.0*x*y + 1.0
        N10 = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0
        N11 = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + -1.0*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N12 = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + -1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 1.0*y + 0.0*z + 0.0*x*y + 0.0
        N13 = lambda x,y,z: -2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N14 = lambda x,y,z: 0.0*x**2*z + -2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + 2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N15 = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda x,y,z: -2.0*x*z + -0.5*z**2 + -2.0*y*z + 2.0*x + 1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: -2.0*y*z + -0.5*z**2 + -2.0*x*z + 2.0*y + 1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: -1.0*x**2 + -1.0*y**2 + -1.0*z*x + -1.0*z*y + -2.0*x*y + 1.0*z + 1.5*x + 1.5*y + -0.5]
        dN2 = [lambda x,y,z: -2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + 0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: -1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.5*x + 0.0*y + 0.0]
        dN3 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: -2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + 0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + -1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.5*y + 0.0]
        dN4 = [lambda x,y,z: 2.0*x*z + -0.5*z**2 + 2.0*y*z + 2.0*x + -1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: 2.0*y*z + -0.5*z**2 + 2.0*x*z + 2.0*y + -1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: 1.0*x**2 + 1.0*y**2 + -1.0*z*x + -1.0*z*y + 2.0*x*y + 1.0*z + -1.5*x + -1.5*y + 0.5]
        dN5 = [lambda x,y,z: 2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + -0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + -0.5*x + 0.0*y + 0.0]
        dN6 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + -0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + -0.5*y + 0.0]
        dN7 = [lambda x,y,z: 4.0*x*z + 0.0*z**2 + 2.0*y*z + -4.0*x + -2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + -2.0*x + 0.0*y + 0.0]
        dN8 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: 4.0*y*z + 0.0*z**2 + 2.0*x*z + -4.0*y + -2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + -2.0*y + 0.0]
        dN9 = [lambda x,y,z: 0.0*x*z + 1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z*x + 2.0*z*y + 0.0*x*y + -2.0*z + 0.0*x + 0.0*y + 0.0]
        dN10 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN11 = [lambda x,y,z: 0.0*x*z + -1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -2.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN12 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + -1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + -2.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN13 = [lambda x,y,z: -4.0*x*z + 0.0*z**2 + -2.0*y*z + -4.0*x + 2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: -2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 2.0*x + 0.0*y + 0.0]
        dN14 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: -4.0*y*z + 0.0*z**2 + -2.0*x*z + -4.0*y + 2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + -2.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 2.0*y + 0.0]
        dN15 = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10, dN11, dN12, dN13, dN14, dN15])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN2 = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN3 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN4 = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN5 = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN6 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN7 = [lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN8 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN9 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*x + 2.0*y + -2.0]
        ddN10 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN11 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*x + 0.0*y + 0.0]
        ddN12 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + -2.0*y + 0.0]
        ddN13 = [lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN14 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN15 = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10, ddN11, ddN12, ddN13, ddN14, ddN15])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()