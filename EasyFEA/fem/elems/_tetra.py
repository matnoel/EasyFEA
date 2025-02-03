# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Tetra element module."""

import numpy as np

from .._group_elems import _GroupElem

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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,1,2,
                0,3,1,
                0,2,3,
                1,3,2]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _N(self) -> np.ndarray:

        N1 = lambda x,y,z: 1-x-y-z
        N2 = lambda x,y,z: x
        N3 = lambda x,y,z: y
        N4 = lambda x,y,z: z

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
        dN2 = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
        dN3 = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
        dN4 = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
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

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return super().origin

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:        
        return [0,4,1,5,2,6,
                0,7,3,9,1,4,
                0,6,2,8,3,7,
                1,9,3,8,2,5]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _N(self) -> np.ndarray:

        N1 = lambda x,y,z: 2.0*x**2 + 2.0*y**2 + 2.0*z**2 + 4.0*x*y + 4.0*x*z + 4.0*y*z + -3.0*x + -3.0*y + -3.0*z + 1.0
        N2 = lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0
        N3 = lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0
        N4 = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + -1.0*z + 0.0
        N5 = lambda x,y,z: -4.0*x**2 + 0.0*y**2 + 0.0*z**2 + -4.0*x*y + -4.0*x*z + 0.0*y*z + 4.0*x + 0.0*y + 0.0*z + 0.0
        N6 = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 4.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N7 = lambda x,y,z: 0.0*x**2 + -4.0*y**2 + 0.0*z**2 + -4.0*x*y + 0.0*x*z + -4.0*y*z + 0.0*x + 4.0*y + 0.0*z + 0.0
        N8 = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -4.0*z**2 + 0.0*x*y + -4.0*x*z + -4.0*y*z + 0.0*x + 0.0*y + 4.0*z + 0.0
        N9 = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 4.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N10 = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 4.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0

        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda x,y,z: 4.0*x + 4.0*y + 4.0*z + -3.0,   lambda x,y,z: 4.0*y + 4.0*x + 4.0*z + -3.0,   lambda x,y,z: 4.0*z + 4.0*x + 4.0*y + -3.0]
        dN2 = [lambda x,y,z: 4.0*x + 0.0*y + 0.0*z + -1.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN3 = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 4.0*y + 0.0*x + 0.0*z + -1.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN4 = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 4.0*z + 0.0*x + 0.0*y + -1.0]
        dN5 = [lambda x,y,z: -8.0*x + -4.0*y + -4.0*z + 4.0,   lambda x,y,z: 0.0*y + -4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + -4.0*x + 0.0*y + 0.0]
        dN6 = [lambda x,y,z: 0.0*x + 4.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN7 = [lambda x,y,z: 0.0*x + -4.0*y + 0.0*z + 0.0,   lambda x,y,z: -8.0*y + -4.0*x + -4.0*z + 4.0,   lambda x,y,z: 0.0*z + 0.0*x + -4.0*y + 0.0]
        dN8 = [lambda x,y,z: 0.0*x + 0.0*y + -4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + -4.0*z + 0.0,   lambda x,y,z: -8.0*z + -4.0*x + -4.0*y + 4.0]
        dN9 = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 4.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 4.0*y + 0.0]
        dN10 = [lambda x,y,z: 0.0*x + 0.0*y + 4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 4.0*x + 0.0*y + 0.0]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda x,y,z: 4.0,   lambda x,y,z: 4.0,   lambda x,y,z: 4.0]
        ddN2 = [lambda x,y,z: 4.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN3 = [lambda x,y,z: 0.0,   lambda x,y,z: 4.0,   lambda x,y,z: 0.0]
        ddN4 = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 4.0]
        ddN5 = [lambda x,y,z: -8.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN6 = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN7 = [lambda x,y,z: 0.0,   lambda x,y,z: -8.0,   lambda x,y,z: 0.0]
        ddN8 = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: -8.0]
        ddN9 = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN10 = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()