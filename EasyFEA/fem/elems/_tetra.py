# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Tetra element module."""

from ... import np, plt

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
        return [0,1,2,0,3,1,0,2,3,1,3,2]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 1-x-y-z
        N2t = lambda x,y,z: x
        N3t = lambda x,y,z: y
        N4t = lambda x,y,z: z

        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -1,   lambda x,y,z: -1,   lambda x,y,z: -1]
        dN2t = [lambda x,y,z: 1,    lambda x,y,z: 0,    lambda x,y,z: 0]
        dN3t = [lambda x,y,z: 0,    lambda x,y,z: 1,    lambda x,y,z: 0]
        dN4t = [lambda x,y,z: 0,    lambda x,y,z: 0,    lambda x,y,z: 1]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

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
        return [0,4,1,5,2,6,0,7,3,9,1,4,0,6,2,8,3,7,1,9,3,8,2,5]
    
    @property
    def segments(self) -> np.ndarray:
        return np.array([[0,1],[0,3],[3,1],[2,0],[2,3],[2,1]])

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: 2.0*x**2 + 2.0*y**2 + 2.0*z**2 + 4.0*x*y + 4.0*x*z + 4.0*y*z + -3.0*x + -3.0*y + -3.0*z + 1.0
        N2t = lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0
        N3t = lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0
        N4t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z**2 + 0.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + -1.0*z + 0.0
        N5t = lambda x,y,z: -4.0*x**2 + 0.0*y**2 + 0.0*z**2 + -4.0*x*y + -4.0*x*z + 0.0*y*z + 4.0*x + 0.0*y + 0.0*z + 0.0
        N6t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 4.0*x*y + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N7t = lambda x,y,z: 0.0*x**2 + -4.0*y**2 + 0.0*z**2 + -4.0*x*y + 0.0*x*z + -4.0*y*z + 0.0*x + 4.0*y + 0.0*z + 0.0
        N8t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -4.0*z**2 + 0.0*x*y + -4.0*x*z + -4.0*y*z + 0.0*x + 0.0*y + 4.0*z + 0.0
        N9t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 0.0*x*z + 4.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0
        N10t = lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*y + 4.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: 4.0*x + 4.0*y + 4.0*z + -3.0,   lambda x,y,z: 4.0*y + 4.0*x + 4.0*z + -3.0,   lambda x,y,z: 4.0*z + 4.0*x + 4.0*y + -3.0]
        dN2t = [lambda x,y,z: 4.0*x + 0.0*y + 0.0*z + -1.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 4.0*y + 0.0*x + 0.0*z + -1.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN4t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 4.0*z + 0.0*x + 0.0*y + -1.0]
        dN5t = [lambda x,y,z: -8.0*x + -4.0*y + -4.0*z + 4.0,   lambda x,y,z: 0.0*y + -4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + -4.0*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*x + 4.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 4.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN7t = [lambda x,y,z: 0.0*x + -4.0*y + 0.0*z + 0.0,   lambda x,y,z: -8.0*y + -4.0*x + -4.0*z + 4.0,   lambda x,y,z: 0.0*z + 0.0*x + -4.0*y + 0.0]
        dN8t = [lambda x,y,z: 0.0*x + 0.0*y + -4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + -4.0*z + 0.0,   lambda x,y,z: -8.0*z + -4.0*x + -4.0*y + 4.0]
        dN9t = [lambda x,y,z: 0.0*x + 0.0*y + 0.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 4.0*z + 0.0,   lambda x,y,z: 0.0*z + 0.0*x + 4.0*y + 0.0]
        dN10t = [lambda x,y,z: 0.0*x + 0.0*y + 4.0*z + 0.0,   lambda x,y,z: 0.0*y + 0.0*x + 0.0*z + 0.0,   lambda x,y,z: 0.0*z + 4.0*x + 0.0*y + 0.0]


        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: 4.0,   lambda x,y,z: 4.0,   lambda x,y,z: 4.0]
        ddN2t = [lambda x,y,z: 4.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN3t = [lambda x,y,z: 0.0,   lambda x,y,z: 4.0,   lambda x,y,z: 0.0]
        ddN4t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 4.0]
        ddN5t = [lambda x,y,z: -8.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN6t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN7t = [lambda x,y,z: 0.0,   lambda x,y,z: -8.0,   lambda x,y,z: 0.0]
        ddN8t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: -8.0]
        ddN9t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]
        ddN10t = [lambda x,y,z: 0.0,   lambda x,y,z: 0.0,   lambda x,y,z: 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()