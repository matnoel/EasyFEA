# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Prism element module."""

from ... import np, plt

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

    def _Ntild(self) -> np.ndarray:        

        N1t = lambda x,y,z: 0.5*x*z + 0.5*y*z + -0.5*x + -0.5*y + -0.5*z + 0.5
        N2t = lambda x,y,z: -0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N3t = lambda x,y,z: 0.0*x*z + -0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0
        N4t = lambda x,y,z: -0.5*x*z + -0.5*y*z + -0.5*x + -0.5*y + 0.5*z + 0.5
        N5t = lambda x,y,z: 0.5*x*z + 0.0*y*z + 0.5*x + 0.0*y + 0.0*z + 0.0
        N6t = lambda x,y,z: 0.0*x*z + 0.5*y*z + 0.0*x + 0.5*y + 0.0*z + 0.0        

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:        

        dN1t = [lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*z + -0.5, lambda x,y,z: 0.5*x + 0.5*y + -0.5]
        dN2t = [lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -0.5*z + 0.5, lambda x,y,z: 0.0*x + -0.5*y + 0.0]
        dN4t = [lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*z + -0.5, lambda x,y,z: -0.5*x + -0.5*y + 0.5]
        dN5t = [lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.5*z + 0.5, lambda x,y,z: 0.0*x + 0.5*y + 0.0]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])        

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

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

    def _Ntild(self) -> np.ndarray:

        N1t = lambda x,y,z: -1.0*x**2*z + -1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + -2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + 1.5*x*z + 1.5*y*z + -1.0*x + -1.0*y + -0.5*z + 2.0*x*y + 0.0
        N2t = lambda x,y,z: -1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N3t = lambda x,y,z: 0.0*x**2*z + -1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N4t = lambda x,y,z: 1.0*x**2*z + 1.0*y**2*z + -0.5*z**2*x + -0.5*z**2*y + 2.0*x*y*z + 1.0*x**2 + 1.0*y**2 + 0.5*z**2 + -1.5*x*z + -1.5*y*z + -1.0*x + -1.0*y + 0.5*z + 2.0*x*y + 0.0
        N5t = lambda x,y,z: 1.0*x**2*z + 0.0*y**2*z + 0.5*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 1.0*x**2 + 0.0*y**2 + 0.0*z**2 + -0.5*x*z + 0.0*y*z + -1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N6t = lambda x,y,z: 0.0*x**2*z + 1.0*y**2*z + 0.0*z**2*x + 0.5*z**2*y + 0.0*x*y*z + 0.0*x**2 + 1.0*y**2 + 0.0*z**2 + 0.0*x*z + -0.5*y*z + 0.0*x + -1.0*y + 0.0*z + 0.0*x*y + 0.0
        N7t = lambda x,y,z: 2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + -2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N8t = lambda x,y,z: 0.0*x**2*z + 2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + -2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N9t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 1.0*z**2*x + 1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + -1.0*z**2 + 0.0*x*z + 0.0*y*z + -1.0*x + -1.0*y + 0.0*z + 0.0*x*y + 1.0
        N10t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0
        N11t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + -1.0*z**2*x + 0.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 1.0*x + 0.0*y + 0.0*z + 0.0*x*y + 0.0
        N12t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + -1.0*z**2*y + 0.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 1.0*y + 0.0*z + 0.0*x*y + 0.0
        N13t = lambda x,y,z: -2.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + -2.0*x**2 + 0.0*y**2 + 0.0*z**2 + 2.0*x*z + 0.0*y*z + 2.0*x + 0.0*y + 0.0*z + -2.0*x*y + 0.0
        N14t = lambda x,y,z: 0.0*x**2*z + -2.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + -2.0*x*y*z + 0.0*x**2 + -2.0*y**2 + 0.0*z**2 + 0.0*x*z + 2.0*y*z + 0.0*x + 2.0*y + 0.0*z + -2.0*x*y + 0.0
        N15t = lambda x,y,z: 0.0*x**2*z + 0.0*y**2*z + 0.0*z**2*x + 0.0*z**2*y + 2.0*x*y*z + 0.0*x**2 + 0.0*y**2 + 0.0*z**2 + 0.0*x*z + 0.0*y*z + 0.0*x + 0.0*y + 0.0*z + 2.0*x*y + 0.0

        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t, N11t, N12t, N13t, N14t, N15t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda x,y,z: -2.0*x*z + -0.5*z**2 + -2.0*y*z + 2.0*x + 1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: -2.0*y*z + -0.5*z**2 + -2.0*x*z + 2.0*y + 1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: -1.0*x**2 + -1.0*y**2 + -1.0*z*x + -1.0*z*y + -2.0*x*y + 1.0*z + 1.5*x + 1.5*y + -0.5]
        dN2t = [lambda x,y,z: -2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + 0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: -1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.5*x + 0.0*y + 0.0]
        dN3t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: -2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + 0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + -1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.5*y + 0.0]
        dN4t = [lambda x,y,z: 2.0*x*z + -0.5*z**2 + 2.0*y*z + 2.0*x + -1.5*z + -1.0 + 2.0*y,
                lambda x,y,z: 2.0*y*z + -0.5*z**2 + 2.0*x*z + 2.0*y + -1.5*z + -1.0 + 2.0*x,
                lambda x,y,z: 1.0*x**2 + 1.0*y**2 + -1.0*z*x + -1.0*z*y + 2.0*x*y + 1.0*z + -1.5*x + -1.5*y + 0.5]
        dN5t = [lambda x,y,z: 2.0*x*z + 0.5*z**2 + 0.0*y*z + 2.0*x + -0.5*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 1.0*x**2 + 0.0*y**2 + 1.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + -0.5*x + 0.0*y + 0.0]
        dN6t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 2.0*y*z + 0.5*z**2 + 0.0*x*z + 2.0*y + -0.5*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 1.0*y**2 + 0.0*z*x + 1.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + -0.5*y + 0.0]
        dN7t = [lambda x,y,z: 4.0*x*z + 0.0*z**2 + 2.0*y*z + -4.0*x + -2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: 2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + -2.0*x + 0.0*y + 0.0]
        dN8t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: 4.0*y*z + 0.0*z**2 + 2.0*x*z + -4.0*y + -2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + 2.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + -2.0*y + 0.0]
        dN9t = [lambda x,y,z: 0.0*x*z + 1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + -1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + -1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 2.0*z*x + 2.0*z*y + 0.0*x*y + -2.0*z + 0.0*x + 0.0*y + 0.0]
        dN10t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN11t = [lambda x,y,z: 0.0*x*z + -1.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 1.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 0.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + -2.0*z*x + 0.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN12t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 0.0*y*z + 0.0*x + 0.0*z + 0.0 + 0.0*y,
                lambda x,y,z: 0.0*y*z + -1.0*z**2 + 0.0*x*z + 0.0*y + 0.0*z + 1.0 + 0.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + -2.0*z*y + 0.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        dN13t = [lambda x,y,z: -4.0*x*z + 0.0*z**2 + -2.0*y*z + -4.0*x + 2.0*z + 2.0 + -2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + -2.0*x*z + 0.0*y + 0.0*z + 0.0 + -2.0*x,
                lambda x,y,z: -2.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 2.0*x + 0.0*y + 0.0]
        dN14t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + -2.0*y*z + 0.0*x + 0.0*z + 0.0 + -2.0*y,
                lambda x,y,z: -4.0*y*z + 0.0*z**2 + -2.0*x*z + -4.0*y + 2.0*z + 2.0 + -2.0*x,
                lambda x,y,z: 0.0*x**2 + -2.0*y**2 + 0.0*z*x + 0.0*z*y + -2.0*x*y + 0.0*z + 0.0*x + 2.0*y + 0.0]
        dN15t = [lambda x,y,z: 0.0*x*z + 0.0*z**2 + 2.0*y*z + 0.0*x + 0.0*z + 0.0 + 2.0*y,
                lambda x,y,z: 0.0*y*z + 0.0*z**2 + 2.0*x*z + 0.0*y + 0.0*z + 0.0 + 2.0*x,
                lambda x,y,z: 0.0*x**2 + 0.0*y**2 + 0.0*z*x + 0.0*z*y + 2.0*x*y + 0.0*z + 0.0*x + 0.0*y + 0.0]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t, dN11t, dN12t, dN13t, dN14t, dN15t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN2t = [lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN3t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN4t = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: -1.0*x + -1.0*y + 1.0]
        ddN5t = [lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 1.0*x + 0.0*y + 0.0]
        ddN6t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*z + 2.0, lambda x,y,z: 0.0*x + 1.0*y + 0.0]
        ddN7t = [lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN8t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN9t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 2.0*x + 2.0*y + -2.0]
        ddN10t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN11t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -2.0*x + 0.0*y + 0.0]
        ddN12t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + -2.0*y + 0.0]
        ddN13t = [lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN14t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: -4.0*z + -4.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]
        ddN15t = [lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*z + 0.0, lambda x,y,z: 0.0*x + 0.0*y + 0.0]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t, ddN11t, ddN12t, ddN13t, ddN14t, ddN15t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()