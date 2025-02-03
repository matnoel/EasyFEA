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

        N1 = lambda xi,eta: (1-xi)*(1-eta)/4
        N2 = lambda xi,eta: (1+xi)*(1-eta)/4
        N3 = lambda xi,eta: (1+xi)*(1+eta)/4
        N4 = lambda xi,eta: (1-xi)*(1+eta)/4
        
        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda xi,eta: (eta-1)/4,  lambda xi,eta: (xi-1)/4]
        dN2 = [lambda xi,eta: (1-eta)/4,  lambda xi,eta: (-xi-1)/4]
        dN3 = [lambda xi,eta: (1+eta)/4,  lambda xi,eta: (1+xi)/4]
        dN4 = [lambda xi,eta: (-eta-1)/4, lambda xi,eta: (1-xi)/4]
        
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

        N1 = lambda xi,eta: (1-xi)*(1-eta)*(-1-xi-eta)/4
        N2 = lambda xi,eta: (1+xi)*(1-eta)*(-1+xi-eta)/4
        N3 = lambda xi,eta: (1+xi)*(1+eta)*(-1+xi+eta)/4
        N4 = lambda xi,eta: (1-xi)*(1+eta)*(-1-xi+eta)/4
        N5 = lambda xi,eta: (1-xi**2)*(1-eta)/2
        N6 = lambda xi,eta: (1+xi)*(1-eta**2)/2
        N7 = lambda xi,eta: (1-xi**2)*(1+eta)/2
        N8 = lambda xi,eta: (1-xi)*(1-eta**2)/2
        
        N =  np.array([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda xi,eta: (1-eta)*(2*xi+eta)/4,      lambda xi,eta: (1-xi)*(xi+2*eta)/4]
        dN2 = [lambda xi,eta: (1-eta)*(2*xi-eta)/4,      lambda xi,eta: -(1+xi)*(xi-2*eta)/4]
        dN3 = [lambda xi,eta: (1+eta)*(2*xi+eta)/4,      lambda xi,eta: (1+xi)*(xi+2*eta)/4]
        dN4 = [lambda xi,eta: -(1+eta)*(-2*xi+eta)/4,    lambda xi,eta: (1-xi)*(-xi+2*eta)/4]
        dN5 = [lambda xi,eta: -xi*(1-eta),               lambda xi,eta: -(1-xi**2)/2]
        dN6 = [lambda xi,eta: (1-eta**2)/2,               lambda xi,eta: -eta*(1+xi)]
        dN7 = [lambda xi,eta: -xi*(1+eta),               lambda xi,eta: (1-xi**2)/2]
        dN8 = [lambda xi,eta: -(1-eta**2)/2,              lambda xi,eta: -eta*(1-xi)]
                        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8])

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN2 = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN3 = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN4 = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN5 = [lambda xi,eta: -1+eta,     lambda xi,eta: 0]
        ddN6 = [lambda xi,eta: 0,          lambda xi,eta: -1-xi]
        ddN7 = [lambda xi,eta: -1-eta,     lambda xi,eta: 0]
        ddN8 = [lambda xi,eta: 0,          lambda xi,eta: -1+xi]
                        
        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()

    def _EulerBernoulli_N(self) -> np.ndarray:
        return super()._EulerBernoulli_N()

    def _EulerBernoulli_dN(self) -> np.ndarray:
        return super()._EulerBernoulli_dN()

    def _EulerBernoulli_ddN(self) -> np.ndarray:
        return super()._EulerBernoulli_ddN()