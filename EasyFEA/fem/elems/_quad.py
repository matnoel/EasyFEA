# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Quad element module."""

from ... import np, plt

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
        return [0,1,3,1,2,3]

    @property
    def faces(self) -> list[int]:
        return [0,1,2,3,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: (1-xi)*(1-eta)/4
        N2t = lambda xi,eta: (1+xi)*(1-eta)/4
        N3t = lambda xi,eta: (1+xi)*(1+eta)/4
        N4t = lambda xi,eta: (1-xi)*(1+eta)/4
        
        Ntild = np.array([N1t, N2t, N3t, N4t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: (eta-1)/4,  lambda xi,eta: (xi-1)/4]
        dN2t = [lambda xi,eta: (1-eta)/4,  lambda xi,eta: (-xi-1)/4]
        dN3t = [lambda xi,eta: (1+eta)/4,  lambda xi,eta: (1+xi)/4]
        dN4t = [lambda xi,eta: (-eta-1)/4, lambda xi,eta: (1-xi)/4]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()
    
    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()
    
    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

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
        return [4,5,7,5,6,7,0,4,7,4,1,5,5,2,6,6,3,7]

    @property
    def faces(self) -> list[int]:
        return [0,4,1,5,2,6,3,7,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: (1-xi)*(1-eta)*(-1-xi-eta)/4
        N2t = lambda xi,eta: (1+xi)*(1-eta)*(-1+xi-eta)/4
        N3t = lambda xi,eta: (1+xi)*(1+eta)*(-1+xi+eta)/4
        N4t = lambda xi,eta: (1-xi)*(1+eta)*(-1-xi+eta)/4
        N5t = lambda xi,eta: (1-xi**2)*(1-eta)/2
        N6t = lambda xi,eta: (1+xi)*(1-eta**2)/2
        N7t = lambda xi,eta: (1-xi**2)*(1+eta)/2
        N8t = lambda xi,eta: (1-xi)*(1-eta**2)/2
        
        Ntild =  np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]).reshape(-1, 1)

        return Ntild
    
    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: (1-eta)*(2*xi+eta)/4,      lambda xi,eta: (1-xi)*(xi+2*eta)/4]
        dN2t = [lambda xi,eta: (1-eta)*(2*xi-eta)/4,      lambda xi,eta: -(1+xi)*(xi-2*eta)/4]
        dN3t = [lambda xi,eta: (1+eta)*(2*xi+eta)/4,      lambda xi,eta: (1+xi)*(xi+2*eta)/4]
        dN4t = [lambda xi,eta: -(1+eta)*(-2*xi+eta)/4,    lambda xi,eta: (1-xi)*(-xi+2*eta)/4]
        dN5t = [lambda xi,eta: -xi*(1-eta),               lambda xi,eta: -(1-xi**2)/2]
        dN6t = [lambda xi,eta: (1-eta**2)/2,               lambda xi,eta: -eta*(1+xi)]
        dN7t = [lambda xi,eta: -xi*(1+eta),               lambda xi,eta: (1-xi**2)/2]
        dN8t = [lambda xi,eta: -(1-eta**2)/2,              lambda xi,eta: -eta*(1-xi)]
                        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        ddN1t = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN2t = [lambda xi,eta: (1-eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN3t = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1+xi)/2]
        ddN4t = [lambda xi,eta: (1+eta)/2,  lambda xi,eta: (1-xi)/2]
        ddN5t = [lambda xi,eta: -1+eta,     lambda xi,eta: 0]
        ddN6t = [lambda xi,eta: 0,          lambda xi,eta: -1-xi]
        ddN7t = [lambda xi,eta: -1-eta,     lambda xi,eta: 0]
        ddN8t = [lambda xi,eta: 0,          lambda xi,eta: -1+xi]
                        
        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

    def _Nvtild(self) -> np.ndarray:
        return super()._Nvtild()

    def dNvtild(self) -> np.ndarray:
        return super().dNvtild()

    def _ddNvtild(self) -> np.ndarray:
        return super()._ddNvtild()