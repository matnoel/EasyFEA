# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Tri element module."""

from ... import np, plt

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

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: 1-xi-eta
        N2t = lambda xi,eta: xi
        N3t = lambda xi,eta: eta
        
        Ntild = np.array([N1t, N2t, N3t]).reshape(-1,1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: -1, lambda xi,eta: -1]
        dN2t = [lambda xi,eta: 1,  lambda xi,eta: 0]
        dN3t = [lambda xi,eta: 0,  lambda xi,eta: 1]

        dNtild = np.array([dN1t, dN2t, dN3t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        return super()._ddNtild()

    def _dddNtild(self) -> np.ndarray:
        return super()._dddNtild()

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()

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
        return [0,3,5,3,1,4,5,4,2,3,4,5]

    @property
    def faces(self) -> list[int]:
        return [0,3,1,4,2,5,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi,eta: -(1-xi-eta)*(1-2*(1-xi-eta))
        N2t = lambda xi,eta: -xi*(1-2*xi)
        N3t = lambda xi,eta: -eta*(1-2*eta)
        N4t = lambda xi,eta: 4*xi*(1-xi-eta)
        N5t = lambda xi,eta: 4*xi*eta
        N6t = lambda xi,eta: 4*eta*(1-xi-eta)
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        dN1t = [lambda xi,eta: 4*xi+4*eta-3,  lambda xi,eta: 4*xi+4*eta-3]
        dN2t = [lambda xi,eta: 4*xi-1,        lambda xi,eta: 0]
        dN3t = [lambda xi,eta: 0,              lambda xi,eta: 4*eta-1]
        dN4t = [lambda xi,eta: 4-8*xi-4*eta,  lambda xi,eta: -4*xi]
        dN5t = [lambda xi,eta: 4*eta,          lambda xi,eta: 4*xi]
        dN6t = [lambda xi,eta: -4*eta,         lambda xi,eta: 4-4*xi-8*eta]
        
        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:
        ddN1t = [lambda xi,eta: 4,  lambda xi,eta: 4]
        ddN2t = [lambda xi,eta: 4,  lambda xi,eta: 0]
        ddN3t = [lambda xi,eta: 0,  lambda xi,eta: 4]
        ddN4t = [lambda xi,eta: -8, lambda xi,eta: 0]
        ddN5t = [lambda xi,eta: 0,  lambda xi,eta: 0]
        ddN6t = [lambda xi,eta: 0,  lambda xi,eta: -8]
        
        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t])

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
        return list(np.array([10,1,4,10,4,5,10,5,6,10,6,7,10,7,8,10,8,9,10,9,1,2,5,6,3,7,8])-1)
    
    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,5,6,2,7,8,0]

    def _Ntild(self) -> np.ndarray:

        N1t = lambda xi, eta : -4.5*xi**3 + -4.5*eta**3 + -13.5*xi**2*eta + -13.5*xi*eta**2 + 9.0*xi**2 + 9.0*eta**2 + 18.0*xi*eta + -5.5*xi + -5.5*eta + 1.0
        N2t = lambda xi, eta : 4.5*xi**3 + 0.0*eta**3 + -1.093e-15*xi**2*eta + -8.119e-16*xi*eta**2 + -4.5*xi**2 + 0.0*eta**2 + 1.124e-15*xi*eta + 1.0*xi + 0.0*eta + 0.0
        N3t = lambda xi, eta : 0.0*xi**3 + 4.5*eta**3 + -3.747e-16*xi**2*eta + 2.998e-15*xi*eta**2 + 0.0*xi**2 + -4.5*eta**2 + -7.494e-16*xi*eta + 0.0*xi + 1.0*eta + 0.0
        N4t = lambda xi, eta : 13.5*xi**3 + 0.0*eta**3 + 27.0*xi**2*eta + 13.5*xi*eta**2 + -22.5*xi**2 + 0.0*eta**2 + -22.5*xi*eta + 9.0*xi + 0.0*eta + 0.0
        N5t = lambda xi, eta : -13.5*xi**3 + 0.0*eta**3 + -13.5*xi**2*eta + -4.247e-15*xi*eta**2 + 18.0*xi**2 + 0.0*eta**2 + 4.5*xi*eta + -4.5*xi + 0.0*eta + 0.0
        N6t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 13.5*xi**2*eta + 1.049e-14*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N7t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 0.0*xi**2*eta + 13.5*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N8t = lambda xi, eta : 0.0*xi**3 + -13.5*eta**3 + -1.499e-15*xi**2*eta + -13.5*xi*eta**2 + 0.0*xi**2 + 18.0*eta**2 + 4.5*xi*eta + 0.0*xi + -4.5*eta + 0.0
        N9t = lambda xi, eta : 0.0*xi**3 + 13.5*eta**3 + 13.5*xi**2*eta + 27.0*xi*eta**2 + 0.0*xi**2 + -22.5*eta**2 + -22.5*xi*eta + 0.0*xi + 9.0*eta + 0.0
        N10t = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + -27.0*xi**2*eta + -27.0*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + 27.0*xi*eta + 0.0*xi + 0.0*eta + 0.0
        
        Ntild = np.array([N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t, N9t, N10t]).reshape(-1, 1)

        return Ntild

    def _dNtild(self) -> np.ndarray:

        N1_xi = lambda xi, eta : -13.5*xi**2 + -27.0*xi*eta + -13.5*eta**2 + 18.0*xi + 18.0*eta + -5.5
        N2_xi = lambda xi, eta : 13.5*xi**2 + -2.186e-15*xi*eta + -8.119e-16*eta**2 + -9.0*xi + 1.124e-15*eta + 1.0
        N3_xi = lambda xi, eta : 0.0*xi**2 + -7.494e-16*xi*eta + 2.998e-15*eta**2 + 0.0*xi + -7.494e-16*eta + 0.0
        N4_xi = lambda xi, eta : 40.5*xi**2 + 54.0*xi*eta + 13.5*eta**2 + -45.0*xi + -22.5*eta + 9.0
        N5_xi = lambda xi, eta : -40.5*xi**2 + -27.0*xi*eta + -4.247e-15*eta**2 + 36.0*xi + 4.5*eta + -4.5
        N6_xi = lambda xi, eta : 0.0*xi**2 + 27.0*xi*eta + 1.049e-14*eta**2 + 0.0*xi + -4.5*eta + 0.0
        N7_xi = lambda xi, eta : 0.0*xi**2 + 0.0*xi*eta + 13.5*eta**2 + 0.0*xi + -4.5*eta + 0.0
        N8_xi = lambda xi, eta : 0.0*xi**2 + -2.998e-15*xi*eta + -13.5*eta**2 + 0.0*xi + 4.5*eta + 0.0
        N9_xi = lambda xi, eta : 0.0*xi**2 + 27.0*xi*eta + 27.0*eta**2 + 0.0*xi + -22.5*eta + 0.0
        N10_xi = lambda xi, eta : 0.0*xi**2 + -54.0*xi*eta + -27.0*eta**2 + 0.0*xi + 27.0*eta + 0.0

        N1_eta = lambda xi, eta : -13.5*eta**2 + -13.5*xi**2 + -27.0*xi*eta + 18.0*eta + 18.0*xi + -5.5
        N2_eta = lambda xi, eta : 0.0*eta**2 + -1.093e-15*xi**2 + -1.624e-15*xi*eta + 0.0*eta + 1.124e-15*xi + 0.0
        N3_eta = lambda xi, eta : 13.5*eta**2 + -3.747e-16*xi**2 + 5.995e-15*xi*eta + -9.0*eta + -7.494e-16*xi + 1.0
        N4_eta = lambda xi, eta : 0.0*eta**2 + 27.0*xi**2 + 27.0*xi*eta + 0.0*eta + -22.5*xi + 0.0
        N5_eta = lambda xi, eta : 0.0*eta**2 + -13.5*xi**2 + -8.493e-15*xi*eta + 0.0*eta + 4.5*xi + 0.0
        N6_eta = lambda xi, eta : 0.0*eta**2 + 13.5*xi**2 + 2.098e-14*xi*eta + 0.0*eta + -4.5*xi + 0.0
        N7_eta = lambda xi, eta : 0.0*eta**2 + 0.0*xi**2 + 27.0*xi*eta + 0.0*eta + -4.5*xi + 0.0
        N8_eta = lambda xi, eta : -40.5*eta**2 + -1.499e-15*xi**2 + -27.0*xi*eta + 36.0*eta + 4.5*xi + -4.5
        N9_eta = lambda xi, eta : 40.5*eta**2 + 13.5*xi**2 + 54.0*xi*eta + -45.0*eta + -22.5*xi + 9.0
        N10_eta = lambda xi, eta : 0.0*eta**2 + -27.0*xi**2 + -54.0*xi*eta + 0.0*eta + 27.0*xi + 0.0

        dN1t = [N1_xi, N1_eta]
        dN2t = [N2_xi, N2_eta]
        dN3t = [N3_xi, N3_eta]
        dN4t = [N4_xi, N4_eta]
        dN5t = [N5_xi, N5_eta]
        dN6t = [N6_xi, N6_eta]
        dN7t = [N7_xi, N7_eta]
        dN8t = [N8_xi, N8_eta]
        dN9t = [N9_xi, N9_eta]
        dN10t = [N10_xi, N10_eta]

        dNtild = np.array([dN1t, dN2t, dN3t, dN4t, dN5t, dN6t, dN7t, dN8t, dN9t, dN10t])

        return dNtild

    def _ddNtild(self) -> np.ndarray:

        N1_xi2 = lambda xi, eta : -27.0*xi + -27.0*eta + 18.0
        N2_xi2 = lambda xi, eta : 27.0*xi + -2.186e-15*eta + -9.0
        N3_xi2 = lambda xi, eta : 0.0*xi + -7.494e-16*eta + 0.0
        N4_xi2 = lambda xi, eta : 81.0*xi + 54.0*eta + -45.0
        N5_xi2 = lambda xi, eta : -81.0*xi + -27.0*eta + 36.0
        N6_xi2 = lambda xi, eta : 0.0*xi + 27.0*eta + 0.0
        N7_xi2 = lambda xi, eta : 0.0*xi + 0.0*eta + 0.0
        N8_xi2 = lambda xi, eta : 0.0*xi + -2.998e-15*eta + 0.0
        N9_xi2 = lambda xi, eta : 0.0*xi + 27.0*eta + 0.0
        N10_xi2 = lambda xi, eta : 0.0*xi + -54.0*eta + 0.0

        N1_eta2 = lambda xi, eta : -27.0*eta + -27.0*xi + 18.0
        N2_eta2 = lambda xi, eta : 0.0*eta + -1.624e-15*xi + 0.0
        N3_eta2 = lambda xi, eta : 27.0*eta + 5.995e-15*xi + -9.0
        N4_eta2 = lambda xi, eta : 0.0*eta + 27.0*xi + 0.0
        N5_eta2 = lambda xi, eta : 0.0*eta + -8.493e-15*xi + 0.0
        N6_eta2 = lambda xi, eta : 0.0*eta + 2.098e-14*xi + 0.0
        N7_eta2 = lambda xi, eta : 0.0*eta + 27.0*xi + 0.0
        N8_eta2 = lambda xi, eta : -81.0*eta + -27.0*xi + 36.0
        N9_eta2 = lambda xi, eta : 81.0*eta + 54.0*xi + -45.0
        N10_eta2 = lambda xi, eta : 0.0*eta + -54.0*xi + 0.0

        ddN1t = [N1_xi2, N1_eta2]
        ddN2t = [N2_xi2, N2_eta2]
        ddN3t = [N3_xi2, N3_eta2]
        ddN4t = [N4_xi2, N4_eta2]
        ddN5t = [N5_xi2, N5_eta2]
        ddN6t = [N6_xi2, N6_eta2]
        ddN7t = [N7_xi2, N7_eta2]
        ddN8t = [N8_xi2, N8_eta2]
        ddN9t = [N9_xi2, N9_eta2]
        ddN10t = [N10_xi2, N10_eta2]

        ddNtild = np.array([ddN1t, ddN2t, ddN3t, ddN4t, ddN5t, ddN6t, ddN7t, ddN8t, ddN9t, ddN10t])

        return ddNtild

    def _dddNtild(self) -> np.ndarray:
        
        N1_xi3 = lambda xi, eta : -27.0
        N2_xi3 = lambda xi, eta : 27.0
        N3_xi3 = lambda xi, eta : 0.0
        N4_xi3 = lambda xi, eta : 81.0
        N5_xi3 = lambda xi, eta : -81.0
        N6_xi3 = lambda xi, eta : 0.0
        N7_xi3 = lambda xi, eta : 0.0
        N8_xi3 = lambda xi, eta : 0.0
        N9_xi3 = lambda xi, eta : 0.0
        N10_xi3 = lambda xi, eta : 0.0

        N1_eta3 = lambda xi, eta : -27.0
        N2_eta3 = lambda xi, eta : 0.0
        N3_eta3 = lambda xi, eta : 27.0
        N4_eta3 = lambda xi, eta : 0.0
        N5_eta3 = lambda xi, eta : 0.0
        N6_eta3 = lambda xi, eta : 0.0
        N7_eta3 = lambda xi, eta : 0.0
        N8_eta3 = lambda xi, eta : -81.0
        N9_eta3 = lambda xi, eta : 81.0
        N10_eta3 = lambda xi, eta : 0.0

        dddN1t = [N1_xi3, N1_eta3]
        dddN2t = [N2_xi3, N2_eta3]
        dddN3t = [N3_xi3, N3_eta3]
        dddN4t = [N4_xi3, N4_eta3]
        dddN5t = [N5_xi3, N5_eta3]
        dddN6t = [N6_xi3, N6_eta3]
        dddN7t = [N7_xi3, N7_eta3]
        dddN8t = [N8_xi3, N8_eta3]
        dddN9t = [N9_xi3, N9_eta3]
        dddN10t = [N10_xi3, N10_eta3]

        dddNtild = np.array([dddN1t, dddN2t, dddN3t, dddN4t, dddN5t, dddN6t, dddN7t, dddN8t, dddN9t, dddN10t])

        return dddNtild

    def _ddddNtild(self) -> np.ndarray:
        return super()._ddddNtild()