# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Tri element module."""

import numpy as np

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

    def _N(self) -> np.ndarray:

        N1 = lambda xi,eta: 1-xi-eta
        N2 = lambda xi,eta: xi
        N3 = lambda xi,eta: eta
        
        N = np.array([N1, N2, N3]).reshape(-1,1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda xi,eta: -1, lambda xi,eta: -1]
        dN2 = [lambda xi,eta: 1,  lambda xi,eta: 0]
        dN3 = [lambda xi,eta: 0,  lambda xi,eta: 1]

        dN = np.array([dN1, dN2, dN3])

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()

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
        return [0,3,5,
                3,1,4,
                5,4,2,
                3,4,5]

    @property
    def faces(self) -> list[int]:
        return [0,3,1,4,2,5,0]

    def _N(self) -> np.ndarray:

        N1 = lambda xi,eta: -(1-xi-eta)*(1-2*(1-xi-eta))
        N2 = lambda xi,eta: -xi*(1-2*xi)
        N3 = lambda xi,eta: -eta*(1-2*eta)
        N4 = lambda xi,eta: 4*xi*(1-xi-eta)
        N5 = lambda xi,eta: 4*xi*eta
        N6 = lambda xi,eta: 4*eta*(1-xi-eta)
        
        N = np.array([N1, N2, N3, N4, N5, N6]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda xi,eta: 4*xi+4*eta-3,  lambda xi,eta: 4*xi+4*eta-3]
        dN2 = [lambda xi,eta: 4*xi-1,        lambda xi,eta: 0]
        dN3 = [lambda xi,eta: 0,              lambda xi,eta: 4*eta-1]
        dN4 = [lambda xi,eta: 4-8*xi-4*eta,  lambda xi,eta: -4*xi]
        dN5 = [lambda xi,eta: 4*eta,          lambda xi,eta: 4*xi]
        dN6 = [lambda xi,eta: -4*eta,         lambda xi,eta: 4-4*xi-8*eta]
        
        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6])

        return dN

    def _ddN(self) -> np.ndarray:
        ddN1 = [lambda xi,eta: 4,  lambda xi,eta: 4]
        ddN2 = [lambda xi,eta: 4,  lambda xi,eta: 0]
        ddN3 = [lambda xi,eta: 0,  lambda xi,eta: 4]
        ddN4 = [lambda xi,eta: -8, lambda xi,eta: 0]
        ddN5 = [lambda xi,eta: 0,  lambda xi,eta: 0]
        ddN6 = [lambda xi,eta: 0,  lambda xi,eta: -8]
        
        ddNtild = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6])

        return ddNtild

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
        return [0,3,8,
                3,4,9,
                4,1,5,
                4,5,9,
                3,9,8,
                8,9,7,
                9,5,6,
                9,6,7,
                7,6,2]
    
    @property
    def faces(self) -> list[int]:
        return [0,3,4,1,5,6,2,7,8,0]

    def _N(self) -> np.ndarray:

        N1 = lambda xi, eta : -4.5*xi**3 + -4.5*eta**3 + -13.5*xi**2*eta + -13.5*xi*eta**2 + 9.0*xi**2 + 9.0*eta**2 + 18.0*xi*eta + -5.5*xi + -5.5*eta + 1.0
        N2 = lambda xi, eta : 4.5*xi**3 + 0.0*eta**3 + -1.093e-15*xi**2*eta + -8.119e-16*xi*eta**2 + -4.5*xi**2 + 0.0*eta**2 + 1.124e-15*xi*eta + 1.0*xi + 0.0*eta + 0.0
        N3 = lambda xi, eta : 0.0*xi**3 + 4.5*eta**3 + -3.747e-16*xi**2*eta + 2.998e-15*xi*eta**2 + 0.0*xi**2 + -4.5*eta**2 + -7.494e-16*xi*eta + 0.0*xi + 1.0*eta + 0.0
        N4 = lambda xi, eta : 13.5*xi**3 + 0.0*eta**3 + 27.0*xi**2*eta + 13.5*xi*eta**2 + -22.5*xi**2 + 0.0*eta**2 + -22.5*xi*eta + 9.0*xi + 0.0*eta + 0.0
        N5 = lambda xi, eta : -13.5*xi**3 + 0.0*eta**3 + -13.5*xi**2*eta + -4.247e-15*xi*eta**2 + 18.0*xi**2 + 0.0*eta**2 + 4.5*xi*eta + -4.5*xi + 0.0*eta + 0.0
        N6 = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 13.5*xi**2*eta + 1.049e-14*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N7 = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + 0.0*xi**2*eta + 13.5*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + -4.5*xi*eta + 0.0*xi + 0.0*eta + 0.0
        N8 = lambda xi, eta : 0.0*xi**3 + -13.5*eta**3 + -1.499e-15*xi**2*eta + -13.5*xi*eta**2 + 0.0*xi**2 + 18.0*eta**2 + 4.5*xi*eta + 0.0*xi + -4.5*eta + 0.0
        N9 = lambda xi, eta : 0.0*xi**3 + 13.5*eta**3 + 13.5*xi**2*eta + 27.0*xi*eta**2 + 0.0*xi**2 + -22.5*eta**2 + -22.5*xi*eta + 0.0*xi + 9.0*eta + 0.0
        N10 = lambda xi, eta : 0.0*xi**3 + 0.0*eta**3 + -27.0*xi**2*eta + -27.0*xi*eta**2 + 0.0*xi**2 + 0.0*eta**2 + 27.0*xi*eta + 0.0*xi + 0.0*eta + 0.0
        
        N = np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

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

        dN1 = [N1_xi, N1_eta]
        dN2 = [N2_xi, N2_eta]
        dN3 = [N3_xi, N3_eta]
        dN4 = [N4_xi, N4_eta]
        dN5 = [N5_xi, N5_eta]
        dN6 = [N6_xi, N6_eta]
        dN7 = [N7_xi, N7_eta]
        dN8 = [N8_xi, N8_eta]
        dN9 = [N9_xi, N9_eta]
        dN10 = [N10_xi, N10_eta]

        dN = np.array([dN1, dN2, dN3, dN4, dN5, dN6, dN7, dN8, dN9, dN10])

        return dN

    def _ddN(self) -> np.ndarray:

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

        ddN1 = [N1_xi2, N1_eta2]
        ddN2 = [N2_xi2, N2_eta2]
        ddN3 = [N3_xi2, N3_eta2]
        ddN4 = [N4_xi2, N4_eta2]
        ddN5 = [N5_xi2, N5_eta2]
        ddN6 = [N6_xi2, N6_eta2]
        ddN7 = [N7_xi2, N7_eta2]
        ddN8 = [N8_xi2, N8_eta2]
        ddN9 = [N9_xi2, N9_eta2]
        ddN10 = [N10_xi2, N10_eta2]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4, ddN5, ddN6, ddN7, ddN8, ddN9, ddN10])

        return ddN

    def _dddN(self) -> np.ndarray:
        
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

        dddN1 = [N1_xi3, N1_eta3]
        dddN2 = [N2_xi3, N2_eta3]
        dddN3 = [N3_xi3, N3_eta3]
        dddN4 = [N4_xi3, N4_eta3]
        dddN5 = [N5_xi3, N5_eta3]
        dddN6 = [N6_xi3, N6_eta3]
        dddN7 = [N7_xi3, N7_eta3]
        dddN8 = [N8_xi3, N8_eta3]
        dddN9 = [N9_xi3, N9_eta3]
        dddN10 = [N10_xi3, N10_eta3]

        dddN = np.array([dddN1, dddN2, dddN3, dddN4, dddN5, dddN6, dddN7, dddN8, dddN9, dddN10])

        return dddN

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()