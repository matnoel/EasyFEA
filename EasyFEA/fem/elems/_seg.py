# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Seg element module."""

import numpy as np

from .._group_elems import _GroupElem

class SEG2(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----+----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles
    
    @property
    def faces(self) -> list[int]:
        return [0,1]

    def _N(self) -> np.ndarray:

        N1 = lambda x: 0.5*(1-x)
        N2 = lambda x: 0.5*(1+x)

        N = np.array([N1, N2]).reshape(-1, 1)

        return N
    
    def _dN(self) -> np.ndarray:

        dN1 = [lambda x: -0.5]
        dN2 = [lambda x: 0.5]

        dN = np.array([dN1, dN2]).reshape(-1,1)

        return dN

    def _ddN(self) -> np.ndarray:
        return super()._ddN()

    def _dddN(self) -> np.ndarray:
        return super()._dddN()
    
    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()

    def _EulerBernoulli_N(self) -> np.ndarray:

        phi_1 = lambda x : 0.5 + -0.75*x + 0.0*x**2 + 0.25*x**3
        psi_1 = lambda x : 0.125 + -0.125*x + -0.125*x**2 + 0.125*x**3
        phi_2 = lambda x : 0.5 + 0.75*x + 0.0*x**2 + -0.25*x**3
        psi_2 = lambda x : -0.125 + -0.125*x + 0.125*x**2 + 0.125*x**3

        Nv = np.array([phi_1, psi_1, phi_2, psi_2]).reshape(-1,1)

        return Nv

    def _EulerBernoulli_dN(self) -> np.ndarray:

        phi_1_x = lambda x : -0.75 + 0.0*x + 0.75*x**2
        psi_1_x = lambda x : -0.125 + -0.25*x + 0.375*x**2
        phi_2_x = lambda x : 0.75 + 0.0*x + -0.75*x**2
        psi_2_x = lambda x : -0.125 + 0.25*x + 0.375*x**2

        dNv = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x]).reshape(-1,1)

        return dNv

    def _EulerBernoulli_ddN(self) -> np.ndarray:

        phi_1_xx = lambda x : 0.0 + 1.5*x
        psi_1_xx = lambda x : -0.25 + 0.75*x
        phi_2_xx = lambda x : 0.0 + -1.5*x
        psi_2_xx = lambda x : 0.25 + 0.75*x

        ddNvtild = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx]).reshape(-1,1)

        return ddNvtild

class SEG3(_GroupElem):
    #      v
    #      ^
    #      |
    #      |
    # 0----2----1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles
    
    @property
    def faces(self) -> list[int]:
        return [0,2,1]

    def _N(self) -> np.ndarray:

        N1 = lambda x: -0.5*(1-x)*x
        N2 = lambda x: 0.5*(1+x)*x
        N3 = lambda x: (1+x)*(1-x)

        N = np.array([N1, N2, N3]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda x: x-0.5]
        dN2 = [lambda x: x+0.5]
        dN3 = [lambda x: -2*x]

        dN = np.array([dN1, dN2, dN3]).reshape(-1,1)

        return dN

    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda x: 1]
        ddN2 = [lambda x: 1]
        ddN3 = [lambda x: -2]

        ddN = np.array([ddN1, ddN2, ddN3])

        return ddN

    def _dddN(self) -> np.ndarray:
        return super()._dddN()

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()
        
    def _EulerBernoulli_N(self) -> np.ndarray:

        phi_1 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + -1.25*x**3 + -0.5*x**4 + 0.75*x**5
        psi_1 = lambda x : 0.0 + 0.0*x + 0.125*x**2 + -0.125*x**3 + -0.125*x**4 + 0.125*x**5
        phi_2 = lambda x : 0.0 + 0.0*x + 1.0*x**2 + 1.25*x**3 + -0.5*x**4 + -0.75*x**5
        psi_2 = lambda x : 0.0 + 0.0*x + -0.125*x**2 + -0.125*x**3 + 0.125*x**4 + 0.125*x**5
        phi_3 = lambda x : 1.0 + 0.0*x + -2.0*x**2 + 0.0*x**3 + 1.0*x**4 + 0.0*x**5
        psi_3 = lambda x : 0.0 + 0.5*x + 0.0*x**2 + -1.0*x**3 + 0.0*x**4 + 0.5*x**5

        Nv = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3]).reshape(-1,1)

        return Nv

    def _EulerBernoulli_dN(self) -> np.ndarray:

        phi_1_x = lambda x : 0.0 + 2.0*x + -3.75*x**2 + -2.0*x**3 + 3.75*x**4
        psi_1_x = lambda x : 0.0 + 0.25*x + -0.375*x**2 + -0.5*x**3 + 0.625*x**4
        phi_2_x = lambda x : 0.0 + 2.0*x + 3.75*x**2 + -2.0*x**3 + -3.75*x**4
        psi_2_x = lambda x : 0.0 + -0.25*x + -0.375*x**2 + 0.5*x**3 + 0.625*x**4
        phi_3_x = lambda x : 0.0 + -4.0*x + 0.0*x**2 + 4.0*x**3 + 0.0*x**4
        psi_3_x = lambda x : 0.5 + 0.0*x + -3.0*x**2 + 0.0*x**3 + 2.5*x**4

        dNv = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x]).reshape(-1,1)

        return dNv

    def _EulerBernoulli_ddN(self) -> np.ndarray:
        
        phi_1_xx = lambda x : 2.0 + -7.5*x + -6.0*x**2 + 15.0*x**3
        psi_1_xx = lambda x : 0.25 + -0.75*x + -1.5*x**2 + 2.5*x**3
        phi_2_xx = lambda x : 2.0 + 7.5*x + -6.0*x**2 + -15.0*x**3
        psi_2_xx = lambda x : -0.25 + -0.75*x + 1.5*x**2 + 2.5*x**3
        phi_3_xx = lambda x : -4.0 + 0.0*x + 12.0*x**2 + 0.0*x**3
        psi_3_xx = lambda x : 0.0 + -6.0*x + 0.0*x**2 + 10.0*x**3

        ddNv = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx]).reshape(-1,1)

        return ddNv

class SEG4(_GroupElem):
    #       v
    #       ^
    #       |
    #       |
    # 0---2-+-3---1 --> u

    def __init__(self, gmshId: int, connect: np.ndarray, coordoGlob: np.ndarray, nodes: np.ndarray):

        super().__init__(gmshId, connect, coordoGlob, nodes)

    @property
    def origin(self) -> list[int]:
        return [-1]

    @property
    def triangles(self) -> list[int]:
        return super().triangles

    @property
    def faces(self) -> list[int]:
        return [0,2,3,1]

    def _N(self) -> np.ndarray:

        N1 = lambda x : -0.5625*x**3 + 0.5625*x**2 + 0.0625*x - 0.0625
        N2 = lambda x : 0.5625*x**3 + 0.5625*x**2 - 0.0625*x - 0.0625
        N3 = lambda x : 1.6875*x**3 - 0.5625*x**2 - 1.6875*x + 0.5625
        N4 = lambda x : -1.6875*x**3 - 0.5625*x**2 + 1.6875*x + 0.5625

        N = np.array([N1, N2, N3, N4]).reshape(-1, 1)

        return N

    def _dN(self) -> np.ndarray:

        dN1 = [lambda x : -1.6875*x**2 + 1.125*x + 0.0625]
        dN2 = [lambda x : 1.6875*x**2 + 1.125*x - 0.0625]
        dN3 = [lambda x : 5.0625*x**2 - 1.125*x - 1.6875]
        dN4 = [lambda x : -5.0625*x**2 - 1.125*x + 1.6875]

        dN = np.array([dN1, dN2, dN3, dN4])

        return dN
    
    def _ddN(self) -> np.ndarray:

        ddN1 = [lambda x : -3.375*x + 1.125]
        ddN2 = [lambda x : 3.375*x + 1.125]
        ddN3 = [lambda x : 10.125*x - 1.125]
        ddN4 = [lambda x : -10.125*x - 1.125]

        ddN = np.array([ddN1, ddN2, ddN3, ddN4])

        return ddN

    def _dddN(self) -> np.ndarray:
        
        dddN1 = [lambda x : -3.375]
        dddN2 = [lambda x : 3.375]
        dddN3 = [lambda x : 10.125]
        dddN4 = [lambda x : -10.125]

        dddN = np.array([dddN1, dddN2, dddN3, dddN4])

        return dddN

    def _ddddN(self) -> np.ndarray:
        return super()._ddddN()

    def _EulerBernoulli_N(self) -> np.ndarray:

        phi_1 = lambda x : 0.025390624999999556 + -0.029296874999997335*x + -0.4746093750000018*x**2 + 0.548828124999992*x**3 + 2.3730468750000036*x**4 + -2.7597656249999916*x**5 + -1.4238281250000018*x**6 + 1.740234374999997*x**7
        psi_1 = lambda x : 0.0019531250000000555 + -0.0019531249999997224*x + -0.03710937500000017*x**2 + 0.03710937499999917*x**3 + 0.19335937500000028*x**4 + -0.19335937499999917*x**5 + -0.15820312500000014*x**6 + 0.15820312499999972*x**7
        phi_2 = lambda x : 0.025390625 + 0.02929687499999778*x + -0.47460937499999734*x**2 + -0.5488281249999911*x**3 + 2.373046874999995*x**4 + 2.75976562499999*x**5 + -1.4238281249999976*x**6 + -1.7402343749999962*x**7
        psi_2 = lambda x : -0.001953125 + -0.0019531249999998335*x + 0.03710937499999983*x**2 + 0.03710937499999928*x**3 + -0.19335937499999967*x**4 + -0.19335937499999908*x**5 + 0.15820312499999983*x**6 + 0.15820312499999964*x**7
        phi_3 = lambda x : 0.474609375 + -2.373046874999991*x + 0.4746093749999929*x**2 + 9.017578124999972*x**3 + -2.3730468749999845*x**4 + -10.91601562499997*x**5 + 1.4238281249999922*x**6 + 4.271484374999989*x**7
        psi_3 = lambda x : 0.05273437499999978 + -0.1582031249999971*x + -0.5800781250000018*x**2 + 1.7402343749999911*x**3 + 1.001953125000004*x**4 + -3.0058593749999907*x**5 + -0.4746093750000019*x**6 + 1.4238281249999967*x**7
        phi_4 = lambda x : 0.4746093749999991 + 2.3730468749999902*x + 0.4746093750000089*x**2 + -9.017578124999972*x**3 + -2.373046875000015*x**4 + 10.916015624999972*x**5 + 1.423828125000007*x**6 + -4.27148437499999*x**7
        psi_4 = lambda x : -0.05273437500000022 + -0.15820312499999734*x + 0.5800781249999978*x**2 + 1.7402343749999911*x**3 + -1.0019531249999953*x**4 + -3.0058593749999902*x**5 + 0.47460937499999767*x**6 + 1.4238281249999964*x**7

        Nv = np.array([phi_1, psi_1, phi_2, psi_2, phi_3, psi_3, phi_4, psi_4]).reshape(-1,1)

        return Nv
        
    def _EulerBernoulli_dN(self) -> np.ndarray:

        phi_1_x = lambda x : -0.029296874999997335 + -0.9492187500000036*x + 1.646484374999976*x**2 + 9.492187500000014*x**3 + -13.798828124999957*x**4 + -8.54296875000001*x**5 + 12.181640624999979*x**6
        psi_1_x = lambda x : -0.0019531249999997224 + -0.07421875000000033*x + 0.1113281249999975*x**2 + 0.7734375000000011*x**3 + -0.9667968749999958*x**4 + -0.9492187500000009*x**5 + 1.107421874999998*x**6
        phi_2_x = lambda x : 0.02929687499999778 + -0.9492187499999947*x + -1.6464843749999734*x**2 + 9.49218749999998*x**3 + 13.798828124999948*x**4 + -8.542968749999986*x**5 + -12.181640624999973*x**6
        psi_2_x = lambda x : -0.0019531249999998335 + 0.07421874999999967*x + 0.11132812499999784*x**2 + -0.7734374999999987*x**3 + -0.9667968749999954*x**4 + 0.949218749999999*x**5 + 1.1074218749999976*x**6
        phi_3_x = lambda x : -2.373046874999991 + 0.9492187499999858*x + 27.052734374999915*x**2 + -9.492187499999938*x**3 + -54.58007812499985*x**4 + 8.542968749999954*x**5 + 29.900390624999925*x**6
        psi_3_x = lambda x : -0.1582031249999971 + -1.1601562500000036*x + 5.220703124999973*x**2 + 4.007812500000016*x**3 + -15.029296874999954*x**4 + -2.8476562500000115*x**5 + 9.966796874999977*x**6
        phi_4_x = lambda x : 2.3730468749999902 + 0.9492187500000178*x + -27.052734374999915*x**2 + -9.49218750000006*x**3 + 54.58007812499986*x**4 + 8.542968750000043*x**5 + -29.900390624999932*x**6
        psi_4_x = lambda x : -0.15820312499999734 + 1.1601562499999956*x + 5.220703124999973*x**2 + -4.007812499999981*x**3 + -15.02929687499995*x**4 + 2.847656249999986*x**5 + 9.966796874999975*x**6

        dNv = np.array([phi_1_x, psi_1_x, phi_2_x, psi_2_x, phi_3_x, psi_3_x, phi_4_x, psi_4_x]).reshape(-1,1)

        return dNv    

    def _EulerBernoulli_ddN(self) -> np.ndarray:
        
        phi_1_xx = lambda x : -0.9492187500000036 + 3.292968749999952*x + 28.476562500000043*x**2 + -55.19531249999983*x**3 + -42.71484375000006*x**4 + 73.08984374999987*x**5
        psi_1_xx = lambda x : -0.07421875000000033 + 0.222656249999995*x + 2.3203125000000036*x**2 + -3.867187499999983*x**3 + -4.746093750000004*x**4 + 6.6445312499999885*x**5
        phi_2_xx = lambda x : -0.9492187499999947 + -3.2929687499999467*x + 28.476562499999943*x**2 + 55.195312499999794*x**3 + -42.71484374999993*x**4 + -73.08984374999984*x**5
        psi_2_xx = lambda x : 0.07421874999999967 + 0.22265624999999567*x + -2.320312499999996*x**2 + -3.867187499999982*x**3 + 4.746093749999995*x**4 + 6.644531249999985*x**5
        phi_3_xx = lambda x : 0.9492187499999858 + 54.10546874999983*x + -28.476562499999815*x**2 + -218.3203124999994*x**3 + 42.714843749999766*x**4 + 179.40234374999955*x**5
        psi_3_xx = lambda x : -1.1601562500000036 + 10.441406249999947*x + 12.023437500000048*x**2 + -60.117187499999815*x**3 + -14.238281250000057*x**4 + 59.80078124999986*x**5
        phi_4_xx = lambda x : 0.9492187500000178 + -54.10546874999983*x + -28.47656250000018*x**2 + 218.32031249999943*x**3 + 42.71484375000021*x**4 + -179.4023437499996*x**5
        psi_4_xx = lambda x : 1.1601562499999956 + 10.441406249999947*x + -12.023437499999943*x**2 + -60.1171874999998*x**3 + 14.23828124999993*x**4 + 59.80078124999985*x**5

        ddNv = np.array([phi_1_xx, psi_1_xx, phi_2_xx, psi_2_xx, phi_3_xx, psi_3_xx, phi_4_xx, psi_4_xx]).reshape(-1,1)

        return ddNv