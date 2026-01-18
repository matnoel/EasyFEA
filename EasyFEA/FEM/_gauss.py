# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Gauss class used to determine the coordinates and weights of integration points."""

import numpy as np

# utils
from ._utils import ElemType, MatrixType
from ..Utilities import _types


class Gauss:
    """Gauss quadrature."""

    def __init__(self, elemType: ElemType, matrixType: MatrixType):
        """Creates integration points.

        Parameters
        ----------
        elemType : ElemType
            element type.
        matrixType : MatrixType
            matrix type (e.g [MatrixType.rigi, MatrixType.mass, MatrixType.beam])
        """

        if isinstance(matrixType, (MatrixType, str)):
            coord, weights = Gauss.Gauss_factory(elemType, matrixType)
        elif isinstance(matrixType, int):
            coord, weights = Gauss._Gauss_factory_nPg(elemType, matrixType)
        else:
            raise NotImplementedError

        self.__coord = coord
        self.__weights = weights

    @property
    def coord(self) -> _types.FloatArray:
        """integration point coordinates"""
        return self.__coord

    @property
    def weights(self) -> _types.FloatArray:
        """integration point weights"""
        return self.__weights

    @property
    def nPg(self) -> int:
        """number of integration points"""
        return self.__weights.size

    @staticmethod
    def _Triangle(nPg: int) -> tuple[_types.Numbers, _types.Numbers, _types.Numbers]:
        """available [1, 3, 6, 7, 12]\n
        order = [1, 2, 3, 4, 5]"""
        if nPg == 1:
            ksis = [1 / 3]
            etas = [1 / 3]

            weights = [1 / 2]

        elif nPg == 3:
            ksis = [1 / 6, 2 / 3, 1 / 6]
            etas = [1 / 6, 1 / 6, 2 / 3]

            weights = [1 / 6] * 3

        elif nPg == 6:
            a = 0.445948490915965
            b = 0.091576213509771
            p1 = 0.11169079483905
            p2 = 0.0549758718227661

            ksis = [b, 1 - 2 * b, b, a, a, 1 - 2 * a]
            etas = [b, b, 1 - 2 * b, 1 - 2 * a, a, a]

            weights = [p2, p2, p2, p1, p1, p1]

        elif nPg == 7:
            a = 0.470142064105115
            b = 0.101286507323456
            p1 = 0.066197076394253
            p2 = 0.062969590272413

            ksis = [1 / 3, a, 1 - 2 * a, a, b, 1 - 2 * b, b]
            etas = [1 / 3, a, a, 1 - 2 * a, b, b, 1 - 2 * b]

            weights = [9 / 80, p1, p1, p1, p2, p2, p2]

        elif nPg == 12:
            a = 0.063089014491502
            b = 0.249286745170910
            c = 0.310352451033785
            d = 0.053145049844816
            p1 = 0.025422453185103
            p2 = 0.058393137863189
            p3 = 0.041425537809187

            ksis = [a, 1 - 2 * a, a, b, 1 - 2 * b, b, c, d, 1 - c - d, 1 - c - d, c, d]
            etas = [a, a, 1 - 2 * a, b, b, 1 - 2 * b, d, c, c, d, 1 - c - d, 1 - c - d]

            weights = [p1, p1, p1, p2, p2, p2, p3, p3, p3, p3, p3, p3]
        else:
            raise NotImplementedError("unknown nPg")

        return ksis, etas, weights

    @staticmethod
    def _Quadrangle(nPg: int) -> tuple[_types.Numbers, _types.Numbers, _types.Numbers]:
        """available [4, 9]\n
        order = [1, 2]"""
        if nPg == 4:
            a = 1 / np.sqrt(3)
            ksis = [-a, a, a, -a]
            etas = [-a, -a, a, a]

            weights = [1.0] * nPg
        elif nPg == 9:
            a = 0.774596669241483

            ksis = [-a, a, a, -a, 0.0, a, 0.0, -a, 0.0]
            etas = [-a, -a, a, a, -a, 0.0, a, 0.0, 0.0]
            weights = [
                25 / 81,
                25 / 81,
                25 / 81,
                25 / 81,
                40 / 81,
                40 / 81,
                40 / 81,
                40 / 81,
                64 / 81,
            ]
        else:
            raise NotImplementedError("unknown nPg")

        return ksis, etas, weights

    @staticmethod
    def _Tetrahedron(
        nPg: int,
    ) -> tuple[_types.Numbers, _types.Numbers, _types.Numbers, _types.Numbers]:
        """available [1, 4, 5, 15]\n
        order = [1, 2, 3, 5]"""

        if nPg == 1:
            x = [1 / 4]
            y = [1 / 4]
            z = [1 / 4]

            weights = [1 / 6]

        elif nPg == 4:
            a: float = (5 - np.sqrt(5)) / 20
            b: float = (5 + 3 * np.sqrt(5)) / 20

            x = [a, a, a, b]
            y = [a, a, b, a]
            z = [a, b, a, a]

            weights = [1 / 24] * nPg

        elif nPg == 5:
            a = 1 / 4
            b = 1 / 6
            c = 1 / 2

            x = [a, b, b, b, c]
            y = [a, b, b, c, b]
            z = [a, b, c, b, b]

            weights = [-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40]

        elif nPg == 15:
            a = 1 / 4
            b1: float = (7 + np.sqrt(15)) / 34
            b2: float = (7 - np.sqrt(15)) / 34
            c1: float = (13 - 3 * np.sqrt(15)) / 34
            c2: float = (13 + 3 * np.sqrt(15)) / 34
            d: float = (5 - np.sqrt(15)) / 20
            e: float = (5 + np.sqrt(15)) / 20

            x = [a, b1, b1, b1, c1, b2, b2, b2, c2, d, d, e, d, e, e]
            y = [a, b1, b1, c1, b1, b2, b2, c2, b2, d, e, d, e, d, e]
            z = [a, b1, c1, b1, b1, b2, c2, b2, b2, e, d, d, e, e, d]

            p1: float = 8 / 405
            p2: float = (2665 - 14 * np.sqrt(15)) / 226800
            p3: float = (2665 + 14 * np.sqrt(15)) / 226800
            p4: float = 5 / 567

            weights = [p1, p2, p2, p2, p2, p3, p3, p3, p3, p4, p4, p4, p4, p4, p4]
        else:
            raise NotImplementedError("unknown nPg")

        return x, y, z, weights

    @staticmethod
    def _Hexahedron(
        nPg: int,
    ) -> tuple[_types.Numbers, _types.Numbers, _types.Numbers, _types.Numbers]:
        """available [8, 27]\n
        order = [3, 5]"""

        if nPg == 8:
            a: float = 1 / np.sqrt(3)

            x = [-a, -a, -a, -a, a, a, a, a]
            y = [-a, -a, a, a, -a, -a, a, a]
            z = [-a, a, -a, a, -a, a, -a, a]

            weights = [1.0] * nPg

        elif nPg == 27:
            a = np.sqrt(3 / 5)
            c1: float = 5 / 9
            c2: float = 8 / 9

            x = [-a] * 9
            x.extend([0] * 9)
            x.extend([a] * 9)
            y = [-a, -a, -a, 0.0, 0.0, 0.0, a, a, a] * 3
            z = [-a, 0.0, a] * 9

            c13 = c1**3
            c23 = c2**3

            c12 = c1**2 * c2
            c22 = c1 * c2**2

            weights = [
                c13,
                c12,
                c13,
                c12,
                c22,
                c12,
                c13,
                c12,
                c13,
                c12,
                c22,
                c12,
                c22,
                c23,
                c22,
                c12,
                c22,
                c12,
                c13,
                c12,
                c13,
                c12,
                c22,
                c12,
                c13,
                c12,
                c13,
            ]
        else:
            raise NotImplementedError("unknown nPg")

        return x, y, z, weights

    @staticmethod
    def _Prism(
        nPg: int,
    ) -> tuple[_types.Numbers, _types.Numbers, _types.Numbers, _types.Numbers]:
        """available [6, 8, 21]\n
        order X = [3, 3, 5]\n
        order Y & Z = [2, 3, 5]"""

        if nPg == 6:
            a: float = 1 / np.sqrt(3)

            xc = [-a, -a, -a, a, a, a]
            yc = [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
            zc = [0.5, 0.5, 0.0, 0.5, 0.5, 0.0]

            weights = [1 / 6] * nPg

        elif nPg == 8:
            a = 0.577350269189626

            xc = [-a, -a, -a, -a, a, a, a, a]
            yc = [1 / 3, 0.6, 0.2, 0.2] * 2
            zc = [1 / 3, 0.2, 0.6, 0.2] * 2

            weights = [-27 / 96, 25 / 96, 25 / 96, 25 / 96] * 2

        elif nPg == 21:
            al: float = np.sqrt(3 / 5)
            c1: float = 5 / 9
            c2: float = 8 / 9
            a = (6 + np.sqrt(15)) / 21
            b: float = (6 - np.sqrt(15)) / 21
            cp: float = (155 + np.sqrt(15)) / 2400
            cm: float = (155 - np.sqrt(15)) / 2400

            xc = [
                -al,
                -al,
                -al,
                -al,
                -al,
                -al,
                -al,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                al,
                al,
                al,
                al,
                al,
                al,
                al,
            ]
            yc = [1 / 3, a, 1 - 2 * a, a, b, 1 - 2 * b, b] * 3
            zc = [1 / 3, a, a, 1 - 2 * a, b, b, 1 - 2 * b] * 3

            weights = [
                c1 * 9 / 80,
                c1 * cp,
                c1 * cp,
                c1 * cp,
                c1 * cm,
                c1 * cm,
                c1 * cm,
                c2 * 9 / 80,
                c2 * cp,
                c2 * cp,
                c2 * cp,
                c2 * cm,
                c2 * cm,
                c2 * cm,
                c1 * 9 / 80,
                c1 * cp,
                c1 * cp,
                c1 * cp,
                c1 * cm,
                c1 * cm,
                c1 * cm,
            ]
        else:
            raise NotImplementedError("unknown nPg")

        # xc, yc, zc -> base code aster
        # z, x, y -> gmsh
        # yc -> x, zc -> y, xc -> z
        x = np.array(yc)
        y = np.array(zc)
        z = np.array(xc)

        return x, y, z, weights

    @staticmethod
    def Gauss_factory(
        elemType: ElemType, matrixType: MatrixType
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the integration points and weights based on the element and matrix type."""

        assert matrixType in MatrixType.Get_types()

        # TODO create a function to calculate the order directly?

        if elemType == ElemType.SEG2:
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType in [MatrixType.mass, MatrixType.beam]:
                nPg = 2
            else:
                raise ValueError("unknown matrixType")
            x, weights = np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.SEG3:
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType == MatrixType.mass:
                nPg = 3
            elif matrixType == MatrixType.beam:
                nPg = 4
            else:
                raise ValueError("unknown matrixType")
            x, weights = np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.SEG4:
            if matrixType == MatrixType.rigi:
                nPg = 2
            elif matrixType == MatrixType.mass:
                nPg = 4
            elif matrixType == MatrixType.beam:
                nPg = 6
            else:
                raise ValueError("unknown matrixType")
            x, weights = np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.SEG5:
            if matrixType == MatrixType.rigi:
                nPg = 4
            elif matrixType == MatrixType.mass:
                nPg = 5
            elif matrixType == MatrixType.beam:
                nPg = 8
            else:
                raise ValueError("unknown matrixType")
            x, weights = np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.TRI3:
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType == MatrixType.mass:
                nPg = 3
            else:
                raise ValueError("unknown matrixType")
            xis, etas, weights = Gauss._Triangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.TRI6:
            if matrixType == MatrixType.rigi:
                nPg = 3
            elif matrixType == MatrixType.mass:
                nPg = 6
            else:
                raise ValueError("unknown matrixType")
            xis, etas, weights = Gauss._Triangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.TRI10:
            nPg = 6
            xis, etas, weights = Gauss._Triangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.TRI15:
            nPg = 12
            xis, etas, weights = Gauss._Triangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.QUAD4:
            nPg = 4
            xis, etas, weights = Gauss._Quadrangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.QUAD8:
            if matrixType == MatrixType.rigi:
                nPg = 4
            elif matrixType == MatrixType.mass:
                nPg = 9
            else:
                raise ValueError("unknown matrixType")
            xis, etas, weights = Gauss._Quadrangle(nPg)  # type: ignore [assignment]
        elif elemType == ElemType.QUAD9:
            nPg = 9
            xis, etas, weights = Gauss._Quadrangle(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.TETRA4:
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType == MatrixType.mass:
                nPg = 4
            else:
                raise ValueError("unknown matrixType")
            x, y, z, weights = Gauss._Tetrahedron(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.TETRA10:
            nPg = 4
            x, y, z, weights = Gauss._Tetrahedron(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.HEXA8:
            nPg = 8
            x, y, z, weights = Gauss._Hexahedron(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.HEXA20:
            nPg = 27
            x, y, z, weights = Gauss._Hexahedron(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.HEXA27:
            nPg = 27
            x, y, z, weights = Gauss._Hexahedron(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.PRISM6:
            nPg = 6
            x, y, z, weights = Gauss._Prism(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.PRISM15:
            nPg = 6
            x, y, z, weights = Gauss._Prism(nPg)  # type: ignore [assignment]

        elif elemType == ElemType.PRISM18:
            nPg = 21
            x, y, z, weights = Gauss._Prism(nPg)  # type: ignore [assignment]

        else:
            raise NotImplementedError(f"Element {elemType} not implemented.")

        if elemType in ElemType.Get_1D():
            coord = np.asarray([x]).T.reshape((nPg, 1))  # type: ignore
        elif elemType in ElemType.Get_2D():
            coord = np.asarray([xis, etas]).T.reshape((nPg, 2))  # type: ignore
        elif elemType in ElemType.Get_3D():
            coord = np.asarray([x, y, z]).T.reshape((nPg, 3))  # type: ignore
        else:
            raise ValueError("Unknown element type")

        weights = np.asarray(weights).reshape(nPg)

        return coord, weights

    @staticmethod
    def _Gauss_factory_nPg(
        elemType: ElemType, nPg
    ) -> tuple[_types.FloatArray, _types.FloatArray]:
        """Returns the integration points and weights based on the element type and the number of Gauss points (nPg)."""

        if elemType.startswith(ElemType.SEG2.topology):
            x, weights = np.polynomial.legendre.leggauss(nPg)

        elif elemType.startswith(ElemType.TRI3.topology):
            xis, etas, weights = Gauss._Triangle(nPg)  # type: ignore [assignment]

        elif elemType.startswith(ElemType.QUAD4.topology):
            xis, etas, weights = Gauss._Quadrangle(nPg)  # type: ignore [assignment]

        elif elemType.startswith(ElemType.TETRA4.topology):
            x, y, z, weights = Gauss._Tetrahedron(nPg)  # type: ignore [assignment]

        elif elemType.startswith(ElemType.HEXA8.topology):
            x, y, z, weights = Gauss._Hexahedron(nPg)  # type: ignore [assignment]

        elif elemType.startswith(ElemType.PRISM6.topology):
            x, y, z, weights = Gauss._Prism(nPg)  # type: ignore [assignment]

        else:
            raise NotImplementedError(f"Element {elemType} not implemented.")

        if elemType in ElemType.Get_1D():
            coord = np.asarray([x]).T.reshape((nPg, 1))  # type: ignore
        elif elemType in ElemType.Get_2D():
            coord = np.asarray([xis, etas]).T.reshape((nPg, 2))  # type: ignore
        elif elemType in ElemType.Get_3D():
            coord = np.asarray([x, y, z]).T.reshape((nPg, 3))  # type: ignore
        else:
            raise ValueError("Unknown element type")

        weights = np.asarray(weights).reshape(nPg)

        return coord, weights
