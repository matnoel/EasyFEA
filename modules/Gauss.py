"""Module used to create integration / Gauss points."""

import numpy as np

class Gauss:

    def __init__(self, elemType: str, matrixType: str):
        """Integration point construction

        Parameters
        ----------
        elemType : str
            element type
        matrixType : str
            [MatrixType.rigi, MatrixType.mass,MatrixType.beam]
        """

        coord, weights = Gauss.__calc_gauss(elemType, matrixType)

        self.__coord = coord
        self.__weights = weights

    @property
    def coord(self) -> np.ndarray:
        """integration point coordinates"""
        return self.__coord
    
    @property
    def weights(self) -> np.ndarray:
        """integration point weights"""
        return self.__weights

    @property
    def nPg(self) -> int:
        """number of integration points"""
        return self.__weights.size

    @staticmethod
    def __CoordoPoidsGaussTriangle(nPg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """available [1, 3, 6, 7, 12]"""
        if nPg == 1:
            ksis = 1/3
            etas = 1/3

            poids = 1/2

        elif nPg == 3:
            ksis = [1/6, 2/3, 1/6]
            etas = [1/6, 1/6, 2/3]

            poids = [1/6] * 3

        elif nPg == 6:
            a = 0.445948490915965
            b = 0.091576213509771
            p1 = 0.11169079483905
            p2 = 0.0549758718227661
            
            ksis = [b, 1-2*b, b, a, a, 1-2*a]
            etas = [b, b, 1-2*b, 1-2*a, a, a]

            poids = [p2, p2, p2, p1, p1, p1]

        elif nPg == 7:
            a = 0.470142064105115
            b = 0.101286507323456
            p1 = 0.066197076394253
            p2 = 0.062969590272413

            ksis = [1/3, a, 1-2*a, a, b, 1-2*b, b]
            etas = [1/3, a, a, 1-2*a, b, b, 1-2*b]

            poids = [9/80, p1, p1, p1, p2, p2, p2]

        elif nPg == 12:
            a = 0.063089014491502
            b = 0.249286745170910
            c = 0.310352451033785
            d = 0.053145049844816
            p1 = 0.025422453185103
            p2 = 0.058393137863189
            p3 = 0.041425537809187

            ksis = [a, 1-2*a, a, b, 1-2*b, b, c, d, 1-c-d, 1-c-d, c, d]
            etas = [a, a, 1-2*a, b, b, 1-2*b, d, c, c, d, 1-c-d, 1-c-d]

            poids = [p1, p1, p1, p2, p2, p2, p3, p3, p3, p3, p3, p3]

        return ksis, etas, poids

    @staticmethod
    def __CoordoPoidsGaussQuad(nPg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """available [4, 9]"""
        if nPg == 4:
            a = 1/np.sqrt(3)
            ksis = [-a, a, a, -a]
            etas = [-a, -a, a, a]

            poids = [1]*nPg
        elif nPg == 9:
            a = 0.774596669241483

            ksis = [-a, a, a, -a, 0, a, 0, -a, 0]
            etas = [-a, -a, a, a, -a, 0, a, 0, 0]
            poids = [25/81, 25/81, 25/81, 25/81, 40/81, 40/81, 40/81, 40/81, 64/81]

        return ksis, etas, poids

    @staticmethod
    def __CoordoPoidsGaussTetra(nPg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """available [1, 4, 5, 15]"""
        if nPg == 1:            

            x = 1/4
            y = 1/4
            z = 1/4

            poids = 1/6

        elif nPg == 4:

            a = (5-np.sqrt(5))/20
            b = (5+3*np.sqrt(5))/20

            x = [a, a, a, b]
            y = [a, a, b, a]
            z = [a, b, a, a]

            poids = [1/24]*nPg

        elif nPg == 5:

            a = 1/4
            b = 1/6
            c = 1/2

            x = [a, b, b, b, c]
            y = [a, b, b, c, b]
            z = [a, b, c, b, b]

            poids = [-2/15, 3/40, 3/40, 3/40, 3/40]

        elif nPg == 15:

            a = 1/4
            b1 = (7+np.sqrt(15))/34; b2 = (7-np.sqrt(15))/34
            c1 = (13-3*np.sqrt(15))/34; c2 = (13+3*np.sqrt(15))/34
            d = (5-np.sqrt(15))/20
            e = (5+np.sqrt(15))/20

            x = [a, b1, b1, b1 , c1, b2, b2, b2, c2, d, d, e, d, e, e]
            y = [a, b1, b1, c1, b1, b2, b2, c2, b2, d, e, d, e, d, e]
            z = [a, b1, c1, b1, b1, b2, c2, b2, b2, e, d, d, e, e, d]

            p1 = 8/405
            p2 = (2665 - 14*np.sqrt(15))/226800
            p3 = (2665 + 14*np.sqrt(15))/226800
            p4 = 5/567

            poids = [p1, p2, p2, p2, p2, p3, p3, p3, p3, p4, p4, p4, p4, p4, p4]

        return x, y, z, poids
    
    @staticmethod
    def __CoordoPoidsGaussHexa(nPg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """available [8]"""
        if nPg == 8:            

            m1r3 = -1/np.sqrt(3)
            p1r3 = 1/np.sqrt(3)

            x = [m1r3, m1r3, m1r3, m1r3, p1r3, p1r3, p1r3, p1r3]
            y = [m1r3, m1r3, p1r3, p1r3, m1r3, m1r3, p1r3, p1r3]
            z = [m1r3, p1r3, m1r3, p1r3, m1r3, p1r3, m1r3, p1r3]

            poids = [1]*nPg

        if nPg == 27:

            a = np.sqrt(3/5)
            c1 = 5/9
            c2 = 8/9

            x = [-a]*9; x.extend([0]*9); x.extend([a]*9)
            y = [-a,-a,-a,0,0,0,a,a,a]*3
            z = [-a,0,a]*9

            c13 = c1**3
            c23 = c2**3

            c12 = c1**2*c2
            c22 = c1*c2**2

            poids = [c13,c12,c13, c12,c22,c12, c13,c12,c13, c12,c22,c12, c22,c23,c22, c12,c22,c12, c13,c12,c13, c12,c22,c12, c13,c12,c13]

        return x, y, z, poids
    
    @staticmethod
    def __CoordoPoidsGaussPrism(nPg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """available [6, 8]"""
        if nPg == 6:
            
            m1r3 = -1/np.sqrt(3)
            p1r3 = 1/np.sqrt(3)

            # X, Y, Z -> base code aster
            X = [m1r3, m1r3, m1r3, p1r3, p1r3, p1r3]
            Y = [0.5, 0, 0.5, 0.5, 0, 0.5]
            Z = [0.5, 0.5, 0, 0.5, 0.5, 0]

            # x <- y, y <- z, z <- x  
            x = np.array(Y)
            y = np.array(Z)
            z = np.array(X)

            poids = [1/6]*nPg

        elif nPg == 8:
            
            a=0.577350269189626

            # X, Y, Z -> base code aster
            X = [-a, -a, -a, -a, a, a, a, a]
            Y = [1/3, 0.6, 0.2, 0.2]*2
            Z = [1/3, 0.2, 0.6, 0.2]*2

            # x <- y, y <- z, z <- x  
            x = np.array(Y)
            y = np.array(Z)
            z = np.array(X)

            poids = [-27/96, 25/96, 25/96, 25/96]*2

        return x, y, z, poids

    @staticmethod
    def __calc_gauss(elemType: str, matrixType: str):
        """Calculation of integration points according to element and matrix type
        """

        from GroupElem import GroupElem, ElemType, MatrixType

        assert matrixType in GroupElem.get_MatrixType()

        # TODO faire une fonction pour calculer directement lordre ?

        if elemType == ElemType.SEG2:
            dim = 1
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType in [MatrixType.mass, MatrixType.beam]:
                nPg = 2
            x, weights =  np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.SEG3:
            dim = 1
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType in [MatrixType.mass]:
                nPg = 3
            elif matrixType == MatrixType.beam:
                nPg = 4
            x, weights =  np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.SEG4:
            dim = 1
            if matrixType == MatrixType.rigi:
                nPg = 2
            elif matrixType == MatrixType.mass:
                nPg = 4
            elif matrixType == MatrixType.beam:
                nPg = 6
            x, weights =  np.polynomial.legendre.leggauss(nPg)
            
        elif elemType == ElemType.SEG5:
            dim = 1
            if matrixType == MatrixType.rigi:
                nPg = 4
            elif matrixType  == MatrixType.mass:
                nPg = 5
            elif matrixType == MatrixType.beam:
                nPg = 8
            x, weights =  np.polynomial.legendre.leggauss(nPg)

        elif elemType == ElemType.TRI3:
            dim = 2            
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType == MatrixType.mass:
                nPg = 3
            ksis, etas, weights = Gauss.__CoordoPoidsGaussTriangle(nPg)

        elif elemType == ElemType.TRI6:
            dim = 2            
            if matrixType == MatrixType.rigi:
                nPg = 3
            elif matrixType == MatrixType.mass:
                nPg = 6
            ksis, etas, weights = Gauss.__CoordoPoidsGaussTriangle(nPg)

        elif elemType == ElemType.TRI10:
            dim = 2            
            nPg = 6
            ksis, etas, weights = Gauss.__CoordoPoidsGaussTriangle(nPg)

        # elif elemType == ElemType.TRI15:
        #     dim = 2            
        #     if matrixType == MatrixType.rigi:
        #         nPg = 6
        #     elif matrixType == MatrixType.masse:
        #         nPg = 12
        #     ksis, etas, weights = Gauss.__CoordoPoidsGaussTriangle(nPg)

        elif elemType == ElemType.QUAD4:
            dim = 2            
            nPg = 4
            ksis, etas, weights = Gauss.__CoordoPoidsGaussQuad(nPg)
            
        elif elemType == ElemType.QUAD8:
            dim = 2            
            if matrixType == MatrixType.rigi:
                nPg = 4
            elif matrixType == MatrixType.mass:
                nPg = 9
            ksis, etas, weights = Gauss.__CoordoPoidsGaussQuad(nPg)
                    
        elif elemType == ElemType.TETRA4:
            dim = 3            
            if matrixType == MatrixType.rigi:
                nPg = 1
            elif matrixType == MatrixType.mass:
                nPg = 4            
            x, y, z, weights = Gauss.__CoordoPoidsGaussTetra(nPg)

        elif elemType == ElemType.TETRA10:
            dim = 3            
            nPg = 4
            x, y, z, weights = Gauss.__CoordoPoidsGaussTetra(nPg)

        elif elemType == ElemType.HEXA8:
            dim = 3            
            if matrixType in [MatrixType.rigi, MatrixType.mass]:
                nPg = 8
                x, y, z, weights = Gauss.__CoordoPoidsGaussHexa(nPg)

        elif elemType == ElemType.HEXA20:
            dim = 3
            if matrixType in [MatrixType.rigi, MatrixType.mass]:
                nPg = 8
                x, y, z, weights = Gauss.__CoordoPoidsGaussHexa(nPg)

        elif elemType == ElemType.PRISM6:
            dim = 3            
            if matrixType in [MatrixType.rigi, MatrixType.mass]:
                nPg = 6

                x, y, z, weights = Gauss.__CoordoPoidsGaussPrism(nPg)

        elif elemType == ElemType.PRISM15:
            dim = 3            
            if matrixType in [MatrixType.rigi, MatrixType.mass]:
                nPg = 6

                x, y, z, weights = Gauss.__CoordoPoidsGaussPrism(nPg)

        else:
            raise Exception("Element not implemented")

        if dim == 1:
            coord = np.array([x]).T.reshape((nPg,1))
        elif dim == 2:
            coord = np.array([ksis, etas]).T.reshape((nPg,2))
        elif dim == 3:
            coord = np.array([x, y, z]).T.reshape((nPg,3))

        weights = np.array(weights).reshape(nPg)

        return coord, weights