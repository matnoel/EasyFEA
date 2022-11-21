import numpy as np

class Gauss:
    """Classe des points d'intégration"""

    def __init__(self, elemType: str, matriceType: str):
        """Construction de points d'intégration

        Parameters
        ----------
        elemType : str
            type d'element
        matriceType : str
            [MatriceType.rigi, MatriceType.masse,MatriceType.beam]
        """

        coord, poids = Gauss.__calc_gauss(elemType, matriceType)

        self.__coord = coord
        self.__poids = poids

    @property
    def coord(self) -> np.ndarray:
        """coordonnées des points d'intégration"""
        return self.__coord
    
    @property
    def poids(self) -> np.ndarray:
        """poids des points d'intégration"""
        return self.__poids

    @property
    def nPg(self) -> int:
        """nombres de points d'intégration"""
        return self.__poids.size

    @staticmethod
    def __CoordoPoidsGaussTriangle(nPg: int):
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
    def __CoordoPoidsGaussQuad(nPg: int):
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
    def __calc_gauss(elemType: str, matriceType: str):
        """Calcul des points d'intégrations en fonction du type d'element et de matrice

        Parameters
        ----------
        elemType : ElemType
            type d'element
        matriceType : MatriceType
            type de matrice

        Returns
        -------
        np.ndarray, np.ndarray
            coord, poids
        """

        from GroupElem import GroupElem, ElemType, MatriceType

        assert matriceType in GroupElem.get_MatriceType()

        # TODO faire une fonction pour calculer directement lordre ?

        if elemType == ElemType.SEG2:
            dim = 1
            if matriceType == MatriceType.rigi:
                nPg = 1
            elif matriceType in [MatriceType.masse,MatriceType.beam]:
                nPg = 2
            x, poids =  np.polynomial.legendre.leggauss(nPg)
        elif elemType == ElemType.SEG3:
            dim = 1
            if matriceType == MatriceType.rigi:
                nPg = 1
            elif matriceType in [MatriceType.masse]:
                nPg = 3
            elif matriceType == MatriceType.beam:
                nPg = 4
            x, poids =  np.polynomial.legendre.leggauss(nPg)
        elif elemType == ElemType.SEG4:
            dim = 1
            if matriceType == MatriceType.rigi:
                nPg = 2
            elif matriceType == MatriceType.masse:
                nPg = 4
            elif matriceType == MatriceType.beam:
                nPg = 6
            x, poids =  np.polynomial.legendre.leggauss(nPg)
        elif elemType == ElemType.SEG5:
            dim = 1
            if matriceType == MatriceType.rigi:
                nPg = 4
            elif matriceType  == MatriceType.masse:
                nPg = 5
            elif matriceType == MatriceType.beam:
                nPg = 8
            x, poids =  np.polynomial.legendre.leggauss(nPg)
        elif elemType == ElemType.TRI3:
            dim = 2            
            if matriceType == MatriceType.rigi:
                nPg = 1
            elif matriceType == MatriceType.masse:
                nPg = 3
            ksis, etas, poids = Gauss.__CoordoPoidsGaussTriangle(nPg)
        elif elemType == ElemType.TRI6:
            dim = 2            
            if matriceType == MatriceType.rigi:
                nPg = 3
            elif matriceType == MatriceType.masse:
                nPg = 6
            ksis, etas, poids = Gauss.__CoordoPoidsGaussTriangle(nPg)
        elif elemType == ElemType.TRI10:
            dim = 2            
            if matriceType == MatriceType.rigi:
                nPg = 6
            else:
                nPg = 7
            ksis, etas, poids = Gauss.__CoordoPoidsGaussTriangle(nPg)
        elif elemType == ElemType.TRI15:
            dim = 2            
            if matriceType == MatriceType.rigi:
                nPg = 7
            else:
                nPg = 12
                # nPg = 7
            ksis, etas, poids = Gauss.__CoordoPoidsGaussTriangle(nPg)
        elif elemType == ElemType.QUAD4:
            dim = 2            
            if matriceType in [MatriceType.rigi,MatriceType.masse]:
                nPg = 4
            ksis, etas, poids = Gauss.__CoordoPoidsGaussQuad(nPg)
            
        elif elemType == ElemType.QUAD8:
            dim = 2            
            if matriceType in [MatriceType.rigi,MatriceType.masse]:
                nPg = 9
            ksis, etas, poids = Gauss.__CoordoPoidsGaussQuad(nPg)
                    
        elif elemType == ElemType.TETRA4:
            dim = 3            
            if matriceType == MatriceType.rigi:
                nPg = 1

                x = 1/4
                y = 1/4
                z = 1/4

                poids = 1/6
            elif matriceType == MatriceType.masse:
                nPg = 4

                a = (5-np.sqrt(5))/20
                b = (5+3*np.sqrt(5))/20

                x = [a, a, a, b]
                y = [a, a, b, a]
                z = [a, b, a, a]

                poids = [1/24]*nPg
        
        elif elemType == ElemType.HEXA8:
            dim = 3            
            if matriceType in [MatriceType.rigi, MatriceType.masse]:
                nPg = 8

                m1r3 = -1/np.sqrt(3)
                p1r3 = 1/np.sqrt(3)

                x = [m1r3, m1r3, m1r3, m1r3, p1r3, p1r3, p1r3, p1r3]
                y = [m1r3, m1r3, p1r3, p1r3, m1r3, m1r3, p1r3, p1r3]
                z = [m1r3, p1r3, m1r3, p1r3, m1r3, p1r3, m1r3, p1r3]

                poids = [1]*nPg

        elif elemType == ElemType.PRISM6:
            dim = 3            
            if matriceType in [MatriceType.rigi, MatriceType.masse]:
                nPg = 6

                m1r3 = -1/np.sqrt(3)
                p1r3 = 1/np.sqrt(3)

                ordre = [2,0,1,5,3,4]

                x = [m1r3, m1r3, m1r3, p1r3, p1r3, p1r3]
                y = [0.5, 0, 0.5, 0.5, 0, 0.5]
                z = [0.5, 0.5, 0, 0.5, 0.5, 0]

                x = np.array(x)[ordre]
                y = np.array(y)[ordre]
                z = np.array(z)[ordre]

                poids = [1/6]*nPg

        else:
            raise "Element inconnue"

        if dim == 1:
            coord = np.array([x]).T.reshape((nPg,1))
        elif dim == 2:
            coord = np.array([ksis, etas]).T.reshape((nPg,2))
        elif dim == 3:
            coord = np.array([x, y, z]).T.reshape((nPg,3))

        poids = np.array(poids).reshape(nPg)

        return coord, poids