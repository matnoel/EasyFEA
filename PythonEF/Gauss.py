from typing import cast
import numpy as np

class Gauss:

    def __init__(self, elemType: str, matriceType: str):
        
        coord, poids = Gauss.__calc_gauss(elemType, matriceType)

        self.__coord = coord
        self.__poids = poids

    @property
    def coord(self) -> np.ndarray:
        return self.__coord
    
    @property
    def poids(self) -> np.ndarray:
        return self.__poids

    @property
    def nPg(self) -> int:
        return self.__poids.size

    @staticmethod
    def __calc_gauss(elemType: str, matriceType: str):

        from GroupElem import GroupElem

        assert matriceType in GroupElem.get_MatriceType()

        if elemType == "SEG2":
            dim = 1
            if matriceType == "rigi":
                nPg = 1
                x = 0
                poids = 2
            elif matriceType == "masse":
                nPg = 2
                x = [-np.sqrt(1/3), np.sqrt(1/3)]
                poids = [1]*2

        elif elemType == "SEG3":
            dim = 1
            if matriceType == "rigi":
                nPg = 2
                x = [-np.sqrt(1/3), np.sqrt(1/3)]
                poids = [1]*2
            elif matriceType == "masse":
                nPg = 3
                x = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
                poids = [5/9, 8/9, 5/9]

        elif elemType == "TRI3":
            dim = 2            
            if matriceType == "rigi":
                nPg = 1

                ksis = 1/3
                etas = 1/3

                poids = 1/2
            elif matriceType == "masse":
                nPg = 3

                ksis = [1/6, 2/3, 1/6]
                etas = [1/6, 1/6, 2/3]

                poids = [1/6] * 3
            
        elif elemType == "TRI6":
            dim = 2            
            if matriceType == "rigi":
                nPg = 3

                ksis = [1/6, 2/3, 1/6]
                etas = [1/6, 1/6, 2/3]

                poids = [1/6] * 3
            elif matriceType == "masse":
                nPg = 6

                a = 0.445948490915965
                b = 0.091576213509771
                p1 = 0.111690794839005
                p2 = 0.054975871827661
                ksis = [b, 1-2*b, b, a, a, 1-2*a]
                etas = [b, b, 1-2*b, 1-2*a, a, a]

                poids = [p2, p2, p2, p1, p1, p1]
            
        elif elemType == "QUAD4":
            dim = 2            
            if matriceType in ["rigi","masse"]:
                nPg = 4

                a = 1/np.sqrt(3)
                ksis = [-a, a, a, -a]
                etas = [-a, -a, a, a]

                poids = [1]*nPg
            
        elif elemType == "QUAD8":
            dim = 2            
            if matriceType in ["rigi","masse"]:
                nPg = 9

                a = 0.774596669241483

                ksis = [-a, a, a, -a, 0, a, 0, -a, 0]
                etas = [-a, -a, a, a, -a, 0, a, 0, 0]
                poids = [25/81, 25/81, 25/81, 25/81, 40/81, 40/81, 40/81, 40/81, 64/81]
                    
        elif elemType == "TETRA4":
            dim = 3            
            if matriceType == "rigi":
                nPg = 1

                x = 1/4
                y = 1/4
                z = 1/4

                poids = 1/6
            elif matriceType == "masse":
                nPg = 4

                a = (5-np.sqrt(5))/20
                b = (5+3*np.sqrt(5))/20

                x = [a, a, a, b]
                y = [a, a, b, a]
                z = [a, b, a, a]

                poids = [1/24]*nPg
        
        elif elemType == "HEXA8":
            dim = 3            
            if matriceType in ["rigi", "masse"]:
                nPg = 8

                m1r3 = -1/np.sqrt(3)
                p1r3 = 1/np.sqrt(3)

                x = [m1r3, m1r3, m1r3, m1r3, p1r3, p1r3, p1r3, p1r3]
                y = [m1r3, m1r3, p1r3, p1r3, m1r3, m1r3, p1r3, p1r3]
                z = [m1r3, p1r3, m1r3, p1r3, m1r3, p1r3, m1r3, p1r3]

                poids = [1]*nPg

        elif elemType == "PRISM6":
            dim = 3            
            if matriceType in ["rigi", "masse"]:
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