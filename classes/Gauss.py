from typing import cast
import numpy as np

class Gauss:

    def __init__(self, elemType: str, matriceType: str):
        
        coord, poids = Gauss.__calc_gauss(elemType, matriceType)

        self.__coord = coord
        self.__poids = poids

    def __get_coord(self):
        return self.__coord
    coord = cast(np.ndarray, property(__get_coord))

    def __get_poids(self):
        return self.__poids
    poids = cast(np.ndarray, property(__get_poids))

    def __get_nPg(self):
        return len(self.__poids)
    nPg = property(__get_nPg)

    @staticmethod
    def __calc_gauss(elemType: str, matriceType: str):

        assert matriceType in ["rigi", "mass"]

        if elemType == "TRI3":

            if matriceType == "rigi":

                nPg = 1

                ksis = 1/3
                etas = 1/3
                poids = 1/2

            elif matriceType == "mass":

                nPg = 3

                ksis = [1/6, 2/3, 1/6]
                etas = [1/6, 1/6, 2/3]
                poids = [1/6] * 3

        elif elemType == "TRI6":

            if matriceType == "rigi":

                nPg = 3

                ksis = [1/6, 2/3, 1/6]
                etas = [1/6, 1/6, 2/3]
                poids = [1/6] * 3

            elif matriceType == "mass":

                nPg = 6

                a = 0.445948490915965
                b = 0.091576213509771
                p1 = 0.111690794839005
                p2 = 0.054975871827661

                ksis = [b, 1-2*b, b, a, a, 1-2*a]
                etas = [b, b, 1-2*b, 1-2*a, a, a]
                poids = [p2, p2, p2, p1, p1, p1]

        elif elemType == "QUAD4":

            if matriceType in ["rigi", "mass"]:
                nPg = 4

                a = 1/np.sqrt(3)

                ksis = [-a, a, a, -a]
                etas = [-a, -a, a, a]
                poids = [1]*nPg

        elif elemType == "QUAD8":

            if matriceType in ["rigi", "mass"]:

                nPg = 9

                a = 0.774596669241483

                ksis = [-a, a, a, -a, 0, a, 0, -a, 0]
                etas = [-a, -a, a, a, -a, 0, a, 0, 0]
                poids = [25/81, 25/81, 25/81, 25/81, 40/81, 40/81, 40/81, 40/81, 64/81]

        elif elemType == "TETRA4":

            if matriceType == "rigi":

                nPg = 1

                x = 1/4
                y = 1/4
                z = 1/4
                poids = 1/6

            elif matriceType == "mass":

                nPg = 4

                a = (5-np.sqrt(5))/20
                b = (5+3*np.sqrt(5))/20

                x = [a, a, a, b]
                y = [a, a, b, a]
                z = [a, b, a, a]
                poids = [1/24]*nPg
        
        if elemType == "TETRA4":

            coord = np.array([x, y, z]).reshape((nPg, 3))

        else:

            coord = np.array([ksis, etas]).reshape((nPg, 2))

        poids = np.array(poids).reshape((nPg, 1))

        return coord, poids