import time
import numpy as np

class TicTac:
    
    __Historique = [] 
    @staticmethod
    def getHistorique():
        return TicTac.__Historique

    def __init__(self):
        self.__start = time.time()

    def Tac(self, texte: str, affichage: bool):

        tf = np.abs(self.__start - time.time())

        texte = "\n{} ({:.3f} s)".format(texte, tf)
        TicTac.__Historique.append([tf, texte])

        self.__start = time.time()

        if affichage:
            print(texte)

        return tf
