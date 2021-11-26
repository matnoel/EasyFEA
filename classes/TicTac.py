import time
import numpy as np

class TicTac:

    __start = 0

    @staticmethod
    def Tic():
        TicTac.__start = time.time()
    
    @staticmethod
    def Tac(texte: str, affichage: bool):
        if affichage:
            tf = np.abs(TicTac.__start - time.time())
            print("\n{} ({:.3f} s)".format(texte, tf))
        return tf
