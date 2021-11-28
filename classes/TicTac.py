import time
import numpy as np

class TicTac:

    __start = 0

    @staticmethod
    def Tic():
        TicTac.__start = time.time()
    
    @staticmethod
    def Tac(texte: str, affichage: bool):
        tf = np.abs(TicTac.__start - time.time())
        if affichage:
            print("\n{} ({:.3f} s)".format(texte, tf))
        return tf
