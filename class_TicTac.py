import time
import numpy as np

class TicTac:

    __start = None

    @staticmethod
    def Tic():
        TicTac.__start = time.time()
    
    @staticmethod
    def Tac(texte: str, affichage: bool):
        if affichage:
            print("\n{} ({:.3f} s)".format(texte, np.abs(TicTac.__start - time.time())))
