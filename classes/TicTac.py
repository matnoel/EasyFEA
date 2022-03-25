import time
import numpy as np

class TicTac:
    
    __Historique = {} 
       
    @staticmethod
    def getResume():

        # print("\n Résumé TicTac :")

        import Affichage

        Affichage.Affichage.NouvelleSection("Résumé TicTac")        

        for categorie in TicTac.__Historique:
            histoCategorie = np.array(np.array(TicTac.__Historique[categorie])[:,0] , dtype=np.float64)
            texte = "{} : {:.3f} s".format(categorie, np.sum(histoCategorie))
            print(texte)

    def __init__(self):
        self.__start = time.time()

    def Tac(self, categorie: str, texte: str, affichage: bool):

        tf = np.abs(self.__start - time.time())

        texteAvecLeTemps = "\n{} ({:.3f} s)".format(texte, tf)        
        
        value = [tf, texte]

        if categorie in TicTac.__Historique:
            old = list(TicTac.__Historique[categorie])
            old.append(value)
            TicTac.__Historique[categorie] = old
        else:
            TicTac.__Historique[categorie] = [value]
        
        self.__start = time.time()

        if affichage:
            print(texteAvecLeTemps)

        return tf
    
    
