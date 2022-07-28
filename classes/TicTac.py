import time
import numpy as np
import matplotlib.pyplot as plt

class TicTac:
    
    __Historique = {}
    """historique des temps (catégorie, temps, texte)"""
       
    @staticmethod
    def getResume():
        """Construit le résumé de TicTac"""

        # print("\n Résumé TicTac :")

        import Affichage

        Affichage.NouvelleSection("Résumé TicTac")

        for categorie in TicTac.__Historique:
            histoCategorie = np.array(np.array(TicTac.__Historique[categorie])[:,0] , dtype=np.float64)
            tempsCatégorie = np.sum(histoCategorie)
            texte = f"{categorie} : {tempsCatégorie:.3f} s"
            print(texte)

    @staticmethod 
    def getGraphs(folder=""):

        # On construit un disque avec toute les catégories
        tempsCatégories = []
        tempsCatégoriesStr = []

        for categorie in TicTac.__Historique:
            histoCategorie = np.array(np.array(TicTac.__Historique[categorie])[:,0] , dtype=np.float64)
            tempsCatégories.append(np.sum(histoCategorie))
            tempsCatégoriesStr.append(f"{np.round(np.sum(histoCategorie),3)} s")

        fig1, ax1 = plt.subplots()

        # # Camembert
        # my_circle = plt.Circle( (0,0), 0, color='white')
        # # Give color names
        # plt.pie(tempsCatégories, labels=tempsCatégories,
        # wedgeprops = { 'linewidth' : 0, 'edgecolor' : 'white' })
        # p = plt.gcf()
        # ax1.add_artist(my_circle)


        # Batton horizontal
        ax1.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)
        ax1.set_axisbelow(True)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax1.spines["left"].set_lw(1.5)

        ax1.grid(axis = "x", color="#A8BAC4", lw=1.2)

        y_pos = np.arange(len(TicTac.__Historique))
        ax1.barh(y_pos, tempsCatégories)
        plt.yticks(y_pos, TicTac.__Historique)

        # # This allows us to determine exactly where each bar is located
        # y = [i * 0.9 for i in range(len(TicTac.__Historique))]
        
        # PAD = 0.3
        # for name, count, y_pos in zip(TicTac.__Historique, tempsCatégories, y):
        #     x = 0
        #     color = "white"
        #     path_effects = None
        #     if count < 8:
        #         x = count
        #         color = "blue"    
        #         path_effects=[withStroke(linewidth=6, foreground="white")]
            
        #     ax1.text(
        #         x + PAD, y_pos + 0.5 / 2, name, 
        #         color=color, fontfamily="Econ Sans Cnd", fontsize=18, va="center",
        #         path_effects=path_effects
        #     )
        

        # # On contstruit un disque pour chaque sous catégorie d'une catégorie


        

    def __init__(self):
        self.__start = time.time()

    def Tac(self, categorie: str, texte: str, affichage: bool):
        """calcul le temps et stock dans l'historique"""

        tf = np.abs(self.__start - time.time())

        texteAvecLeTemps = "{} ({:.3f} s)".format(texte, tf)        
        
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
    
    
