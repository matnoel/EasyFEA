
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class TicTac:
    
    __Historique = {}
    """historique des temps = { catégorie: list( [texte, temps] ) }"""
       
    @staticmethod
    def getResume():
        """Construit le résumé de TicTac"""

        if TicTac.__Historique == {}: return

        # print("\n Résumé TicTac :")

        import Affichage

        Affichage.NouvelleSection("Résumé TicTac")

        for categorie in TicTac.__Historique:
            histoCategorie = np.array(np.array(TicTac.__Historique[categorie])[:,1] , dtype=np.float64)
            tempsCatégorie = np.sum(histoCategorie)
            texte = f"{categorie} : {tempsCatégorie:.3f} s"
            print(texte)

    @staticmethod
    def __plotBar(ax: plt.Axes, categories: str, temps: float, titre: str):
        # Parmètres axes
        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)
        ax.yaxis.set_visible(False)
        ax.set_axisbelow(True)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_lw(1.5)

        ax.grid(axis = "x", lw=1.2)

        for i, (c, t) in enumerate(zip(categories, temps)):
            # ax1.barh(i, t, height=0.55, align="edge", label=c)
            ax.barh(i, t, align="edge", label=c)

        plt.legend()
        ax.set_title(titre)

    @staticmethod 
    def getGraphs(folder="", details=True):

        if TicTac.__Historique == {}: return

        historique = TicTac.__Historique        
        tempsTotCategorie = []
        categories = list(historique.keys())

        if details:
            for c in categories:

                tempsSousCategorie = np.array(np.array(historique[c])[:,1] , dtype=np.float64) #temps des sous categories de c
                tempsTotCategorie.append(np.sum(tempsSousCategorie)) #somme tout les temps de cette catégorie
                sousCategories = np.array(np.array(historique[c])[:,0] , dtype=str) #sous catégories

                # On construit un tableau pour les sommé sur les sous catégories
                dfSousCategorie = pd.DataFrame({'sous categories' : sousCategories, 'temps': tempsSousCategorie})
                dfSousCategorie = dfSousCategorie.groupby(['sous categories']).sum()
                sousCategories = dfSousCategorie.index.tolist()
                # print(dfSousCategorie)

                if len(sousCategories) > 1:
                    fig, ax = plt.subplots()
                    TicTac.__plotBar(ax, sousCategories, dfSousCategorie['temps'].tolist(), c)
                
                    if folder != "":
                        import PostTraitement
                        PostTraitement.Save_fig(folder, c)
                
        
        fig, ax = plt.subplots()
        TicTac.__plotBar(ax, historique, tempsTotCategorie, "Simulation")

        if folder != "":
            import PostTraitement
            PostTraitement.Save_fig(folder, "Simulation")

        # # Camembert
        # my_circle = plt.Circle( (0,0), 0, color='white')
        # # Give color names
        # plt.pie(tempsCatégories, labels=tempsCatégories,
        # wedgeprops = { 'linewidth' : 0, 'edgecolor' : 'white' })
        # p = plt.gcf()
        # ax1.add_artist(my_circle)

        # # On contstruit un disque pour chaque sous catégorie d'une catégorie


        

    def __init__(self):
        self.__start = time.time()

    def Tac(self, categorie: str, texte: str, affichage: bool):
        """calcul le temps et stock dans l'historique"""

        tf = np.abs(self.__start - time.time())

        texteAvecLeTemps = "{} ({:.3f} s)".format(texte, tf)        
        
        value = [texte, tf]

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
    
    
