
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Tic:

    @staticmethod
    def Clear():
        """Supprime l'historique"""
        Tic.__Historique = {}
    
    __Historique = {}
    """historique des temps = { catégorie: list( [texte, temps] ) }"""
       
    @staticmethod
    def getResume(verbosity=True):
        """Construit le résumé de TicTac"""

        if Tic.__Historique == {}: return

        # print("\n Résumé TicTac :")

        # import Affichage
        # Affichage.NouvelleSection("Résumé TicTac")

        resume = ""

        for categorie in Tic.__Historique:
            histoCategorie = np.array(np.array(Tic.__Historique[categorie])[:,1] , dtype=np.float64)
            tempsCatégorie = np.sum(histoCategorie)
            tempsCatégorie, unite = Tic.Get_temps_unite(tempsCatégorie)
            resumeCatégorie = f"{categorie} : {tempsCatégorie:.3f} {unite}"
            if verbosity: print(resumeCatégorie)
            resume += '\n' + resumeCatégorie

        return resume
            

    @staticmethod
    def __plotBar(ax: plt.Axes, categories: list, temps: list, titre: str):
        # Parmètres axes
        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)
        ax.yaxis.set_visible(False)
        ax.set_axisbelow(True)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_lw(1.5)

        ax.grid(axis = "x", lw=1.2)

        tempsMax = np.max(temps)

        # Je veux que si le temps représente < 0.5 tempsTotal on affiche le texte a droite
        # Sinon on va lafficher a gauche

        for i, (texte, tmps) in enumerate(zip(categories, temps)):
            # height=0.55
            # ax.barh(i, t, height=height, align="center", label=c)            
            ax.barh(i, tmps, align="center", label=texte)
            
            # On rajoute un peu d'espace a la fin du texte
            espace = " "
            texte = espace + texte + espace

            if tmps/tempsMax < 0.4:
                ax.text(tmps, i, texte, color='black',
                verticalalignment='center', horizontalalignment='left')
            else:
                ax.text(tmps, i, texte, color='white',
                verticalalignment='center', horizontalalignment='right')

        # plt.legend()
        ax.set_title(titre)

    @staticmethod 
    def getGraphs(folder="", details=True, title="Simulation"):

        import PostTraitement as PostTraitement

        if Tic.__Historique == {}: return

        historique = Tic.__Historique        
        tempsTotCategorie = []
        categories = list(historique.keys())

        for c in categories:

            tempsSousCategorie = np.array(np.array(historique[c])[:,1] , dtype=np.float64) #temps des sous categories de c
            tempsTotCategorie.append(np.sum(tempsSousCategorie)) #somme tout les temps de cette catégorie
            sousCategories = np.array(np.array(historique[c])[:,0] , dtype=str) #sous catégories

            # On construit un tableau pour les sommé sur les sous catégories
            dfSousCategorie = pd.DataFrame({'sous categories' : sousCategories, 'temps': tempsSousCategorie})
            dfSousCategorie = dfSousCategorie.groupby(['sous categories']).sum()
            dfSousCategorie = dfSousCategorie.sort_values(by='temps')
            sousCategories = dfSousCategorie.index.tolist()

            # print(dfSousCategorie)

            if len(sousCategories) > 1 and details and tempsTotCategorie[-1]>0:
                fig, ax = plt.subplots()
                Tic.__plotBar(ax, sousCategories, dfSousCategorie['temps'].tolist(), c)
            
                if folder != "":                        
                    PostTraitement.Save_fig(folder, c)
            
        

        # On construit un tableau pour les sommé sur les sous catégories
        dfCategorie = pd.DataFrame({'categories' : categories, 'temps': tempsTotCategorie})
        dfCategorie = dfCategorie.groupby(['categories']).sum()
        dfCategorie = dfCategorie.sort_values(by='temps')
        categories = dfCategorie.index.tolist()
        
        fig, ax = plt.subplots()
        Tic.__plotBar(ax, categories, dfCategorie['temps'], "Simulation")

        if folder != "":            
            PostTraitement.Save_fig(folder, title)

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

    def Get_temps_unite(temps):
        """Renvoie le temps et l'unité"""
        if temps > 1:
            if temps < 60:
                unite = "s"
                coef = 1
            elif temps > 60 and temps < 3600:
                unite = "m"
                coef = 1/60
            elif temps > 3600 and temps < 86400:
                unite = "h"
                coef = 1/3600
            else:
                unite = "j"
                coef = 1/86400
        elif temps < 1 and temps > 1e-3:
            coef = 1e3
            unite = "ms"
        elif temps < 1e-3:
            coef = 1e6
            unite = "µs"

        return temps*coef, unite

    def Tac(self, categorie: str, texte: str, affichage: bool):
        """calcul le temps et stock dans l'historique"""

        tf = np.abs(self.__start - time.time())

        tfCoef, unite = Tic.Get_temps_unite(tf)

        texteAvecLeTemps = f"{texte} ({tfCoef:.3f} {unite})"
        
        value = [texte, tf]

        if categorie in Tic.__Historique:
            old = list(Tic.__Historique[categorie])
            old.append(value)
            Tic.__Historique[categorie] = old
        else:
            Tic.__Historique[categorie] = [value]
        
        self.__start = time.time()

        if affichage:
            print(texteAvecLeTemps)

        return tf
    
    
