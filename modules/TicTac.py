
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Tic:

    def __init__(self):
        self.__start = time.time()

    @staticmethod
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

    def Tac(self, categorie="", texte="", affichage=False) -> float:
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
    
    @staticmethod
    def Clear():
        """Supprime l'historique"""
        Tic.__Historique = {}
    
    __Historique = {}
    """historique des temps = { catégorie: list( [texte, temps] ) }"""
       
    @staticmethod
    def Resume(verbosity=True):
        """Construit le résumé de TicTac"""

        if Tic.__Historique == {}: return

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
    def __plotBar(ax: plt.Axes, categories: list, temps: list, reps: int, titre: str):
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

        for i, (texte, tmps, rep) in enumerate(zip(categories, temps, reps)):
            # height=0.55
            # ax.barh(i, t, height=height, align="center", label=c)            
            ax.barh(i, tmps, align="center", label=texte)
            
            # On rajoute un peu d'espace a la fin du texte
            espace = " "

            temps, unite = Tic.Get_temps_unite(tmps/rep)

            
            if rep > 1:
                repTemps = f" ({rep} x {np.round(temps,2)} {unite})"
            else:
                repTemps = f" ({np.round(temps,2)} {unite})"

            texte = espace + texte + repTemps + espace

            if tmps/tempsMax < 0.6:
                ax.text(tmps, i, texte, color='black',
                verticalalignment='center', horizontalalignment='left')
            else:
                ax.text(tmps, i, texte, color='white',
                verticalalignment='center', horizontalalignment='right')

        # plt.legend()
        ax.set_title(titre)

    @staticmethod 
    def Plot_History(folder="", details=True):
        """Affiche l'historique

        Parameters
        ----------
        folder : str, optional
            dossier dans lequel on va sauvegarder les figures, by default ""
        details : bool, optional
            Affiche de détails de l'historique, by default True
        """

        import Display

        if Tic.__Historique == {}: return

        historique = Tic.__Historique        
        tempsTotCategorie = []
        categories = list(historique.keys())

        # récupère le temps de chaque catégorie
        tempsCategorie = [np.sum(np.array(np.array(historique[c])[:,1] , dtype=np.float64)) for c in categories]

        categories = np.array(categories)[np.argsort(tempsCategorie)][::-1]

        for i, c in enumerate(categories):

            #temps des sous categories de c
            tempsSousCategorie = np.array(np.array(historique[c])[:,1] , dtype=np.float64)
            tempsTotCategorie.append(np.sum(tempsSousCategorie)) #somme tout les temps de cette catégorie

            sousCategories = np.array(np.array(historique[c])[:,0] , dtype=str) #sous catégories

            # On construit un tableau pour les sommé sur les sous catégories
            dfSousCategorie = pd.DataFrame({'sous categories' : sousCategories, 'temps': tempsSousCategorie, 'rep': 1})
            dfSousCategorie = dfSousCategorie.groupby(['sous categories']).sum()
            dfSousCategorie = dfSousCategorie.sort_values(by='temps')
            sousCategories = dfSousCategorie.index.tolist()

            # print(dfSousCategorie)

            if len(sousCategories) > 1 and details and tempsTotCategorie[-1]>0:
                fig, ax = plt.subplots()
                Tic.__plotBar(ax, sousCategories, dfSousCategorie['temps'].tolist(), dfSousCategorie['rep'].tolist(), c)
            
                if folder != "":                        
                    Display.Save_fig(folder, f"TicTac{i}_{c}")

        # On construit un tableau pour les sommé sur les sous catégories
        dfCategorie = pd.DataFrame({'categories' : categories, 'temps': tempsTotCategorie})
        dfCategorie = dfCategorie.groupby(['categories']).sum()
        dfCategorie = dfCategorie.sort_values(by='temps')
        categories = dfCategorie.index.tolist()
        
        fig, ax = plt.subplots()
        Tic.__plotBar(ax, categories, dfCategorie['temps'], [1]*dfCategorie.shape[0], "Simulation")

        if folder != "":            
            Display.Save_fig(folder, "TicTac_Simulation")

        # # Camembert
        # my_circle = plt.Circle( (0,0), 0, color='white')
        # # Give color names
        # plt.pie(tempsCatégories, labels=tempsCatégories,
        # wedgeprops = { 'linewidth' : 0, 'edgecolor' : 'white' })
        # p = plt.gcf()
        # ax1.add_artist(my_circle)

        # # On contstruit un disque pour chaque sous catégorie d'une catégorie
    
