# %%

import os

from typing import cast

from classes.Materiau import Materiau
from classes.ModelGmsh import ModelGmsh
from classes.Mesh import Mesh
from classes.Simu import Simu
from classes.Affichage import Affichage

import numpy as np
import matplotlib.pyplot as plt

from classes.TicTac import TicTac

os.system("cls")    #nettoie le terminal

tic = TicTac()

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13

P = 800 #N

# Paramètres maillage
taille = h/20

# Materiau
materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=0, tailleElement=taille)
(coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
mesh = Mesh(dim, coordo, connect)

# Récupère les noeuds qui m'interessent

noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(dim, mesh, materiau)

# # Affichage etc
# fig, ax = Affichage.PlotMesh(simu, deformation=True)
# Affichage.AfficheNoeudsMaillage(simu, ax, noeuds_en_0, c='red')
# Affichage.AfficheNoeudsMaillage(simu, ax, noeuds_en_L, c='blue', showId=True)
# Affichage.PlotMesh(simu, deformation=True)
# Affichage.AfficheNoeudsMaillage(simu, showId=True)

# Renseigne les condtions limites
simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x","y"])

# simu.Condition_Dirichlet(noeuds_en_L, valeur=-10, directions=["x"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=1, directions=["y"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=1, directions=["x","y"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=-1, directions=["x","y"])

simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["y"])
# simu.Condition_Neumann(noeuds_en_L, valeur=P, directions=["y"])


# Assemblage du système matricielle
simu.Assemblage_u(epaisseur=b)

simu.Solve_u(resolution=2, calculContraintesEtDeformation=True, interpolation=True)

tic.Tac("Temps total", True)

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

# print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_n"]))) 

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_n"]))) 
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_n"]))) 

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_n"]))) 
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_n"]))) 

if plotResult:
        
        # Affichage.PlotMesh(mesh, simu.resultats, deformation=False)
        fig, ax = Affichage.PlotMesh(simu, deformation=True)
        # Affichage.AfficheNoeudsMaillage(simu, showId=True)
        Affichage.PlotResult(simu, "amplitude", deformation=True)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_n", affichageMaillage=False)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_e", affichageMaillage=True)        
        # Affichage.PlotResult(mesh, simu.resultats, "Svm_e")
        Affichage.PlotResult(simu, "Svm_e")
        Affichage.PlotResult(simu, "Svm_n")
        Affichage.PlotResult(simu, "Svm_n", affichageMaillage=True, deformation=True)

        # Affichage.PlotResult(mesh, simu.resultats, "dy_n")
        # Affichage.PlotResult(mesh, simu.resultats, "dy_e", deformation=True, affichageMaillage=True)
        
        plt.show()

Affichage.NouvelleSection("Fin du programme")

# %%
