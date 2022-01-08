# %%

import os

from typing import cast

from classes.Affichage import Affichage
from classes.Materiau import Materiau
from classes.ModelGmsh import ModelGmsh
from classes.Mesh import Mesh
from classes.Simu import Simu

import numpy as np
import matplotlib.pyplot as plt

os.system("cls")    #nettoie le terminal

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13

P = 800 #N

# Paramètres maillage
taille = h/3

# Materiau
materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=1, tailleElement=taille)
(coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
mesh = Mesh(dim, coordo, connect)

# Récupère les noeuds qui m'interessent
noeuds_en_L = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == L]
noeuds_en_0 = [n for n in range(mesh.Nn) if mesh.coordo[n,0] == 0]

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(dim, mesh, materiau)

simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x","y"])

# simu.Condition_Dirichlet(noeuds_en_L, valeur=-10, directions=["x"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=1, directions=["y"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=1, directions=["x","y"])
# simu.Condition_Dirichlet(noeuds_en_L, valeur=-1, directions=["x","y"])


simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["y"])
# simu.Condition_Neumann(noeuds_en_L, valeur=P, directions=["y"])




simu.Assemblage_u(epaisseur=b)



simu.Solve_u(resolution=2)

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
        Affichage.PlotResult(simu, "Svm_n")
        Affichage.PlotResult(simu, "Svm_n", affichageMaillage=True, deformation=True)

        # Affichage.PlotResult(mesh, simu.resultats, "dy_n")
        # Affichage.PlotResult(mesh, simu.resultats, "dy_e", deformation=True, affichageMaillage=True)
        
        plt.show()

Affichage.NouvelleSection("Fin du programme")

# %%
