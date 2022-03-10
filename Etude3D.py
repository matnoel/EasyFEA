# %%

import os

from classes.Simu import Simu
from classes.Materiau import Materiau
from classes.ModelGmsh import ModelGmsh
from classes.Mesh import Mesh
from classes.Affichage import Affichage

import numpy as np
import matplotlib.pyplot as plt

from classes.TicTac import TicTac

os.system("cls")    #nettoie terminal

# Data --------------------------------------------------------------------------------------------

dim = 3

plotResult = True

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = 800 #N

# Paramètres maillage
nBe = 2
taille = h/nBe

if nBe > 3:
        plotResult = False


materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=0, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False)

dir_path = os.path.dirname(os.path.realpath(__file__))
fichier = dir_path + '\\models\\part.stp'

(coordo, connect) = modelGmsh.Importation3D(fichier)
mesh = Mesh(dim, coordo, connect)

noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
print("\n==========================================================")
print("Traitement :")

simu = Simu(dim,mesh, materiau, verbosity=True)

simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["z"])
simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x", "y", "z"])

simu.Assemblage_u()

simu.Solve_u(calculContraintesEtDeformation=True, interpolation=False)

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Résultats :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"]))

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_e"])))

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_n"])))
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_n"])))

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_n"])))
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_n"])))

print("\nUz max = {:.6f} mm".format(np.max(simu.resultats["dz_n"])))
print("Uz min = {:.6f} mm".format(np.min(simu.resultats["dz_n"])))

if plotResult:

        tic = TicTac()

        Affichage.Plot_Maillage(simu, deformation=False)
        Affichage.Plot_Maillage(simu, deformation=True, facteurDef=20)
        Affichage.Plot_Result(simu, "Svm_e", deformation=True, affichageMaillage=True)
        
        tic.Tac("Affichage des figures", plotResult)

        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")




# %%
