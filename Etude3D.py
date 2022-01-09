from ntpath import join
import os

from typing import cast
from classes.Affichage import Affichage

from classes.Simu import Simu
from classes.Materiau import Materiau
from classes.ModelGmsh import ModelGmsh
from classes.Mesh import Mesh

import numpy as np
import matplotlib.pyplot as plt

os.system("cls")    #nettoie terminal

# Data --------------------------------------------------------------------------------------------

dim = 3

plotResult = True

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = -800 #N

# Paramètres maillage
type = "TETRA4"
taille = h/2

materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=0, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False)

dir_path = os.path.dirname(os.path.realpath(__file__))
fichier = dir_path + '\\models\\part.stp'

(coordo, connect) = modelGmsh.Importation3D(fichier)
mesh = Mesh(dim, coordo, connect)

# ------------------------------------------------------------------------------------------------------
print("\n==========================================================")
print("Traitement :")

simu = Simu(dim,mesh, materiau, verbosity=True)

simu.Assemblage_u(epaisseur=b)

coordo = mesh.coordo
noeuds_en_L = [n for n in range(mesh.Nn) if coordo[n,0] == L]
noeuds_en_0 = [n for n in range(mesh.Nn) if coordo[n,0] == 0]

simu.Condition_Neumann(noeuds_en_L, valeur=P, directions=["z"])

simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x", "y", "z"])

simu.Solve_u(calculContraintesEtDeformation=True, interpolation=True)

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Résultats :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"]))

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_n"])))

print("\nSxx max = {:.6f} MPa".format(np.max(simu.resultats["Sxx_n"])))
print("Syy max = {:.6f} MPa".format(np.max(simu.resultats["Syy_n"])))
print("Szz max = {:.6f} MPa".format(np.max(simu.resultats["Szz_n"])))
print("Sxy max = {:.6f} MPa".format(np.max(simu.resultats["Sxy_n"])))
print("Syz max = {:.6f} MPa".format(np.max(simu.resultats["Syz_n"])))
print("Sxz max = {:.6f} MPa".format(np.max(simu.resultats["Sxz_n"])))

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_n"])))
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_n"])))

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_n"])))
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_n"])))

print("\nUz max = {:.6f} mm".format(np.max(simu.resultats["dz_n"])))
print("Uz min = {:.6f} mm".format(np.min(simu.resultats["dz_n"])))


if plotResult:

        Affichage.PlotMesh(simu, deformation=False)
        Affichage.PlotMesh(simu, deformation=True, facteurDef=20)
        Affichage.PlotResult(simu, "Svm_e", deformation=True, affichageMaillage=True)

        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



