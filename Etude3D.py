import os

from typing import cast
from Affichage import Affichage
from class_Noeud import Noeud

from class_Simu import Simu
from class_Materiau import Materiau
from class_ModelGmsh import ModelGmsh
from class_Mesh import Mesh

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
taille = h

materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False)

fichier = "part.stp"

(coordo, connect) = modelGmsh.Importation3D(fichier)
mesh = Mesh(dim, coordo, connect)

# ------------------------------------------------------------------------------------------------------
print("==========================================================")
print("Traitement :")

simu = Simu(dim,mesh, materiau, verbosity=True)

simu.AssemblageKglobFglob(epaisseur=b)

noeuds_en_L = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == L]
noeuds_en_0 = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == 0]

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, directions=["z"])

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="x")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="y")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="z")

simu.Solve()

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

        Affichage.PlotMesh(mesh, simu.resultats, deformation=False)
        Affichage.PlotMesh(mesh, simu.resultats, deformation=True, facteurDef=20)
        Affichage.PlotResult(mesh, simu.resultats, "Svm_e", deformation=True, affichageMaillage=True)

        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



