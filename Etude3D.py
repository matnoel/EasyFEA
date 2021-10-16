import os
import sys
from typing import cast
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

affichageGMSH = False

plotResult = True

type = "T4"
maillageOrganisé = False

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = -800 #N

# Paramètres maillage
type = "TETRA4"
taille = L

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

simu.Assemblage(epaisseur=b)

noeuds_en_L = []
noeuds_en_0 = []
for n in mesh.noeuds:
        n = cast(Noeud, n)        
        if n.coordo[0] == L:
                noeuds_en_L.append(n)
        if n.coordo[0] == 0:
                noeuds_en_0.append(n)

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, direction="Z")

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="X")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Y")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Z")

simu.Solve()

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Résultats :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"]))

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm"])))

print("\nSxx max = {:.6f} MPa".format(np.max(simu.resultats["Sxx"])))
print("Syy max = {:.6f} MPa".format(np.max(simu.resultats["Syy"])))
print("Szz max = {:.6f} MPa".format(np.max(simu.resultats["Szz"])))
print("Sxy max = {:.6f} MPa".format(np.max(simu.resultats["Sxy"])))
print("Syz max = {:.6f} MPa".format(np.max(simu.resultats["Syz"])))
print("Sxz max = {:.6f} MPa".format(np.max(simu.resultats["Sxz"])))

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx"])))
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx"])))

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy"])))
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy"])))

print("\nUz max = {:.6f} mm".format(np.max(simu.resultats["dz"])))
print("Uz min = {:.6f} mm".format(np.min(simu.resultats["dz"])))


if plotResult:

        simu.PlotMesh(deformation=True)
        simu.PlotResult(resultat="dx", deformation=True)
        # simu.PlotResult(resultat="dy")
        # simu.PlotResult(resultat="dz", affichageMaillage=True)

        # simu.PlotResult(resultat="Sxx")
        # simu.PlotResult(resultat="Syy")
        # simu.PlotResult(resultat="Sxy")
        simu.PlotResult(resultat="Svm", affichageMaillage=True, deformation=True)
        
        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



