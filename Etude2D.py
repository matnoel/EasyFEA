import os
from typing import cast

from Affichage import Affichage
from class_Materiau import Materiau
from class_ModelGmsh import ModelGmsh
from class_Noeud import Noeud
from class_Simu import Simu
from class_Mesh import Mesh

import numpy as np
import matplotlib.pyplot as plt


os.system("cls")    #nettoie le terminal

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = -800 #N

# Paramètres maillage
type = ModelGmsh.get_typesMaillage2D()[1]
taille = h/10

# Materiau
materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille)

(coordo, connect) = modelGmsh.ConstructionRectangle(L, h)

mesh = Mesh(dim, coordo, connect)

# Récupère les noeuds qui m'interessent
noeuds_en_L = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == L]
noeuds_en_0 = [mesh.noeuds[i] for i in range(mesh.Nn) if mesh.noeuds[i].coordo[0] == 0]

# ------------------------------------------------------------------------------------------------------

print("\n==========================================================")
print("Traitement :")

simu = Simu(dim, mesh, materiau)

simu.AssemblageKglobFglob(epaisseur=b)

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, directions=["y"])

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="x")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="y")

simu.Solve()

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Post traitement :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_e"]))) 

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_e"]))) 
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_e"]))) 

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_e"]))) 
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_e"]))) 

if plotResult:
        
        # Affichage.PlotMesh(mesh, simu.resultats, deformation=False)
        Affichage.PlotMesh(mesh, simu.resultats, deformation=True)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_n", affichageMaillage=True)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_e", affichageMaillage=True)        
        # Affichage.PlotResult(mesh, simu.resultats, "Svm_n")
        # Affichage.PlotResult(mesh, simu.resultats, "Svm_e")

        Affichage.PlotResult(mesh, simu.resultats, "dy_n")
        Affichage.PlotResult(mesh, simu.resultats, "dy_e")
        
                
        
        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



