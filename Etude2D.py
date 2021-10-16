import os
from typing import cast
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
type = "QUAD8"
taille = h/2

materiau = Materiau(dim)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=type, tailleElement=taille)
(coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
mesh = Mesh(dim, coordo, connect)

# ------------------------------------------------------------------------------------------------------

print("\n==========================================================")
print("Traitement :")

simu = Simu(dim, mesh, materiau)

simu.Assemblage(epaisseur=b)

# Récupère la coordonnées des noeuds qui m'interesse
noeuds_en_L = []
noeuds_en_0 = []
for n in simu.mesh.noeuds:
        n = cast(Noeud, n)        
        if n.coordo[0] == L:
                noeuds_en_L.append(n)
        if n.coordo[0] == 0:
                noeuds_en_0.append(n)

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, direction="Y")

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="X")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Y")

simu.Solve()

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Post traitement :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm"]))) 

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx"]))) 
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx"]))) 

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy"]))) 
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy"]))) 

if plotResult:

        simu.PlotMesh()
        simu.PlotResult(resultat="dx",affichageMaillage=True, deformation=True)
        # simu.PlotResult(resultat="dy")

        # simu.PlotResult(resultat="Sxx")
        # simu.PlotResult(resultat="Syy")
        # simu.PlotResult(resultat="Sxy")
        simu.PlotResult(resultat="Svm", affichageMaillage=True, deformation=True)
        
        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



