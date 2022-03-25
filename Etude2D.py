# %%

from Materiau import Materiau
from ModelGmsh import ModelGmsh
from Mesh import Mesh
from Simu import Simu
from Affichage import Affichage

import numpy as np
import matplotlib.pyplot as plt

from TicTac import TicTac

Affichage.Clear()

ticTot = TicTac()

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13

P = 800 #N

# Paramètres maillage
taille = h/100

# Materiau
materiau = Materiau(dim, epaisseur=b, contraintesPlanes=True)

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
simu.Assemblage_u()

simu.Solve_u(resolution=2, calculContraintesEtDeformation=True, interpolation=False)



# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")



print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_e"]))) 

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_n"]))) 
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_n"]))) 

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_n"]))) 
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_n"])))

if plotResult:

        tic = TicTac()
        
        fig, ax = Affichage.Plot_Maillage(simu, deformation=True)
        # Affichage.Plot_NoeudsMaillage(simu, showId=False)
        Affichage.Plot_Result(simu, "dy_n", deformation=True)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_n", affichageMaillage=False)
        # Affichage.PlotResult(mesh, simu.resultats, "dx_e", affichageMaillage=True)        
        # Affichage.PlotResult(mesh, simu.resultats, "Svm_e")
        Affichage.Plot_Result(simu, "Svm_e")
        # Affichage.PlotResult(simu, "Svm_n")
        # Affichage.PlotResult(simu, "Svm_n", affichageMaillage=True, deformation=True)

        # Affichage.PlotResult(mesh, simu.resultats, "dy_n")
        # Affichage.PlotResult(mesh, simu.resultats, "dy_e", deformation=True, affichageMaillage=True)
        
        tic.Tac("Post Traitement","Affichage des figures", plotResult)

ticTot.Tac("Temps script","Temps total", True)        

TicTac.getResume()        

if plotResult:
        plt.show()

# %%
