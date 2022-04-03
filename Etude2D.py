# %%

import Dossier
from Materiau import Elas_Isot, Materiau
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

saveParaview = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13

P = 800 #N

# Paramètres maillage
taille = h/10

comportement = Elas_Isot(dim, epaisseur=b)

# Materiau
materiau = Materiau(comportement)

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
# fig, ax = Affichage.Plot_Maillage(simu, deformation=True)
# Affichage.Plot_NoeudsMaillage(simu, ax, noeuds_en_0, c='red')
# Affichage.Plot_NoeudsMaillage(simu, ax, noeuds_en_L, c='blue', showId=True)
# Affichage.Plot_Maillage(simu, deformation=True)
# Affichage.Plot_NoeudsMaillage(simu, showId=True)

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

simu.Solve_u(resolution=2, useCholesky=True)


# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")

simu.Resume()


if saveParaview:
        filename = Dossier.NewFile("Etude2D\\solution2D.vtu", results=True)
        simu.SaveParaview(filename=filename)

if plotResult:

        tic = TicTac()
        
        fig, ax = Affichage.Plot_Maillage(simu, deformation=True)
        # plt.savefig(Dossier.NewFile("Etude2D\\maillage2D.png",results=True))
        # Affichage.Plot_NoeudsMaillage(simu, showId=True)
        Affichage.Plot_Result(simu, "dy", deformation=True, valeursAuxNoeuds=True)
        # plt.savefig(Dossier.NewFile("Etude2D\\dy.png",results=True))
        Affichage.Plot_Result(simu, "Svm", deformation=True, valeursAuxNoeuds=True)
        # plt.savefig(Dossier.NewFile("Etude2D\\Svm_n.png",results=True))
        Affichage.Plot_Result(simu, "Svm", deformation=True, valeursAuxNoeuds=False, affichageMaillage=False)
        # plt.savefig(Dossier.NewFile("Etude2D\\Svm_e.png",results=True))
        
        
        tic.Tac("Post Traitement","Affichage des figures", plotResult)

ticTot.Tac("Temps script","Temps total", True)        

TicTac.getResume()        

if plotResult:
        plt.show()

# %%
