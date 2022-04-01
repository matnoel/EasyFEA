# %%

from Dossier import GetPath

from Simu import Simu
from Materiau import Materiau, Elas_Isot
from ModelGmsh import ModelGmsh
from Mesh import Mesh
from Affichage import Affichage

import numpy as np
import matplotlib.pyplot as plt

from TicTac import TicTac

Affichage.Clear()

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

comportement = Elas_Isot(dim, contraintesPlanes=False)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=0, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False)

dir_path = GetPath()
fichier = dir_path + '\\models\\part.stp'

(coordo, connect) = modelGmsh.Importation3D(fichier)
mesh = Mesh(dim, coordo, connect)

noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(dim,mesh, materiau, verbosity=True)

simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["z"])
simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x", "y", "z"])

simu.Assemblage_u()

simu.Solve_u(useCholesky=False)

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Résultats")

simu.Resume()

if plotResult:

        tic = TicTac()

        Affichage.Plot_Maillage(simu, deformation=False)
        Affichage.Plot_Maillage(simu, deformation=True, facteurDef=20)
        Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True)
        
        tic.Tac("Post Traitement","Affichage des figures", plotResult)

        


TicTac.getResume()

if plotResult:        
        plt.show()

# %%
