# %%

import Dossier

from Simu import Simu
from Materiau import Materiau, Elas_Isot
from Interface_Gmsh import Interface_Gmsh
from Mesh import Mesh
import Affichage
import PostTraitement
from Geom import *

import numpy as np
import matplotlib.pyplot as plt

from TicTac import TicTac

Affichage.Clear()

# Data --------------------------------------------------------------------------------------------

dim = 3 

plotResult = True

saveParaview = True

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = 800 #N

# Paramètres maillage
nBe = 3
taille = h/nBe

if nBe > 3:
        plotResult = False

comportement = Elas_Isot(dim, contraintesPlanes=False)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
interfaceGmsh = Interface_Gmsh(gmshVerbosity=False, affichageGmsh=False)

fichier = Dossier.NewFile('models\\part.stp')

mesh = interfaceGmsh.Importation3D(fichier, tailleElement=taille)

noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(mesh, materiau, verbosity=True)

simu.add_surfLoad("displacement",noeuds_en_L, ["y"], [-P/h/b])
simu.add_dirichlet("displacement",noeuds_en_0, ["x","y","z"], [0,0,0])

simu.Assemblage_u()

simu.Solve_u(useCholesky=True)

simu.Save_solutions()

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Résultats")

simu.Resume()

if saveParaview:
        folder = Dossier.NewFile("Etude3D", results=True)        
        PostTraitement.Save_Simulation_in_Paraview(folder, simu)

if plotResult:

        tic = TicTac()

        Affichage.Plot_Maillage(simu, deformation=False)
        # plt.savefig(Dossier.NewFile("Etude3D\\Maillage.png", results=True))
        Affichage.Plot_Maillage(simu, deformation=True, facteurDef=20)
        # plt.savefig(Dossier.NewFile("Etude3D\\MaillageDef.png", results=True))
        Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True, valeursAuxNoeuds=True)
        # plt.savefig(Dossier.NewFile("Etude3D\\Svm_e.png", results=True))
        
        tic.Tac("Post Traitement","Affichage des figures", plotResult)

TicTac.getResume()

if plotResult:        
        plt.show()

# %%
