# %%
import os
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

folder = Dossier.NewFile("Etude3D", results=True)

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
nBe = 1
# nBe = 1
taille = h/nBe
# taille = h/3

if nBe > 3:
    plotResult = False

comportement = Elas_Isot(dim)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
interfaceGmsh = Interface_Gmsh(gmshVerbosity=False, affichageGmsh=False)

fichier = Dossier.NewFile(os.path.join("models","part.stp"))

# mesh = interfaceGmsh.Importation3D(fichier, elemType="HEXA8", tailleElement=taille)
# mesh = interfaceGmsh.Importation3D(fichier, elemType="HEXA8", tailleElement=taille, folder=folder)

domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
mesh = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType="HEXA8", isOrganised=True, nCouches=3)

noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(mesh, materiau, verbosity=True)

simu.add_surfLoad("displacement",noeuds_en_L, [-P/h/b], ["y"])
simu.add_dirichlet("displacement",noeuds_en_0, [0,0,0], ["x","y","z"])

Affichage.Plot_BoundaryConditions(simu)

simu.Assemblage_u()

simu.Solve_u()

simu.Save_Iteration()

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Résultats")

simu.Resume()

if saveParaview:
    
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)

if plotResult:

    tic = TicTac()

    Affichage.Plot_Maillage(simu, deformation=False)        
    Affichage.Plot_Maillage(simu, deformation=True, facteurDef=20)        
    Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True, valeursAuxNoeuds=True)
    
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

TicTac.getResume()

if plotResult:        
        plt.show()

# %%
