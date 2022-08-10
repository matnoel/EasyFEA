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

from TicTac import Tic

# Affichage.Clear()

folder = Dossier.NewFile("Etude3D", results=True)

# Data --------------------------------------------------------------------------------------------

tic= Tic()

dim = 3 

plotResult = True

saveParaview = True

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = 800 #N

# Paramètres maillage
nBe =2
# nBe = 1
taille = h/nBe
# taille = h/3

# if nBe > 11:
#     plotResult = False

comportement = Elas_Isot(dim)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
interfaceGmsh = Interface_Gmsh(gmshVerbosity=False, affichageGmsh=False)

fichier = Dossier.NewFile(os.path.join("models","part.stp"))

# mesh = interfaceGmsh.Importation3D(fichier, elemType="TETRA4", tailleElement=taille)
# mesh = interfaceGmsh.Importation3D(fichier, elemType="HEXA8", tailleElement=taille, folder=folder)
# mesh = interfaceGmsh.Importation3D(fichier, elemType="PRISM6", tailleElement=taille, folder=folder)

domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
mesh = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType="PRISM6", isOrganised=True, nCouches=4)

volume = mesh.volume - L*b*h
aire = mesh.aire - (L*h*4 + 2*b*h)

# circle = Circle(Point(x=L/2, y=0), h*0.8, taille=taille, isCreux=False)
# mesh = interfaceGmsh.PlaqueAvecCercle3D(domain,circle ,[0,0,b], elemType="HEXA8", isOrganised=True, nCouches=3)

# Affichage.Plot_Maillage(mesh)
# plt.show()


noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(mesh, materiau, verbosity=True, useNumba=True)

simu.add_surfLoad("displacement",noeuds_en_L, [-P/h/b], ["y"])
simu.add_dirichlet("displacement",noeuds_en_0, [0,0,0], ["x","y","z"])

Affichage.Plot_BoundaryConditions(simu)
# plt.show()

simu.Assemblage_u()
simu.Solve_u()

simu.Assemblage_u()
simu.Solve_u()

simu.Save_Iteration()

tic.Tac("Temps script", "Temps script", True)

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Résultats")

simu.Resume()

if saveParaview:
    
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)

if plotResult:

    tic = Tic()

    Affichage.Plot_Maillage(simu, deformation=False, folder=folder)        
    Affichage.Plot_Maillage(simu, deformation=True, facteurDef=20, folder=folder)        
    Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True, valeursAuxNoeuds=True, folder=folder)
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

Tic.getResume()

if plotResult:        
    plt.show()

# %%
