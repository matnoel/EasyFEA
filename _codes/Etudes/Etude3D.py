# %%
import os
import Dossier as Dossier

from Simu import Simu
from Materials import Materiau, Elas_Isot
from Interface_Gmsh import Interface_Gmsh
from Mesh import Mesh
import Affichage as Affichage
import PostTraitement as PostTraitement
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

saveParaview = False

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = 800 #N

# Paramètres maillage
# nBe = 20
nBe = 5

taille = h/nBe
# taille = h/3

# if nBe > 11:
#     plotResult = False

comportement = Elas_Isot(dim)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
interfaceGmsh = Interface_Gmsh(gmshVerbosity=False, affichageGmsh=False)

fichier = Dossier.NewFile(os.path.join("3Dmodels","part.stp"))

# # Avec importation
# mesh = interfaceGmsh.Importation3D(fichier, tailleElement=taille)

# "TETRA4", "HEXA8", "PRISM6"
# # Sans importation
domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
circle = Circle(Point(x=L/2, y=0), h*0.8, taille=taille, isCreux=True)
mesh = interfaceGmsh.PlaqueAvecCercle3D(domain,circle ,[0,0,b], elemType="TETRA4", isOrganised=False, nCouches=3)
# mesh = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType="PRISM6", isOrganised=True, nCouches=4)


volume = mesh.volume - L*b*h
aire = mesh.aire - (L*h*4 + 2*b*h)

# Affichage.Plot_Maillage(mesh)
# # plt.show()


noeuds_en_0 = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

# Affichage.Plot_ElementsMaillage(mesh, dimElem=1, nodes=noeuds_en_L, showId=True)
# Affichage.Plot_ElementsMaillage(mesh, dimElem=2, nodes=noeuds_en_L, showId=True)
# plt.show()

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(mesh, materiau, verbosity=True, useNumba=False)

simu.add_surfLoad("displacement",noeuds_en_L, [-P/h/b], ["y"])
simu.add_dirichlet("displacement",noeuds_en_0, [0,0,0], ["x","y","z"])

# Affichage.Plot_BoundaryConditions(simu)
# # plt.show()

simu.Assemblage_u()
simu.Solve_u()

# simu.Assemblage_u()
# simu.Solve_u()

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
    Tic.getGraphs()
    plt.show()

# %%
