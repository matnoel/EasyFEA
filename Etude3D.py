# %%

import Dossier

from Simu import Simu
from Materiau import Materiau, Elas_Isot
from Interface_Gmsh import Interface_Gmsh
from Mesh import Mesh
from Affichage import Affichage
import PostTraitement

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
nBe = 10
taille = h/nBe

if nBe > 3:
        plotResult = False

comportement = Elas_Isot(dim, contraintesPlanes=False)

# Materiau
materiau = Materiau(comportement)

# Construction du modele et du maillage --------------------------------------------------------------------------------
modelGmsh = Interface_Gmsh(dim, organisationMaillage=True, typeElement=0, tailleElement=taille, gmshVerbosity=False, affichageGmsh=False)

fichier = Dossier.NewFile('models\\part.stp')

(coordo, connect) = modelGmsh.Importation3D(fichier)
mesh = Mesh(dim, coordo, connect)

noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Traitement")

simu = Simu(dim,mesh, materiau, verbosity=True)

simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["y"])
simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x", "y", "z"])

simu.Assemblage_u()

simu.Solve_u(useCholesky=True)

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Résultats")

simu.Resume()

if saveParaview:
        filename = Dossier.NewFile("Etude3D\\solution3D", results=True)
        PostTraitement.SaveParaview(simu, filename)

if plotResult:

        tic = TicTac()

        Affichage.Plot_Maillage(simu.mesh, deformation=False)
        # plt.savefig(Dossier.NewFile("Etude3D\\Maillage.png", results=True))
        Affichage.Plot_Maillage(mesh, simu=simu, deformation=True, facteurDef=20)
        # plt.savefig(Dossier.NewFile("Etude3D\\MaillageDef.png", results=True))
        Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True, valeursAuxNoeuds=True)
        # plt.savefig(Dossier.NewFile("Etude3D\\Svm_e.png", results=True))
        
        tic.Tac("Post Traitement","Affichage des figures", plotResult)

TicTac.getResume()

if plotResult:        
        plt.show()

# %%
