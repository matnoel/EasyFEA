# %%

import PostTraitement
import Dossier
import Affichage

from Materiau import PhaseFieldModel, Elas_Isot, Materiau
from Geom import *
from Interface_Gmsh import Interface_Gmsh
from Simu import Simu
from Mesh import Mesh
from TicTac import TicTac

import numpy as np
import matplotlib.pyplot as plt

Affichage.Clear()

test = True

solve = False

plotResult = True
saveParaview = True
makeMovie = False
save = False

# Data --------------------------------------------------------------------------------------------

folder = "Etude_Cisaillement"

comportement = "Elas_Isot" # "Elas_Isot"

split = "Miehe" # "Bourdin","Amor","Miehe"

regularisation = "AT1" # "AT1", "AT2"

openCrack = True

nameSimu = '_'.join([comportement,split,regularisation])
if openCrack: 
        nameSimu += '_openCrack'

dim = 2

# Paramètres géométrie
L = 1e-3;  #m
l0 = 1e-5 # taille fissure test femobject ,7.5e-6, 1e-5
Gc = 2.7e3

# Paramètres maillage
if test:
        taille = 1e-5 #taille maille test fem object
        taille *= 2
else:
        taille = l0/2 #l0/2 2.5e-6

if test:
    folder = Dossier.Append([folder, "Test", nameSimu])
else:
    folder = Dossier.Append([folder, nameSimu])

folder = Dossier.NewFile(folder, results=True)

# Construction du modele et du maillage --------------------------------------------------------------------------------

if solve:

        elemType = "TRI3" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]

        interfaceGmsh = Interface_Gmsh(affichageGmsh=False)

        if openCrack:
                meshName = "carré avec fissure ouverte.msh"
        else:
                meshName = "carré avec fissure fermée.msh"

        mshFileName = Dossier.NewFile(meshName, folder)
        # mshFileName = ""

        domain = Domain(Point(), Point(x=L, y=L))
        line = Line(Point(y=L/2), Point(x=L/2, y=L/2))

        mesh = interfaceGmsh.RectangleAvecFissure(domain=domain, line=line, elemType=elemType,
        elementSize=taille, isOrganised=True, openCrack=openCrack, filename=mshFileName)

        # Affichage.Plot_NoeudsMaillage(mesh, showId=True)
        # Affichage.Plot_Maillage(mesh)
        # plt.show()

        # Récupère les noeuds qui m'interessent
        noeuds_Milieu = mesh.Get_Nodes_Line(line)
        noeuds_Haut = mesh.Get_Nodes_Conditions(conditionY=lambda y: y == L)
        noeuds_Bas = mesh.Get_Nodes_Conditions(conditionY=lambda y: y == 0)
        noeuds_Gauche = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
        noeuds_Droite = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

        NoeudsBord=[]
        NoeudsBord.extend(noeuds_Bas)
        NoeudsBord.extend(noeuds_Droite)
        NoeudsBord.extend(noeuds_Gauche)
        NoeudsBord.extend(noeuds_Haut)

# Simulation  -------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Simulations")

if solve:

        comportement = Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)

        phaseFieldModel = PhaseFieldModel(comportement, split, regularisation, Gc=Gc, l_0=l0)

        materiau = Materiau(comportement, ro=1, phaseFieldModel=phaseFieldModel)

        simu = Simu(mesh, materiau, verbosity=False)

        # Renseignement des conditions limites
        def RenseigneConditionsLimites():
                if not openCrack:
                        simu.add_dirichlet("damage",noeuds_Milieu, ["d"], [1])                        

                # # Conditions en déplacements à Gauche et droite
                simu.add_dirichlet("displacement", noeuds_Gauche,["y"], [0])
                simu.add_dirichlet("displacement", noeuds_Droite,["y"], [0])

                # Conditions en déplacements en Bas
                simu.add_dirichlet("displacement", noeuds_Bas,["x","y"], [0,0])

        RenseigneConditionsLimites()

        N = 400
        # N = 10

        damage_t=[]
        uglob_t=[]

        u_inc = 5e-8
        # u_inc = 5e-7
        dep = 0

        print(phaseFieldModel.resume)

        tic = TicTac()

        for iter in range(N):

                #-------------------------- PFM problem ------------------------------------
                
                simu.Assemblage_d()     # Assemblage
                
                damage = simu.Solve_d() # resolution
                damage_t.append(damage)

                #-------------------------- Dep problem ------------------------------------
                simu.Assemblage_u()

                # Déplacement en haut
                dep += u_inc

                # simu.add_dirichlet("displacement", noeuds_Haut, ["x"], [dep])
                simu.add_dirichlet("displacement", noeuds_Haut, ["x","y"], [dep,0])
                
                uglob = simu.Solve_u(useCholesky=False)
                uglob_t.append(uglob)

                simu.Clear_Bc_Dirichlet()
                RenseigneConditionsLimites()

                simu.Save_solutions()

                # Affiche dans la console
                min = np.round(np.min(damage),3)
                max = np.round(np.max(damage),3)
                norm = np.sum(damage)

                tResolution = tic.Tac("Resolution PhaseField", "Resolution Phase field", False)
                # print(iter+1," : max d = {:.5f}, time = {:.3f}".format(norm, tResolution))
                print(f'{iter+1}/{N} : max d = {max}, min d = {min}, time = {np.round(tResolution,3)} s') 


        # Sauvegarde

        simu.Save(folder)
        
else:   

        simu = Simu.Load(folder)
        


# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Affichage")

if makeMovie:
        # PostTraitement.MakeMovie(folder, "damage", simu)
        PostTraitement.MakeMovie(folder, "Svm", simu)        
        # PostTraitement.MakeMovie(filename, "Syy", simu, valeursAuxNoeuds=True, deformation=True)

def AffichageCL():
        # Affichage noeuds du maillage
        fig1, ax = Affichage.Plot_Maillage(simu.mesh)
        Affichage.Plot_NoeudsMaillage(simu.mesh, ax, noeuds=noeuds_Haut, marker='*', c='blue')
        Affichage.Plot_NoeudsMaillage(simu.mesh, ax, noeuds=noeuds_Milieu, marker='o', c='red')
        Affichage.Plot_NoeudsMaillage(simu.mesh, ax, noeuds=noeuds_Bas, marker='.', c='blue')
        Affichage.Plot_NoeudsMaillage(simu.mesh, ax, noeuds=noeuds_Gauche, marker='.', c='black')
        Affichage.Plot_NoeudsMaillage(simu.mesh, ax, noeuds=noeuds_Droite, marker='.', c='black')

if plotResult:

        # AffichageCL()
        # if save: plt.savefig(f'{folder}\\conditionsLimites.png')

        Wdef = simu.Get_Resultat("Wdef")


        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True,
        affichageMaillage=True, deformation=True, folder=folder)
        

        Affichage.Plot_Result(simu, "dy", folder=folder)
        
if saveParaview:
        PostTraitement.Save_Simulation_in_Paraview(folder, simu)
        



TicTac.getResume()

plt.show()


# %%
