# %%
import os
import PhaseFieldSimulation
from BoundaryCondition import BoundaryCondition

import PostTraitement as PostTraitement
import Dossier as Dossier
import Affichage as Affichage

from Materials import Elas_IsotTrans, PhaseFieldModel, Elas_Isot, Materiau
from Geom import *
from Interface_Gmsh import Interface_Gmsh
from Simu import Simu
from Mesh import Mesh
from TicTac import Tic

import numpy as np
import matplotlib.pyplot as plt

Affichage.Clear()

simulation = "Shear" #"Shear" , "Tension"
nomDossier = '_'.join([simulation,"Benchmark"])

test = True
solve = True
plotResult = False
saveParaview = False
makeMovie = False

# Data --------------------------------------------------------------------------------------------

comportement = "Elas_Isot" # "Elas_Isot", "Elas_IsotTrans"
split = "AnisotStress" # "Bourdin","Amor","Miehe","Stress"
regularisation = "AT2" # "AT1", "AT2"
solveur = "History"
openCrack = True

maxIter = 250
# tolConv = 0.0025
# tolConv = 0.005
tolConv = 1e-0

dim = 2

# Paramètres géométrie
L = 1e-3;  #m
l0 = 1e-5 # taille fissure test femobject ,7.5e-6, 1e-5
# l0 = 7.5e-6
Gc = 2.7e3

# Paramètres maillage
if test:
    taille = 1e-5 #taille maille test fem object
    # taille = 0.001
    # taille *= 1.5
else:
    taille = l0/2 #l0/2 2.5e-6
    # taille = 7.5e-6

folder = PhaseFieldSimulation.ConstruitDossier(dossierSource=nomDossier,
comp=comportement, split=split, regu=regularisation, simpli2D='DP',
tolConv=tolConv, solveur=solveur, test=test, openCrack=False, v=0)

# Construction du modele et du maillage --------------------------------------------------------------------------------



if solve:

    elemType = "TRI3" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]

    interfaceGmsh = Interface_Gmsh(affichageGmsh=False)

    domain = Domain(Point(), Point(x=L, y=L), taille=taille)
    line = Line(Point(y=L/2, isOpen=True), Point(x=L/2, y=L/2), taille=taille)

    mesh = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType,
    isOrganised=True, openCrack=openCrack)

    Affichage.Plot_Maillage(mesh)
    # plt.show()

    # Récupère les noeuds qui m'interessent
    noeuds_Milieu = mesh.Nodes_Line(line)
    noeuds_Haut = mesh.Nodes_Conditions(conditionY=lambda y: y == L)
    noeuds_Bas = mesh.Nodes_Conditions(conditionY=lambda y: y == 0)
    noeuds_Gauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
    noeuds_Droite = mesh.Nodes_Conditions(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

    NoeudsBord=[]
    for noeuds in [noeuds_Bas,noeuds_Droite,noeuds_Haut]:
        NoeudsBord.extend(noeuds)

    if simulation == "Shear":
        ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["x"])
    else:
        ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["y"])

    # Simulation  -------------------------------------------------------------------------------------------
    

    if comportement == "Elas_Isot":
        comportement = Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)
    else:
        # comportement = Elas_IsotTrans(2, El=210e9, Et=20e9, Gl=)
        raise "Pas implémenté pour le moment"

    phaseFieldModel = PhaseFieldModel(comportement, split, regularisation, Gc=Gc, l_0=l0, solveur=solveur)

    materiau = Materiau(phaseFieldModel, ro=1)

    simu = Simu(mesh, materiau, verbosity=False, useNumba=False)

    # Renseignement des conditions limites
    def Chargement(dep):

        simu.Init_Bc()

        if not openCrack:
            simu.add_dirichlet("damage",noeuds_Milieu, [1], ["d"])
        
        if simulation == "Shear":
            # Conditions en déplacements à Gauche et droite
            simu.add_dirichlet("displacement", noeuds_Gauche, [0],["y"])
            simu.add_dirichlet("displacement", noeuds_Droite, [0],["y"])
            simu.add_dirichlet("displacement", noeuds_Haut, [dep,0], ["x","y"])
            # simu.add_dirichlet("displacement", noeuds_Haut, [dep], ["x"])

            # Conditions en déplacements en Bas
            simu.add_dirichlet("displacement", noeuds_Bas, [0,0],["x","y"])
        else:
            simu.add_dirichlet("displacement", noeuds_Haut, [0,dep], ["x","y"])
            simu.add_dirichlet("displacement", noeuds_Bas, [0],["y"])

            
        
    Chargement(0)

    Affichage.NouvelleSection("Simulations")

    if simulation == "Shear":
        if test:
            u_inc = 5e-8
            N = 400
            # N = 10
            # u_inc = 5e-7
        else:
            u_inc = 1e-8
            N = 1500

        chargement = np.linspace(u_inc, u_inc*N, N, endpoint=True)
        
        listInc = [u_inc]
        listThreshold = [chargement[-1]]

    elif simulation == "Tension":
        if test:
            u0 = 1e-7;  N0 = 40
            u1 = 1e-8;  N1 = 400
        else:
            u0 = 1e-8;  N0 = 400
            u1 = 1e-9;  N1 = 4000

        chargement = np.linspace(u0, u0*N0, N0, endpoint=True)
        chargement = np.append(chargement, np.linspace(u1, u1*N1, N1, endpoint=True)+chargement[-1])
        
        listInc = [u0, u1]
        listThreshold = [chargement[N0], chargement[N1]]

    PhaseFieldSimulation.ResumeChargement(simu, chargement[-1],listInc, listThreshold, "displacement")

    damage_t=[]
    uglob_t=[]

    dep = 0

    tic = Tic()

    bord = 0

    
    list_Psi_Elas=[]
    list_Psi_Crack=[]
    deplacements=[]
    forces=[]

    N = chargement.shape[0]
    for iter, dep in enumerate(chargement):

        tic = Tic()

        iterConv=0
        convergence = False
        damage = simu.damage

        Chargement(dep)

        u, d, Kglob, iterConv, dincMax = PhaseFieldSimulation.ResolutionIteration(simu=simu, tolConv=tolConv, maxIter=maxIter)

        if iterConv == maxIter:
            print(f'On converge pas apres {iterConv} itérations')
            break

        simu.Save_Iteration()

        temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
        f = np.sum(np.einsum('ij,j->i', Kglob[ddls_Haut, :].toarray(), u, optimize='optimal'))

        PhaseFieldSimulation.ResumeIteration(simu, iter, dep*1e6, d, iterConv, dincMax, temps, "µm", iter/N, True)
  
        deplacements.append(dep)
        forces.append(f)

        if np.any(damage[NoeudsBord] >= 0.8):                                
            bord +=1

        if bord == 5:
            break
            
    # Sauvegarde
    print()
    PostTraitement.Save_Load_Displacement(forces, deplacements, folder)
    PostTraitement.Save_Simu(simu, folder)
    

    forces = np.array(forces)
    deplacements = np.array(deplacements)

else:   

    simu = PostTraitement.Load_Simu(folder)

    forces, deplacements = PostTraitement.Load_Load_Displacement(folder)

    # Affichage.Plot_Maillage(simu.mesh)
    # plt.show()
        


# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Affichage")

if makeMovie:
    PostTraitement.MakeMovie(folder, "damage", simu)
    # PostTraitement.MakeMovie(folder, "Svm", simu)        
    # PostTraitement.MakeMovie(filename, "Syy", simu, valeursAuxNoeuds=True, deformation=True)

if plotResult:

    Affichage.Plot_BoundaryConditions(simu)

    Affichage.Plot_ForceDep(deplacements*1e6, forces*1e-3, 'ud en µm', 'f en kN', folder)

    Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True,
    affichageMaillage=False, deformation=False, folder=folder)
    

    # Affichage.Plot_Result(simu, "dy", folder=folder, deformation=True)
        
if saveParaview:
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)
        



Tic.getResume()

if solve:
    Tic.getGraphs(folder, False)

plt.show()


# %%
