# %%
import os

from BoundaryCondition import BoundaryCondition

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

simulation = "Shear" #"Shear" , "Tension"
folder = '_'.join(["Benchmarck",simulation])
folder = Dossier.NewFile(folder, results=True)

test = True
solve = True
plotResult = True
saveParaview = False
makeMovie = False

# Data --------------------------------------------------------------------------------------------

comportement = "Elas_Isot" # "Elas_Isot"
split = "AnisotMiehe" # "Bourdin","Amor","Miehe","Stress"
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
        # taille = 0.001
        # taille *= 1.5
else:
        taille = l0/2 #l0/2 2.5e-6

if test:
    folder = Dossier.Join([folder, "Test", nameSimu])
else:
    folder = Dossier.Join([folder, nameSimu])

print(folder)

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

        domain = Domain(Point(), Point(x=L, y=L), taille=taille)
        line = Line(Point(y=L/2, isOpen=True), Point(x=L/2, y=L/2), taille=taille)

        mesh = interfaceGmsh.RectangleAvecFissure(domain=domain, crack=line, elemType=elemType,
        isOrganised=True, openCrack=openCrack)

        # Récupère les noeuds qui m'interessent
        noeuds_Milieu = mesh.Get_Nodes_Line(line)
        noeuds_Haut = mesh.Get_Nodes_Conditions(conditionY=lambda y: y == L)
        noeuds_Bas = mesh.Get_Nodes_Conditions(conditionY=lambda y: y == 0)
        noeuds_Gauche = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
        noeuds_Droite = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

        NoeudsBord=[]
        for noeuds in [noeuds_Bas,noeuds_Droite,noeuds_Haut]:
                NoeudsBord.extend(noeuds)

        ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["x"])

        # Simulation  -------------------------------------------------------------------------------------------
        

        comportement = Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)

        phaseFieldModel = PhaseFieldModel(comportement, split, regularisation, Gc=Gc, l_0=l0)

        materiau = Materiau(comportement, ro=1, phaseFieldModel=phaseFieldModel)

        simu = Simu(mesh, materiau, verbosity=False)

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
                else:
                        simu.add_dirichlet("displacement", noeuds_Haut, [dep], ["y"])

                # Conditions en déplacements en Bas
                simu.add_dirichlet("displacement", noeuds_Bas, [0,0],["x","y"])
           
        Chargement(0)

        Affichage.NouvelleSection("Simulations")

        N = 400
        # N = 10

        damage_t=[]
        uglob_t=[]

        u_inc = 5e-8
        # u_inc = 5e-7
        dep = 0

        tic = TicTac()

        bord = 0

        
        list_Psi_Elas=[]
        list_Psi_Crack=[]
        deplacements=[]
        forces=[]

        maxIter = 250
        # tolConv = 0.0025
        # tolConv = 0.005
        tolConv = 0.01

        for iter in range(N):

                tic = TicTac()

                dep += u_inc

                iterConv=0
                convergence = False
                dold = simu.damage

                Chargement(dep)

                while not convergence:
                
                        iterConv += 1            

                        # Damage
                        simu.Assemblage_d()
                        damage = simu.Solve_d()

                        # Displacement
                        Kglob = simu.Assemblage_u()                        
                        displacement = simu.Solve_u()

                        dincMax = np.max(np.abs(damage-dold))
                        # TODO faire en relatif np.max(np.abs((damage-dold)/dold))?
                        convergence = dincMax <= tolConv
                        # if damage.min()>1e-5:
                        #     convergence=False
                        dold = damage.copy()

                        if iterConv == maxIter:
                                break

                        convergence=True
                
                # TODO Comparer avec code matlab

                if iterConv == maxIter:
                        print(f'On converge pas apres {iterConv} itérations')
                        break

                simu.Save_Iteration()

                temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
                temps = np.round(temps,3)
                max_d = damage.max()
                min_d = damage.min()
                f = np.sum(np.einsum('ij,j->i', Kglob[ddls_Haut, :].toarray(), displacement, optimize=True))

                print(f"{iter+1:4}/{N} : ud = {np.round(dep*1e6,3)} µm,  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{temps} s")

                # # Affiche dans la console
                # min = np.round(np.min(damage),3)
                # max = np.round(np.max(damage),3)
                # norm = np.sum(damage)
                # tResolution = tic.Tac("Resolution PhaseField", "Resolution Phase field", False)
                # # print(iter+1," : max d = {:.5f}, time = {:.3f}".format(norm, tResolution))
                # print(f'{iter+1}/{N} : max d = {max}, min d = {min}, time = {np.round(tResolution,3)} s') 
                
                deplacements.append(dep)
                forces.append(f)

                if np.any(damage[NoeudsBord] >= 0.8):                                
                        bord +=1

                if bord == 5:
                        break
                
        # Sauvegarde
        PostTraitement.Save_Simu(simu, folder)
        
        PostTraitement.Save_Load_Displacement(forces, deplacements, folder)

        forces = np.array(forces)
        deplacements = np.array(deplacements)

else:   

        simu = PostTraitement.Load_Simu(folder)

        forces, deplacements = PostTraitement.Load_Load_Displacement(folder)
        


# ------------------------------------------------------------------------------------------------------
Affichage.NouvelleSection("Affichage")

if makeMovie:
        # PostTraitement.MakeMovie(folder, "damage", simu)
        PostTraitement.MakeMovie(folder, "Svm", simu)        
        # PostTraitement.MakeMovie(filename, "Syy", simu, valeursAuxNoeuds=True, deformation=True)

if plotResult:

        Affichage.Plot_BoundaryConditions(simu)

        fig, ax = plt.subplots()
        ax.plot(deplacements*1e6, np.abs(forces)/1e3, c='blue')
        ax.set_xlabel("ud en µm")
        ax.set_ylabel("f en kN")
        ax.grid()
        PostTraitement.Save_fig(folder, "forcedep")


        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True,
        affichageMaillage=False, deformation=False, folder=folder)
        

        Affichage.Plot_Result(simu, "dy", folder=folder, deformation=True)
        
if saveParaview:
        PostTraitement.Save_Simulation_in_Paraview(folder, simu)
        



TicTac.getResume()

plt.show()


# %%
