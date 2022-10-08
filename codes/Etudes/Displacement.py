# %%
# import sys
# sys.path.append("/home/matthieu/Documents/PythonEF/classes")
import os

import Dossier
import PostTraitement
import Affichage
from Geom import *
from Materials import Elas_Isot, Materiau
from Interface_Gmsh import Interface_Gmsh
from Simu import Simu
from TicTac import Tic

import matplotlib.pyplot as plt


Affichage.Clear()

folder = Dossier.NewFile("Etude2D", results=True)

ticTot = Tic()

# Data --------------------------------------------------------------------------------------------

plotResult = True

saveParaview = True; NParaview = 500

useNumba = True

isLoading = False
initSimu = True

plotIter = True; affichageIter = "Svm"

coefM = 1/4
coefK = 1e-4

Tmax = 2
N = 400
dt = Tmax/N
t = 0

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13
b = 13

P = 800 #N
lineLoad = P/h #N/mm
surfLoad = P/h/b #N/mm2

# Paramètres maillage
taille = h/6
# taille = h/200

comportement = Elas_Isot(dim, epaisseur=b)

# Materiau
materiau = Materiau(comportement, ro=8100*1e-9)
# Construction du modele et du maillage --------------------------------------------------------------------------------


interfaceGmsh = Interface_Gmsh()
if dim == 2:
    domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), taille)
    Line0 = Line(Point(y=-h/2), Point(y=h/2))
    LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
    LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
    circle = Circle(Point(x=L/2, y=0), h*0.2, isCreux=False)
    
    elemType = "QUAD4" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]

    mesh = interfaceGmsh.Rectangle_2D(domain=domain, elemType=elemType, isOrganised=True)
    # mesh = interfaceGmsh.PlaqueAvecCercle(domain=domain, circle=circle, isOrganised=False)
    aire = mesh.aire
elif dim == 3:
    # # Sans importation
    domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
    # circle = Circle(Point(x=L/2, y=0), h*0.8, taille=taille, isCreux=False)
    # mesh = interfaceGmsh.PlaqueAvecCercle3D(domain,circle ,[0,0,b], elemType="HEXA8", isOrganised=False, nCouches=3)
    
    elemType = "TETRA4" # "TETRA4", "HEXA8", "PRISM6"
    mesh = interfaceGmsh.Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=True, nCouches=3)

    volume = mesh.volume - L*b*h
    aire = mesh.aire - (L*h*4 + 2*b*h)

# Affichage.Plot_Maillage(mesh)
# # plt.show()

noeuds_en_0 = mesh.Nodes_Conditions(conditionX=lambda x: x == 0) # noeuds_en_0 = mesh.Nodes_Line(Line0)
noeuds_en_L = mesh.Nodes_Conditions(conditionX=lambda x: x == L) # noeuds_en_L = mesh.Nodes_Line(LineL)

# Affichage.Plot_Maillage(mesh)
# plt.show()

noeuds_en_h = mesh.Nodes_Conditions(conditionY=lambda y: y == h/2) # noeuds_en_h= mesh.Nodes_Line(LineH)

# ------------------------------------------------------------------------------------------------------

simu = Simu(mesh, materiau, useNumba=useNumba, verbosity=False)
simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

def Chargement(isLoading: bool):

    simu.Init_Bc()

    # Renseigne les condtions limites

    if dim == 2:
        simu.add_dirichlet("displacement", noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
    elif dim == 3:
        simu.add_dirichlet("displacement", noeuds_en_0, [0, 0, 0], ["x","y","z"], description="Encastrement")
    
    # simu.add_volumeLoad("displacement", mesh.Nodes_Conditions(conditionX=lambda x: x>0), [-9.81*1e-3], ["y"])

    if isLoading:
        # simu.add_dirichlet("displacement", noeuds_en_h, [lambda x,y,z : -x/L], ["y"], description="f(x)=x/L")

        # simu.add_lineLoad("displacement", noeuds_en_h, [lambda x,y,z : -surfLoad], ["y"], description="Encastrement")
        simu.add_dirichlet("displacement", noeuds_en_L, [-7], ["y"], description="dep")

        # simu.add_surfLoad("displacement",noeuds_en_L, [-surfLoad], ["y"])
        # simu.add_lineLoad("displacement",noeuds_en_L, [-lineLoad], ["y"])
        pass


def Iteration(steadyState: bool, isLoading: bool):

    Chargement(isLoading)

    # Assemblage du système matricielle
    Kglob = simu.Assemblage_u(steadyState)
    # plt.figure()
    # plt.spy(Kglob)

    dep = simu.Solve_u(steadyState)
    
    simu.Save_Iteration()

if initSimu:
    # Init
    Iteration(steadyState=True, isLoading=True)




if N > 1:
    steadyState=False
else:
    steadyState=True

simu.Set_Hyperbolic_AlgoProperties(dt=dt)

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True, deformation=True)

while t <= Tmax:

    Iteration(steadyState=steadyState, isLoading=isLoading)

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True, oldfig=fig, oldax=ax, deformation=True)
        plt.pause(1e-12)

    t += dt

    print(f"{int(t/dt)}", end='\r')

# PostTraitement.Save_Simu(simu, folder)

ticTot.Tac("Temps script","Temps total", True)        

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")

simu.ResumeResultats()

Affichage.Plot_BoundaryConditions(simu)
# plt.show()

# folder=""

if saveParaview:        
    filename = Dossier.NewFile(os.path.join("Etude2D","solution2D"), results=True)
    PostTraitement.Save_Simulation_in_Paraview(folder, simu,Niter=NParaview)

if plotResult:

    tic = Tic()
    # Affichage.Plot_Result(simu, "amplitude")
    Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Affichage.Plot_Result(simu, "dy", deformation=True, valeursAuxNoeuds=False)        
    Affichage.Plot_Result(simu, "Svm", deformation=True, valeursAuxNoeuds=True, affichageMaillage=True)        
    # Affichage.Plot_Result(simu, "Svm", deformation=True, valeursAuxNoeuds=False, affichageMaillage=False, folder=folder)        
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

if plotResult:
    # Tic.getGraphs(details=False)
    plt.show()

# %%
