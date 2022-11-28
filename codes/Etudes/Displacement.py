# %%
# import sys
# sys.path.append("/home/matthieu/Documents/PythonEF/classes")
import os

import Folder
import PostTraitement
import Affichage
from Geom import *
import Materials
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

import matplotlib.pyplot as plt

Affichage.Clear()

dim = 2

folder = Folder.New_File(f"Etude{dim}D", results=True)

tic_Tot = Tic()

# Data --------------------------------------------------------------------------------------------

plotResult = True

saveParaview = False; NParaview = 500

useNumba = True

isLoading = False
initSimu = True

pltMovie = False; NMovie = 400

plotIter = True; affichageIter = "dy"

coefM = 1e-2
coefK = 1e-3*2

# coefM = 0
# coefK = 0

Tmax = 0.5
N = 100
dt = Tmax/N
t = 0

# Paramètres géométrie
L = 120;  #mm
h = 13
b = 13

P = 800 #N
lineLoad = P/h #N/mm
surfLoad = P/h/b #N/mm2

# Paramètres maillage
# taille = h/1
# taille = L/2
taille = h/10

comportement = Materials.Elas_Isot(dim, epaisseur=b)

# Materiau
materiau = Materials.Create_Materiau(comportement, ro=8100*1e-9)
# Construction du modele et du maillage --------------------------------------------------------------------------------


interfaceGmsh = Interface_Gmsh(False)
if dim == 2:
    domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), taille)
    Line0 = Line(Point(y=-h/2), Point(y=h/2))
    LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
    LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
    circle = Circle(Point(x=L/2, y=0), h*0.2, isCreux=False)
    
    elemType = "TRI3" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]

    mesh = interfaceGmsh.Mesh_Rectangle_2D(domain=domain, elemType=elemType, isOrganised=True)
    # mesh = interfaceGmsh.PlaqueAvecCercle(domain=domain, circle=circle, isOrganised=False)
    aire = mesh.aire - L*h
elif dim == 3:
    # # Sans importation
    domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
    # circle = Circle(Point(x=L/2, y=0), h*0.8, taille=taille, isCreux=False)
    # mesh = interfaceGmsh.PlaqueAvecCercle3D(domain,circle ,[0,0,b], elemType="HEXA8", isOrganised=False, nCouches=3)
    
    elemType = "HEXA8" # "TETRA4", "HEXA8", "PRISM6"
    mesh = interfaceGmsh.Mesh_Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=False, nCouches=3)

    volume = mesh.volume - L*b*h
    aire = mesh.aire - (L*h*4 + 2*b*h)

Affichage.Plot_Maillage(mesh)
# Affichage.Plot_NoeudsMaillage(mesh,showId=True)
# plt.show()

noeuds_en_0 = mesh.Nodes_Conditions(conditionX=lambda x: x == 0) # noeuds_en_0 = mesh.Nodes_Line(Line0)
noeuds_en_L = mesh.Nodes_Conditions(conditionX=lambda x: x == L) # noeuds_en_L = mesh.Nodes_Line(LineL)

# Affichage.Plot_Maillage(mesh)
# plt.show()

noeuds_en_h = mesh.Nodes_Conditions(conditionY=lambda y: y == h/2) # noeuds_en_h= mesh.Nodes_Line(LineH)

# ------------------------------------------------------------------------------------------------------

simu = Simulations.Create_Simu(mesh, materiau, useNumba=useNumba, verbosity=False)
simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

def Chargement(isLoading: bool):

    simu.Bc_Init()

    # Renseigne les condtions limites

    if dim == 2:
        simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
    elif dim == 3:
        simu.add_dirichlet(noeuds_en_0, [0, 0, 0], ["x","y","z"], description="Encastrement")
    
    # simu.add_volumeLoad(mesh.Nodes_Conditions(conditionX=lambda x: x>1e-4), [-9.81*1e-3], ["y"])

    if isLoading:
        # simu.add_dirichlet(noeuds_en_h, [lambda x,y,z : -x/L], ["y"], description="f(x)=x/L")

        # simu.add_lineLoad(noeuds_en_h, [lambda x,y,z : -surfLoad], ["y"], description="Encastrement")
        # simu.add_dirichlet(noeuds_en_L, [-7], ["y"], description="dep")

        simu.add_surfLoad(noeuds_en_L, [-surfLoad], ["y"])
        # simu.add_surfLoad(noeuds_en_L, [-surfLoad*(t/Tmax)], ["y"])
        # simu.add_lineLoad(noeuds_en_L, [-lineLoad], ["y"])
        pass


def Iteration(steadyState: bool, isLoading: bool):

    Chargement(isLoading)

    # Assemblage du système matricielle
    Kglob = simu.Get_K_C_M_F(simu.problemType)[0]
    # plt.figure()
    # plt.spy(Kglob)

    if steadyState:
        simu.Solveur_Set_Elliptic_Algorithm()
    else:
        simu.Solveur_Set_Newton_Raphson_Algorithm(dt=dt)

    dep = simu.Solve()
    
    simu.Save_Iteration()

if initSimu:
    # Init
    Iteration(steadyState=True, isLoading=True)

if N > 1:
    steadyState=False
else:
    steadyState=True

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True, deformation=True)

while t <= Tmax:

    Iteration(steadyState=steadyState, isLoading=isLoading)

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True, ax=ax, deformation=True)
        plt.pause(1e-12)

    t += dt

    print(f"{np.round(t,3)} s", end='\r')

# PostTraitement.Save_Simu(simu, folder)

tic_Tot.Tac("Temps script","Temps total", True)        

# Post traitement --------------------------------------------------------------------------------------
Affichage.NouvelleSection("Post traitement")

simu.Resultats_Get_Resume_Iteration()

Affichage.Plot_BoundaryConditions(simu)
# plt.show()

# folder=""

if saveParaview:        
    filename = Folder.New_File(os.path.join("Etude2D","solution2D"), results=True)
    PostTraitement.Make_Paraview(folder, simu,Niter=NParaview)

if pltMovie:
    PostTraitement.Make_Movie(folder, "Svm", simu, affichageMaillage=True, Niter=NMovie, deformation=True, valeursAuxNoeuds=True)

if plotResult:

    tic = Tic()
    # Affichage.Plot_Result(simu, "amplitude")
    Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Affichage.Plot_Result(simu, "dy", deformation=True, valeursAuxNoeuds=False)        
    Affichage.Plot_Result(simu, "Svm", deformation=True, affichageMaillage=True, valeursAuxNoeuds=False)        
    # Affichage.Plot_Result(simu, "Svm", deformation=True, valeursAuxNoeuds=False, affichageMaillage=False, folder=folder)        
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

if plotResult:
    Tic.getGraphs(details=True)
    plt.show()

# %%
