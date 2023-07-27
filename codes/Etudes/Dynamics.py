import os
import matplotlib.pyplot as plt

import Folder
import PostTraitement
import Display
from Geom import *
import Materials
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

# Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

dim = 3
folder = Folder.New_File(f"Dynamics{dim}D", results=True)
plotResult = True

isLoading = False
initSimu = True

saveParaview = False; NParaview = 500
pltMovie = True; NMovie = 400
plotIter = True; affichageIter = "uy"

coefM = 1e-2
coefK = 1e-2*2

# coefM = 0
# coefK = 0

tic_Tot = Tic()

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
taille = h/5


# ----------------------------------------------
# Maillage
# ----------------------------------------------

interfaceGmsh = Interface_Gmsh(False)
if dim == 2:
    domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), taille)
    Line0 = Line(Point(y=-h/2), Point(y=h/2))
    LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
    LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
    circle = Circle(Point(x=L/2, y=0), h*0.2, isCreux=False)
    
    elemType = "QUAD4" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]

    mesh = interfaceGmsh.Mesh_2D(domain, elemType=elemType, isOrganised=True)
    aire = mesh.area - L*h
elif dim == 3:
    # # Sans importation
    domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=taille)
    
    elemType = "HEXA8" # "TETRA4", "TETRA10", "HEXA8", "PRISM6"
    mesh = interfaceGmsh.Mesh_3D(domain, [], [0,0,b], elemType=elemType, nCouches=3)

    volume = mesh.volume - L*b*h
    aire = mesh.area - (L*h*4 + 2*b*h)

Display.Plot_Mesh(mesh)

noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0) # noeuds_en_0 = mesh.Nodes_Line(Line0)
noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == L) # noeuds_en_L = mesh.Nodes_Line(LineL)

# Affichage.Plot_Maillage(mesh)
# plt.show()

noeuds_en_h = mesh.Nodes_Conditions(lambda x,y,z: y == h/2) # noeuds_en_h= mesh.Nodes_Line(LineH)

# ----------------------------------------------
# Comportement et Simu
# ----------------------------------------------

comportement = Materials.Elas_Isot(dim, epaisseur=b)
simu = Simulations.Simu_Displacement(mesh, comportement, useNumba=True, verbosity=False)
simu.rho = 8100*1e-9
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
        simu.add_dirichlet(noeuds_en_L, [-7], ["y"], description="dep")

        # simu.add_surfLoad(noeuds_en_L, [-surfLoad], ["y"])
        # simu.add_surfLoad(noeuds_en_L, [-surfLoad*(t/Tmax)], ["y"])
        # simu.add_lineLoad(noeuds_en_L, [-lineLoad], ["y"])        


def Iteration(steadyState: bool, isLoading: bool):

    Chargement(isLoading)    

    if steadyState:
        simu.Solver_Set_Elliptic_Algorithm()
    else:
        simu.Solver_Set_Newton_Raphson_Algorithm(dt=dt)

    simu.Solve()
    
    simu.Save_Iteration()

if initSimu:
    # Init
    Iteration(steadyState=True, isLoading=True)

if N > 1:
    steadyState=False
else:
    steadyState=True

if plotIter:
    fig, ax, cb = Display.Plot_Result(simu, affichageIter, nodeValues=True, plotMesh=True, deformation=True)

while t <= Tmax:

    Iteration(steadyState=steadyState, isLoading=isLoading)

    if plotIter:
        cb.remove()
        fig, ax, cb = Display.Plot_Result(simu, affichageIter, nodeValues=True, plotMesh=True, ax=ax, deformation=True)
        plt.pause(1e-12)

    t += dt

    print(f"{np.round(t,3)} s", end='\r')

# PostTraitement.Save_Simu(simu, folder)

tic_Tot.Tac("Temps script","Temps total", True)        

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
Display.Section("Post traitement")

simu.Resultats_Get_Resume_Iteration()

Display.Plot_BoundaryConditions(simu)
# plt.show()

# folder=""

if saveParaview:
    PostTraitement.Make_Paraview(folder, simu,Niter=NParaview)

if pltMovie:
    PostTraitement.Make_Movie(folder, "Svm", simu, plotMesh=True, Niter=NMovie, deformation=True, nodeValues=True)

if plotResult:

    tic = Tic()
    simu.Resultats_Resume(True)
    # Affichage.Plot_Result(simu, "amplitude")
    # Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Display.Plot_Result(simu, "uy", deformation=True, nodeValues=False)        
    Display.Plot_Result(simu, "Svm", deformation=False, plotMesh=False, nodeValues=False)
    # Affichage.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

# tic_Tot.Resume()
Tic.Plot_History(folder ,details=True)
plt.show()