import os
import matplotlib.pyplot as plt

import Folder
import PostProcessing
import Display
from Geom import *
import Materials
from Interface_Gmsh import Interface_Gmsh, ElemType
import Simulations
from TicTac import Tic

Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
dim = 2
folder = Folder.New_File(f"Dynamics{dim}D", results=True)
plotResult = True

isLoading = True
initSimu = True

makeParaview = False; NParaview = 500
makeMovie = False; NMovie = 400
plotIter = True; resultToPlot = "uy"

# Dumping
# coefM = 1e-2
# coefK = 1e-2*2
coefM = 1e-3
coefK = 1e-3

tic_Tot = Tic()

# geom
L = 120;  #mm
h = 13
b = 13

# ----------------------------------------------
# Meshing
# ----------------------------------------------
# meshSize = h/1
# meshSize = L/2
meshSize = h/5

interfaceGmsh = Interface_Gmsh(False)
if dim == 2:
    elemType = ElemType.QUAD4 # TRI3, TRI6, TRI10, TRI15, QUAD4, QUAD8
    domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), meshSize)
    Line0 = Line(Point(y=-h/2), Point(y=h/2))
    LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
    LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
    circle = Circle(Point(x=L/2, y=0), h*0.2, isHollow=False)

    mesh = interfaceGmsh.Mesh_2D(domain, elemType=elemType, isOrganised=True)
    area = mesh.area - L*h
elif dim == 3:
    elemType = ElemType.HEXA8 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15
    domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=meshSize)
    mesh = interfaceGmsh.Mesh_3D(domain, [], [0,0,b], elemType=elemType, nLayers=3)

    volume = mesh.volume - L*b*h
    area = mesh.area - (L*h*4 + 2*b*h)

Display.Plot_Mesh(mesh)

nodes_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
nodes_L = mesh.Nodes_Conditions(lambda x,y,z: x == L)
nodes_h = mesh.Nodes_Conditions(lambda x,y,z: y == h/2)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
material = Materials.Elas_Isot(dim, thickness=b)
simu = Simulations.Simu_Displacement(mesh, material, useNumba=True, verbosity=False)
simu.rho = 8100*1e-9
simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

def Loading(isLoading: bool):

    simu.Bc_Init()

    # Boundary conditions
    if dim == 2:
        simu.add_dirichlet(nodes_0, [0, 0], ["x","y"], description="Fixed")
    elif dim == 3:
        simu.add_dirichlet(nodes_0, [0, 0, 0], ["x","y","z"], description="Fixed")
    if isLoading:
        simu.add_dirichlet(nodes_L, [-7], ["y"], description="dep")        

def Iteration(steadyState: bool, isLoading: bool):

    Loading(isLoading)    

    if steadyState:
        simu.Solver_Set_Elliptic_Algorithm()
    else:
        simu.Solver_Set_Newton_Raphson_Algorithm(dt=dt)

    simu.Solve()
    
    simu.Save_Iter()

# first iteration, then drop the beam
Iteration(steadyState=True, isLoading=True)    

if plotIter:
    fig, ax, cb = Display.Plot_Result(simu, resultToPlot, nodeValues=True, plotMesh=True, deformation=True)

Tmax = 0.5
N = 100
dt = Tmax/N
t = 0

while t <= Tmax:

    Iteration(steadyState=False, isLoading=False)

    if plotIter:
        cb.remove()
        fig, ax, cb = Display.Plot_Result(simu, resultToPlot, nodeValues=True, plotMesh=True, ax=ax, deformation=True)
        plt.pause(1e-12)

    t += dt

    print(f"{t:.3f} s", end='\r')

tic_Tot.Tac("Time","total time", True)        

# ----------------------------------------------
# Post processing
# ----------------------------------------------
Display.Section("Post processing")

Display.Plot_BoundaryConditions(simu)

# folder=""

if makeParaview:
    PostProcessing.Make_Paraview(folder, simu,Niter=NParaview)

if makeMovie:
    PostProcessing.Make_Movie(folder, resultToPlot, simu, plotMesh=True, Niter=NMovie, deformation=True, nodeValues=True, factorDef=1)

if plotResult:

    tic = Tic()
    print(simu)
    # Display.Plot_Result(simu, "amplitude")
    # Display.Plot_Mesh(simu, deformation=True, folder=folder)
    Display.Plot_Result(simu, "uy", deformation=True, nodeValues=False)        
    Display.Plot_Result(simu, "Svm", deformation=False, plotMesh=False, nodeValues=False)
    # Display.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)
    
    tic.Tac("Display","Results", plotResult)

Tic.Plot_History(details=True)
plt.show()