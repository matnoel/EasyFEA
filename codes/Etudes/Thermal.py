import matplotlib.pyplot as plt
import Affichage
import PostTraitement
import Folder
import Interface_Gmsh
from Geom import Circle, Domain, Line, Point
import Materials
import Simulations
import numpy as np

Affichage.Clear()

plotIter = True; affichageIter = "thermal"

pltMovie = False; NMovie = 300

folder = Folder.New_File(filename="Thermal", results=True)

dim = 3

a = 1
if dim == 2:
    domain = Domain(Point(), Point(a, a), a/20)
else:
    domain = Domain(Point(), Point(a, a), a/20)
circle = Circle(Point(a/2, a/2), diam=a/3, isCreux=True, meshSize=a/50)
interfaceGmsh = Interface_Gmsh.Interface_Gmsh(False, False, True)

if dim == 2:
    # mesh = interfaceGmsh.Mesh_2D(domain, [], "QUAD4")
    mesh = interfaceGmsh.Mesh_2D(domain, [circle], "QUAD4")
else:
    mesh = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,a], 4, "PRISM6")

thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, epaisseur=1)

simu = Simulations.Simu_Thermal(mesh, thermalModel, False)
simu.rho = 1

noeuds0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
noeudsL = mesh.Nodes_Conditions(lambda x,y,z: x == a)

if dim == 2:
    noeudsCircle = mesh.Nodes_Circle(circle)
else:
    noeudsCircle = mesh.Nodes_Cylindre(circle, [0,0,a])

def Iteration(steadyState: bool):

    simu.Bc_Init()

    simu.add_dirichlet(noeuds0, [0], [""])
    simu.add_dirichlet(noeudsL, [40], [""])

    # simu.add_dirichlet(noeudsCircle, [10], [""])
    # simu.add_dirichlet(noeudsCircle, [10], [""])

    # simu.add_volumeLoad(noeudsCircle, [100], [""])

    thermal = simu.Solve()

    simu.Save_Iteration()

    return thermal

Tmax = 0.5 #s
N = 50
dt = Tmax/N #s
t=0

simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

if Tmax == 0:
    steadyState=True
    plotIter = False
else:
    steadyState=False

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, nodeValues=True, plotMesh=True)

print()

while t < Tmax:

    thermal = Iteration(False)

    t += dt

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, nodeValues=True, plotMesh=True, ax=ax)
        plt.pause(1e-12)

    print(f"{np.round(t)} s",end='\r')
    

# Affichage.Plot_NoeudsMaillage(mesh, noeuds=noeudsCircle)
# Affichage.Plot_ElementsMaillage(mesh, noeuds=noeudsCircle, dimElem=3)
Affichage.Plot_Result(simu, "thermal", plotMesh=True, nodeValues=True)



if dim == 3:
    print(f"Volume : {mesh.volume:.3}")

PostTraitement.Make_Paraview(folder, simu)

if pltMovie:
    PostTraitement.Make_Movie(folder, affichageIter, simu, NMovie)

print(thermal.min())

plt.show()

pass