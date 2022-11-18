import matplotlib.pyplot as plt
import Affichage
import PostTraitement
import Dossier
import Interface_Gmsh
from Geom import Circle, Domain, Line, Point
import Materials
import Simulations
import numpy as np

Affichage.Clear()

plotIter = True; affichageIter = "thermal"

pltMovie = True; NMovie = 300

folder = Dossier.NewFile(filename="Thermal", results=True)

dim = 2

a = 1
if dim == 2:
    domain = Domain(Point(), Point(a, a), a/20)
else:
    domain = Domain(Point(), Point(a, a), a/20)
circle = Circle(Point(a/2, a/2), diam=a/3, isCreux=True, taille=a/50)
interfaceGmsh = Interface_Gmsh.Interface_Gmsh(False, False, True)

if dim == 2:
    # mesh = interfaceGmsh.Rectangle_2D(domain, "QUAD4")
    mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI6")
else:
    mesh = interfaceGmsh.Mesh_PlaqueAvecCercle3D(domain, circle, [0,0,a], 4, elemType="HEXA8")

thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, epaisseur=1)

materiau = Materials.Create_Materiau(thermalModel, verbosity=False)

simu = Simulations.Create_Simu(mesh , materiau, False)

noeuds0 = mesh.Nodes_Conditions(lambda x: x == 0)
noeudsL = mesh.Nodes_Conditions(lambda x: x == a)

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

    thermal = simu.Solve(steadyState)

    simu.Save_Iteration()

    return thermal

Tmax = 60*8 #s
N = 10
dt = Tmax/N #s
t=0

simu.Solveur_Parabolic_Properties(alpha=0.5, dt=dt)

if Tmax == 0:
    steadyState=True
    plotIter = False
else:
    steadyState=False

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True)

print()

while t < Tmax:

    thermal = Iteration(False)

    t += dt

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, affichageIter, valeursAuxNoeuds=True, affichageMaillage=True, ax=ax)
        plt.pause(1e-12)

    print(f"{np.round(t)} s",end='\r')
    

# Affichage.Plot_NoeudsMaillage(mesh, noeuds=noeudsCircle)
# Affichage.Plot_ElementsMaillage(mesh, noeuds=noeudsCircle, dimElem=3)
Affichage.Plot_Result(simu, "thermal", affichageMaillage=True, valeursAuxNoeuds=True)



if dim == 3:
    print(f"Volume : {mesh.volume:.3}")

PostTraitement.Make_Paraview(folder, simu)

if pltMovie:
    PostTraitement.Make_Movie(folder, affichageIter, simu, NMovie)

print(thermal.min())

plt.show()

pass