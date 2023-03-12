import matplotlib.pyplot as plt
import numpy as np

import Simulations
import Materials
from Geom import Domain, Point, Circle
from Interface_Gmsh import Interface_Gmsh
import Affichage
import PostTraitement
import Folder
import TicTac

Affichage.Clear()

plotModel = False
plotIter = True
resultat = "amplitudeSpeed"

makeMovie = False

tMax = 1e-7
Nt = 80
dt = tMax/Nt
load = 1e-3

f0=2
a0=1

t0 = dt*2

a = 1
meshSize = a/100
diam = a/10
r = diam/2

domain = Domain(Point(x=-a/2, y=-a/2), Point(x=a/2, y=a/2), meshSize)
circle = Circle(Point(), diam, meshSize, isCreux=False)


interfaceGmsh = Interface_Gmsh(False)
mesh = interfaceGmsh.Mesh_Domain_Circle_2D(domain, circle, "TRI3")

if plotModel:
    Affichage.Plot_Model(mesh)
    plt.show()

noeudsBord = mesh.Nodes_Tags(["L0","L1","L2","L3"])
noeudCentreCercle = mesh.Nodes_Tags(["P8"])

comportement = Materials.Elas_Isot(2, E=210000e6, v=0.3, contraintesPlanes=False, epaisseur=1)

l = comportement.get_lambda()
mu = comportement.get_mu()

simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False)

simu.Set_Rayleigh_Damping_Coefs(0, 0)
simu.Solver_Set_Newton_Raphson_Algorithm(betha=1/4, gamma=1/2, dt=dt)

t=0

def Chargement():

    simu.add_dirichlet(noeudsBord, [0,0], ["x","y"], description="[0,0]")

    if t == t0:
        simu.add_neumann(noeudCentreCercle, [load,load], ["x","y"], description="[load,load]")
        

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, resultat, nodeValues=True)

tic = TicTac.Tic()

while t <= tMax:

    Chargement()

    simu.Solve()

    simu.Save_Iteration()

    tic.Tac("Simu","Resol\r",True)

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, resultat, nodeValues=True, ax=ax)
        plt.pause(1e-12)

    print(f"{t//dt}",end="\r")

    t += dt



folder = Folder.New_File("Ondes", results=True)

if makeMovie:
    PostTraitement.Make_Movie(folder, resultat, simu)

# PostTraitement.Save_Simulation_in_Paraview(folder, simu)

# TicTac.Tic.getGraphs(details=False)

plt.show()
