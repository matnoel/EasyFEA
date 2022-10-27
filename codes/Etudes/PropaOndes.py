import matplotlib.pyplot as plt
import numpy as np

from Simu import Simu
import Materials
from Geom import Domain, Point, Circle
from Interface_Gmsh import Interface_Gmsh
from Affichage import Plot_Model, Plot_Result, Clear
import PostTraitement
import Dossier
import TicTac

Clear()

plotIter = False; resultat = "amplitudeSpeed"

makeMovie = True

dt = 1e-5
Nt = 50
tMax = dt*Nt
load = 100

a = 1
taille = a/80

domain = Domain(Point(x=-a/2, y=-a/2), Point(x=a/2, y=a/2), taille)
circle = Circle(Point(), a/10, taille, isCreux=False)


interfaceGmsh = Interface_Gmsh(False)
mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")

# Plot_Model(mesh)
noeudsBord = mesh.Nodes_Tag(["L1","L2","L3","L4"])
noeudCentreCercle = mesh.Nodes_Tag(["S2"])
# plt.show()

comportement = Materials.Elas_Isot(2, E=210000e6, v=0.3, contraintesPlanes=False, epaisseur=1)

materiau = Materials.Materiau(comportement, ro=8100)

simu = Simu(mesh, materiau, verbosity=False)

simu.Set_Rayleigh_Damping_Coefs(0, 0)
simu.Set_Hyperbolic_AlgoProperties(betha=1/4, gamma=1/2, dt=dt)

t=0

def Chargement():

    simu.add_dirichlet("displacement", noeudsBord, [0,0], ["x","y"], "[0,0]")

    if t >= 0*dt and t <= 2*dt:
        simu.add_pointLoad("displacement", noeudCentreCercle, [load, 0], ["x","y"], "[load,load]")

    # if t == dt:
    #     # simu.add_dirichlet("displacement", noeudCentre, [1e-2,1e-2], ["x","y"], "[load,load]")
        

if plotIter:
    fig, ax, cb = Plot_Result(simu, resultat, valeursAuxNoeuds=True)

simu.Assemblage_u(steadyState=False)

tic = TicTac.Tic()

while t <= tMax:

    Chargement()

    simu.Solve_u(steadyState=False)

    simu.Save_Iteration()

    tic.Tac("Simu","Resol\r",True)

    if plotIter:
        cb.remove()
        fig, ax, cb = Plot_Result(simu, resultat, valeursAuxNoeuds=True, oldfig=fig, oldax=ax, affichageMaillage=True)
        plt.pause(1e-12)

    print(f"{t//dt}",end="\r")

    t += dt


folder = Dossier.NewFile("Ondes", results=True)

if makeMovie:
    PostTraitement.MakeMovie(folder, resultat, simu)

PostTraitement.Save_Simulation_in_Paraview(folder, simu)

TicTac.Tic.getGraphs(details=True)

plt.show()
