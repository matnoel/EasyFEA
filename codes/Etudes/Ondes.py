import matplotlib.pyplot as plt
import numpy as np

from Simu import Simu
import Materials
from Geom import Domain, Point, Circle
from Interface_Gmsh import Interface_Gmsh
import Affichage
import PostTraitement
import Dossier
import TicTac

Affichage.Clear()

plotModel = False
plotIter = True; resultat = "amplitudeSpeed"

makeMovie = False

Nt = 50
dt = 1e-6
tMax = dt*Nt
load = 100

a = 1
taille = a/100
diam = a/10
r = diam/2

domain = Domain(Point(x=-a/2, y=-a/2), Point(x=a/2, y=a/2), taille)
circle = Circle(Point(), diam, taille, isCreux=False)


interfaceGmsh = Interface_Gmsh(False)
mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")

if plotModel:
    Affichage.Plot_Model(mesh)
    plt.show()
noeudsBord = mesh.Nodes_Tag(["L1","L2","L3","L4"])
noeudCentreCercle = mesh.Nodes_Tag(["L5","L6","L7","L8"])
# Affichage.Plot_Noeuds(mesh, noeudCentreCercle)
# plt.show()

comportement = Materials.Elas_Isot(2, E=210000e6, v=0.3, contraintesPlanes=False, epaisseur=1)

materiau = Materials.Materiau(comportement, ro=8100)

simu = Simu(mesh, materiau, verbosity=False)

simu.Set_Rayleigh_Damping_Coefs(0, 0)
simu.Set_Hyperbolic_AlgoProperties(betha=1/4, gamma=1/2, dt=dt)

t=0

def Chargement():

    simu.add_dirichlet("displacement", noeudsBord, [0,0], ["x","y"], "[0,0]")

    if t >= 0*dt and t <= 1*dt:
        fonctionX = lambda x,y,z: load*(y-circle.center.y)/r
        fonctionY = lambda x,y,z: load*(x-circle.center.x)/r
        simu.add_pointLoad("displacement", noeudCentreCercle, [fonctionX, fonctionY], ["x","y"])

    # if t == dt:
    #     # simu.add_dirichlet("displacement", noeudCentre, [1e-2,1e-2], ["x","y"], "[load,load]")
        

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, resultat, valeursAuxNoeuds=True)

simu.Assemblage_u(steadyState=False)

tic = TicTac.Tic()

while t <= tMax:

    Chargement()

    simu.Solve_u(steadyState=False)

    simu.Save_Iteration()

    tic.Tac("Simu","Resol\r",True)

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, resultat, valeursAuxNoeuds=True, fig=fig, ax=ax, affichageMaillage=True)
        plt.pause(1e-12)

    print(f"{t//dt}",end="\r")

    t += dt



folder = Dossier.NewFile("Ondes", results=True)

if makeMovie:
    PostTraitement.MakeMovie(folder, resultat, simu)

PostTraitement.Save_Simulation_in_Paraview(folder, simu)

TicTac.Tic.getGraphs(details=True)

plt.show()
