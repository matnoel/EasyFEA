import matplotlib.pyplot as plt
import numpy as np

import Simulations
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
load = 1

f0=2
a0=1

t0 = dt
Gt = lambda t: a0*2*((np.pi*f0*(t-t0))**2-1)*np.exp(-(np.pi*f0*(t-t0))**2)
Gx = lambda x: 1/np.sqrt(np.pi)*np.exp(-x**2*10)
# Gx = lambda x: 1

a = 1
taille = a/40
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
# noeudCentreCercle = mesh.Nodes_Tag(["L5","L6","L7","L8"])
noeudCentreCercle = mesh.Nodes_Tag(["P9"])
# Affichage.Plot_Noeuds(mesh, noeudCentreCercle)
# plt.show()

comportement = Materials.Elas_Isot(2, E=210000e6, v=0.3, contraintesPlanes=False, epaisseur=1)

materiau = Materials.Create_Materiau(comportement, ro=8100)

l = comportement.get_lambda()
mu = comportement.get_mu()
ro = materiau.ro

# cp = np.sqrt((l+2*mu)/ro)
# cs = np.sqrt(r/ro)
# a0 = 1
# yx = a0 * np.exp(-(10*x/cp)**2)



# Gt = lambda t: a0*2*((np.pi*f0*(t-t0))**2-1)*np.exp(-(np.pi*f0*(t-t0))**2)


simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)

simu.Set_Rayleigh_Damping_Coefs(0, 0)
simu.Solveur_Newton_Raphson_Properties(betha=1/4, gamma=1/2, dt=dt)

t=0

def Chargement():

    simu.add_dirichlet(noeudsBord, [0,0], ["x","y"], description="[0,0]")


    fonctionX = 0
    # fonctionX = lambda x,y,z: load*(y-circle.center.y)/r
    # fonctionY = lambda x,y,z: load*(x-circle.center.x)/r
    
    fonctionX = lambda x,y,z: Gx(x)*Gt(t)
    fonctionY = lambda x,y,z: Gx(x)*Gt(t)
    simu.add_pointLoad(noeudCentreCercle, [fonctionX, fonctionY], ["x","y"])

    # if t == dt:
    #     # simu.add_dirichlet(noeudCentre, [1e-2,1e-2], ["x","y"], "[load,load]")
        

if plotIter:
    fig, ax, cb = Affichage.Plot_Result(simu, resultat, valeursAuxNoeuds=True)

simu.Assemblage(steadyState=False)

tic = TicTac.Tic()

while t <= tMax:

    Chargement()

    simu.Solve(steadyState=False)

    simu.Save_Iteration()

    tic.Tac("Simu","Resol\r",True)

    if plotIter:
        cb.remove()
        fig, ax, cb = Affichage.Plot_Result(simu, resultat, valeursAuxNoeuds=True, ax=ax)
        plt.pause(1e-12)

    print(f"{t//dt}",end="\r")

    t += dt



folder = Dossier.NewFile("Ondes", results=True)

if makeMovie:
    PostTraitement.MakeMovie(folder, resultat, simu)

# PostTraitement.Save_Simulation_in_Paraview(folder, simu)

# TicTac.Tic.getGraphs(details=False)

plt.show()
