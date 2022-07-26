
from TicTac import TicTac
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import Dossier
import pandas as pd
import PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()

# L'objectif de ce script est de voir du chargement

# Options
comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "PlateWithHole_Chargement"

folder = Dossier.NewFile(nomDossier, results=True)

# Data
coef = 1e-3

L=15*coef
H=30*coef
h=H/2
ep=1*coef
diam=6*coef
r=diam/2

E=12e9
v=0.2
SIG = 10 #Pa

gc = 1.4
l_0 = 0.12 *coef*1.5

# Création du maillage
clD = l_0*2
clC = l_0

point = Point()
domain = Domain(point, Point(x=L, y=H), clD)
circle = Circle(Point(x=L/2, y=H-h), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)
mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

Affichage.Plot_Maillage(mesh,folder=folder)

# Récupérations des noeuds de chargement
B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=H),Point(x=L, y=H))
nodes0 = mesh.Get_Nodes_Line(B_lower)
nodesh = mesh.Get_Nodes_Line(B_upper)
node00 = mesh.Get_Nodes_Point(Point())   
noeuds_cercle = mesh.Get_Nodes_Circle(circle)
noeuds_cercle = noeuds_cercle[np.where(mesh.coordo[noeuds_cercle,1]<=circle.center.y)]

comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel, verbosity=False)

simu = Simu.Simu(mesh, materiau, verbosity=False)

simu.add_dirichlet("displacement", nodes0, [0], ["y"])
simu.add_dirichlet("displacement", node00, [0], ["x"])
# simu.add_surfLoad("displacement", noeuds_cercle, [-SIG], ["y"])
# simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(y-circle.center.y)/r], ["y"])
# simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(x-circle.center.x)/r*(y-circle.center.y)/r], ["x"])

# Sx = F * cos tet * abs(sin tet)
# Sy = F * sin tet * abs(sin tet)
simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(x-circle.center.x)/r * np.abs((y-circle.center.y)/r)], ["x"])
simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(y-circle.center.y)/r * np.abs((y-circle.center.y)/r)], ["y"])

Affichage.Plot_BoundaryConditions(simu)

simu.Assemblage_u()

simu.Solve_u()

simu.Save_Iteration()

Affichage.NouvelleSection("Résultats")

Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", folder=folder, filename='Sxx')
Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", folder=folder, filename='Syy')
Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", folder=folder, filename='Sxy')

# mini = np.min(simu.Get_Resultat("Syy", valeursAuxNoeuds=False))/SIG

PostTraitement.Save_Simulation_in_Paraview(folder, simu)

R = 10
F=15
tet = np.linspace(-np.pi,0,31)

xR = R * np.cos(tet)
yR = R * np.sin(tet)

xf = F * np.cos(tet) 
yf = F * np.sin(tet)

# xf = F * np.cos(tet)* np.abs(np.sin(tet)) + R * np.cos(tet)
# yf = F * np.sin(tet)* np.abs(np.sin(tet))

# xf = F * np.cos(tet)* 1 + R * np.cos(tet)
# yf = F * np.sin(tet)* 1

# xf = - F * np.cos(tet) * np.sin(tet)
# yf = - F * np.sin(tet) * np.sin(tet)

fig, ax = plt.subplots()
plt.plot(xR,yR)
plt.plot(xf,yf)
ax.axis('equal')
for x,y,dx,dy in zip(xR,yR,xf-xR,yf-yR):
    ax.arrow(x,y,dx,dy,width=0.1,length_includes_head=True)
ax.grid()




















TicTac.getResume()

plt.show()





