from TicTac import Tic
import Materials
from Geom import *
import Affichage as Affichage
import Interface_Gmsh as Interface_Gmsh
import Simulations
import Folder
import PostTraitement as PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()

# L'objectif de ce script est de voir du chargement

# Options
dim = 3
comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "PlateWithHole_Chargement"

folder = Folder.New_File(nomDossier, results=True)

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
l_0 = 0.12 *coef*10

# Création du maillage
clD = l_0*2
clC = l_0

point = Point()
domain = Domain(point, Point(x=L, y=H), clD)
circle = Circle(Point(x=L/2, y=H-h), diam, clC, isCreux=True)
val = diam*2
refineGeom = Domain(Point(x=L/2-val/2, y=(H-h)-val/2), Point(x=L/2+val/2, y=(H-h)+val/2), meshSize=clC/2)
# refineGeom = Circle(Point(x=L/2, y=H-h), val, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)
if dim == 2:
    mesh = interfaceGmsh.Mesh_Domain_Circle_2D(domain, circle, "QUAD8", refineGeom=refineGeom)
else:
    mesh = interfaceGmsh.Mesh_Domain_Circle_3D(domain, circle, [0,0,6*coef], nCouches=6, elemType="HEXA8", refineGeom=refineGeom)

Affichage.Plot_Model(mesh)
# plt.show()
Affichage.Plot_Mesh(mesh,folder=folder)

# Récupérations des noeuds de chargement
B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=H),Point(x=L, y=H))

noeuds_22 = mesh.Nodes_Tag(["S6","S7"])

if len(noeuds_22) > 0:
    Affichage.Plot_Nodes(mesh, nodes=noeuds_22)
    # plt.show()

# nodes0 = mesh.Nodes_Line(B_lower)
nodes0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
# nodesh = mesh.Nodes_Line(B_upper)
nodesh = mesh.Nodes_Conditions(lambda x,y,z: y==H)
node00 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
if dim == 2:
    noeuds_cercle = mesh.Nodes_Circle(circle)
else:
    noeuds_cercle = mesh.Nodes_Cylindre(circle,[0,0,1])

noeuds_cercle = noeuds_cercle[np.where(mesh.coordo[noeuds_cercle,1]<=circle.center.y)]

comportement = Materials.Elas_Isot(dim, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l_0)

simu = Simulations.Simu_PhaseField(mesh, phaseFieldModel, verbosity=False)

simu.add_dirichlet(nodes0, [0], ["y"])
simu.add_dirichlet(node00, [0], ["x"])
# simu.add_surfLoad(noeuds_cercle, [-SIG], ["y"])
# simu.add_surfLoad(noeuds_cercle, [lambda x,y,z: SIG*(y-circle.center.y)/r], ["y"])
# simu.add_surfLoad(noeuds_cercle, [lambda x,y,z: SIG*(x-circle.center.x)/r*(y-circle.center.y)/r], ["x"])

# Sx = F * cos tet * abs(sin tet)
# Sy = F * sin tet * abs(sin tet)
simu.add_surfLoad(noeuds_cercle, [lambda x,y,z: SIG*(x-circle.center.x)/r * np.abs((y-circle.center.y)/r)], ["x"])
simu.add_surfLoad(noeuds_cercle, [lambda x,y,z: SIG*(y-circle.center.y)/r * np.abs((y-circle.center.y)/r)], ["y"])

# simu.add_surfLoad(nodesh, [-SIG], ["y"])

Affichage.Plot_BoundaryConditions(simu)

simu.Solve()

simu.Save_Iteration()

Affichage.NouvelleSection("Résultats")

Affichage.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", folder=folder, filename='Sxx')
Affichage.Plot_Result(simu, "Syy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", folder=folder, filename='Syy')
Affichage.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", folder=folder, filename='Sxy')
Affichage.Plot_Result(simu, "Svm", coef=1/SIG, title=r"$\sigma_{vm}/\sigma$", folder=folder, filename='Svm')

# mini = np.min(simu.Get_Resultat("Syy", nodeValues=False))/SIG

# PostTraitement.Save_Simulation_in_Paraview(folder, simu)

R = 10
Load=15
tet = np.linspace(-np.pi,0,31)

xR = R * np.cos(tet)
yR = R * np.sin(tet)

# xf = Load * np.cos(tet) 
# yf = Load * np.sin(tet)

# xf = Load * np.cos(tet)* np.abs(np.sin(tet)) + R * np.cos(tet)
xf = xR
# yf = Load * np.sin(tet)* np.abs(np.sin(tet))
yf = Load * np.sin(tet)**2

# xf = Load * np.cos(tet)* 1 + R * np.cos(tet)
# yf = Load * np.sin(tet)* 1

# xf = - Load * np.cos(tet) * np.sin(tet)
# yf = - Load * np.sin(tet) * np.sin(tet)

fig, ax = plt.subplots()

ax.plot(tet, yf)

# ax.plot(xR,yR)
# ax.plot(xf,yf)
# for x,y,dx,dy in zip(xR,yR,xf-xR,yf-yR):
#     ax.arrow(x,y,dx,dy,width=0.1,length_includes_head=True)
# ax.axis('equal')

ax.grid()




















Tic.Resume()

plt.show()





