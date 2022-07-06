
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

# L'objectif de ce script est de voir l'influence du chamgement de taille du probleme

# Il peut etre interessant de faire varier le domaine la taille et la position du trou

# Options
comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"
contraintesPlanes = True

nom="_".join([comp, split, regu])

nomDossier = "PlateWithHole_Dimension"

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
clD = l_0*5
clC = l_0

list_SxxA = []
list_SyyA = []
list_SxyA = []
list_SxxB = []
list_SyyB = []
list_SxyB = []

param_name='H'
param1 = H
param2 = L
list_cc = np.linspace(1/2,5,30)

for cc in list_cc:

    H = param1*cc
    # L = param2*cc
    h=H/2
    print(cc)
    point = Point()
    domain = Domain(point, Point(x=L, y=H), clD)
    circle = Circle(Point(x=L/2, y=H-h), diam, clC)

    interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)
    mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

    # Affichage.Plot_Maillage(mesh)

    # Récupérations des noeuds de chargement
    B_lower = Line(point,Point(x=L))
    B_upper = Line(Point(y=H),Point(x=L, y=H))
    nodes0 = mesh.Get_Nodes_Line(B_lower)
    nodesh = mesh.Get_Nodes_Line(B_upper)
    node00 = mesh.Get_Nodes_Point(Point())   

    # Noeuds en A et en B
    nodeA = mesh.Get_Nodes_Point(Point(x=L/2, y=H-h+diam/2))
    nodeB = mesh.Get_Nodes_Point(Point(x=L/2+diam/2, y=H-h))

    comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
    phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
    materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel, verbosity=False)

    simu = Simu.Simu(mesh, materiau, verbosity=False)

    simu.add_dirichlet("displacement", nodes0, [0], ["y"])
    simu.add_dirichlet("displacement", node00, [0], ["x"])
    simu.add_surfLoad("displacement", nodesh, [-SIG], ["y"])

    # Affichage.Plot_BoundaryConditions(simu)

    simu.Assemblage_u()

    simu.Solve_u()

    list_SxxA.append(simu.Get_Resultat("Sxx", True)[nodeA])
    list_SyyA.append(simu.Get_Resultat("Syy", True)[nodeA])
    list_SxyA.append(simu.Get_Resultat("Sxy", True)[nodeA])

    list_SxxB.append(simu.Get_Resultat("Sxx", True)[nodeB])
    list_SyyB.append(simu.Get_Resultat("Syy", True)[nodeB])
    list_SxyB.append(simu.Get_Resultat("Sxy", True)[nodeB])

Affichage.NouvelleSection("Résultats")

Affichage.Plot_Maillage(mesh,folder=folder, title=f"mesh_{param_name}")
Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xx}/\sigma$", folder=folder, filename='Sxx')
Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{yy}/\sigma$", folder=folder, filename='Syy')
Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/SIG, title=r"$\sigma_{xy}/\sigma$", folder=folder, filename='Sxy')

fig, ax = plt.subplots()

ax.plot(list_cc, np.array(list_SxxA)/SIG,label='SxxA/SIG')
ax.plot(list_cc, np.array(list_SxyA)/SIG,label='SxyA/SIG')
ax.plot(list_cc, np.array(list_SyyA)/SIG,label='SyyA/SIG')
ax.plot(list_cc, np.array(list_SxxB)/SIG,label='SxxB/SIG')
ax.plot(list_cc, np.array(list_SxyB)/SIG,label='SxyB/SIG')
ax.plot(list_cc, np.array(list_SyyB)/SIG,label='SyyB/SIG')
ax.grid()
plt.legend()
ax.set_title(param_name)
ax.set_xlabel('coef')

PostTraitement.Save_fig(folder, param_name)

TicTac.getResume()

plt.show()





