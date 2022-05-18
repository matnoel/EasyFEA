
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import Dossier

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

test=True

comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe"]
regu = "AT2" # "AT1", "AT2"


nom="_".join([comp, split, regu])

nomDossier = "Benchmarck_Compression"

if test:
    file = Dossier.Append([nomDossier, "Test", nom])
else:
    file = Dossier.Append([nomDossier, nom])

file = Dossier.NewFile(file, results=True)


# Data

L=15e-3
h=30e-3
ep=1
diam=6e-3

E=12e9
v=0.2

gc = 1.4
l_0 = 0.12e-3

# Création de la simulations

if test:
    clD = l_0*2
    clC = l_0
else:
    clD = l_0/2
    clC = l_0/2

point = Point()
domain = Domain(point, Point(x=L, y=h), clD)
circle = Circle(Point(x=L/2, y=h/2), diam, clC)

interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=True)
# mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3", filename=Dossier.Append([file,"mesh.msh"]))
mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=False, epaisseur=ep)
phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
materiau = Materiau.Materiau(comportement, phaseFieldModel)

Affichage.Plot_Maillage(mesh)
plt.show()

simu = Simu.Simu(mesh, materiau, verbosity=False)

# Récupérations des noeuds

B_lower = Line(point,Point(x=L))
B_upper = Line(Point(y=h),Point(x=L, y=h))








