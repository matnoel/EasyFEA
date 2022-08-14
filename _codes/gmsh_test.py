
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simu
import matplotlib.pyplot as plt
from Materiaux import Materiau, Elas_Isot

L = 50e-3
h = L*0.3

pt1 = Point()
pt2 = Point(x=L)
pt3 = Point(x=L,y=h)
pt4 = Point(x=h, y=h)
pt5 = Point(x=h, y=L)
pt6 = Point(y=L)

interface = Interface_Gmsh(affichageGmsh=False)
mesh = interface.MeshFromGeom2D([pt1, pt2, pt3, pt4, pt5, pt6], isOrganised=True, elemType="QUAD8", tailleElement=h/10)

noeudsGauche = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == 0)
noeudsDroit = mesh.Get_Nodes_Conditions(conditionX=lambda x: x == L)

Affichage.Plot_Maillage(mesh)
# plt.show()

comportement = Elas_Isot(2, contraintesPlanes=True, epaisseur=h)

materiau = Materiau(comportement)

simu = Simu.Simu(mesh, materiau)

simu.add_dirichlet("displacement", noeudsGauche, [0,0], ["x","y"])
simu.add_lineLoad("displacement", noeudsDroit, [-800/h], ["y"])

Affichage.Plot_BoundaryConditions(simu)


simu.Assemblage_u()

simu.Solve_u()

Affichage.Plot_Result(simu, "Sxx")
Affichage.Plot_Result(simu, "Syy")
Affichage.Plot_Result(simu, "Sxy")
Affichage.Plot_Result(simu, "Svm")

plt.show()
