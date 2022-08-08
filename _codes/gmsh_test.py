
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simu
import matplotlib.pyplot as plt
import Materiau

L = 180

pt1 = Point()
pt2 = Point(x=L)
pt3 = Point(x=L,y=L)
pt4 = Point(y=L)

interface = Interface_Gmsh(affichageGmsh=True)
mesh = interface.MeshFromGeom2D([pt1, pt2, pt4], isOrganised=False, elemType="QUAD4", tailleElement=L/10)

noeudsBas = mesh.Get_dict_Nodes_Conditions(conditionY=lambda y: y == 0)
noeudsGauche = mesh.Get_dict_Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y > 0 and y < L/2)

Affichage.Plot_Maillage(mesh)
plt.show()

comportement = Materiau.Elas_Isot(2, contraintesPlanes=False)

materiau = Materiau.Materiau(comportement)

simu = Simu.Simu(mesh, materiau)

simu.add_dirichlet("displacement", noeudsBas, [0,0], ["x","y"])
simu.add_lineLoad("displacement", noeudsGauche, [lambda x,y,z : 1000*9.81*(L-y)], ["x"])

Affichage.Plot_BoundaryConditions(simu)


simu.Assemblage_u()

simu.Solve_u()

Affichage.Plot_Result(simu, "Sxx")

plt.show()
