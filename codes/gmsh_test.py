
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simu
import matplotlib.pyplot as plt
from Materials import Materiau, Elas_Isot

dim = 3

L = 120 #mm
h = L*0.3

pt1 = Point()
pt2 = Point(x=L)
pt3 = Point(x=L,y=h)
pt4 = Point(x=h, y=h)
pt5 = Point(x=h, y=L)
pt6 = Point(y=L)
pt7 = Point(x=h, y=h)

listPoint = [pt1, pt2, pt3, pt4, pt5, pt6]
# listPoint = [pt1, pt2, pt3, pt7]

interface = Interface_Gmsh(affichageGmsh=True)
if dim == 2:
    mesh = interface.Mesh_From_Points_2D(listPoint, 
    elemType="TRI3", tailleElement=h/10, isOrganised=False)
elif dim == 3:
    # ["TETRA4", "HEXA8", "PRISM6"]
    mesh = interface.Mesh_From_Points_3D(listPoint, extrude=[0,0,h], nCouches=3,
    elemType="PRISM6", isOrganised=True, tailleElement=h/6)

noeudsGauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0)
noeudsDroit = mesh.Nodes_Conditions(conditionX=lambda x: x == L)

Affichage.Plot_Maillage(mesh)
# plt.show()

comportement = Elas_Isot(dim, contraintesPlanes=True, epaisseur=h)

materiau = Materiau(comportement)

simu = Simu.Simu(mesh, materiau)

simu.add_dirichlet("displacement", noeudsGauche, [0,0], ["x","y"])
if dim == 2:
    simu.add_lineLoad("displacement", noeudsDroit, [-800/h], ["y"])
else:
    simu.add_surfLoad("displacement", noeudsDroit, [-800/(h*h)], ["y"])



simu.Assemblage_u()
simu.Solve_u()

# Affichage.Plot_ElementsMaillage(mesh, nodes=noeudsDroit, dimElem =2)
Affichage.Plot_BoundaryConditions(simu)

Affichage.Plot_Maillage(simu, deformation=True)

Affichage.Plot_Result(simu, "Syy")
Affichage.Plot_Result(simu, "Sxx")
Affichage.Plot_Result(simu, "Sxy")
Affichage.Plot_Result(simu, "Svm")


plt.show()
