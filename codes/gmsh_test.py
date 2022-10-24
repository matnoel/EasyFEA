
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simu
import matplotlib.pyplot as plt
from Materials import Materiau, Elas_Isot
import Dossier

useCPEF = True

interface = Interface_Gmsh(affichageGmsh=False, gmshVerbosity=False)

if useCPEF:
    dim = 3
    h=1
    fichier = Dossier.Join([Dossier.GetPath(), "3Dmodels", "CPEF.stp"])
    mesh = interface.Mesh_Importation3D(fichier, 2)
    Affichage.Plot_Maillage(mesh)
    plt.show()
else:

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

    listObjetsInter = [Circle(Point(x=h/2, y=h*(i+1)), h/4, isCreux=True) for i in range(3)]

    listObjetsInter.extend([Domain(Point(x=h,y=h/2-h*0.1), Point(x=h*2.1,y=h/2+h*0.1), isCreux=True)])    

    if dim == 2:
        mesh = interface.Mesh_From_Points_2D(listPoint, elemType="TRI10", interieursList=listObjetsInter, tailleElement=h/2)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_From_Points_3D(listPoint, extrude=[0,0,h], nCouches=3, elemType="HEXA8", interieursList=listObjetsInter, tailleElement=h/6)

    noeudsGauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0)
    noeudsDroit = mesh.Nodes_Conditions(conditionX=lambda x: x == L)

    Affichage.Plot_Maillage(mesh)
    # plt.show()

comportement = Elas_Isot(dim, contraintesPlanes=True, epaisseur=h)

materiau = Materiau(comportement)

simu = Simu.Simu(mesh, materiau)

if useCPEF:
    simu.add_dirichlet("displacement", mesh.Nodes_Conditions(conditionZ=lambda z : z==0), [0,0,0], ['x','y','z'])
    simu.add_dirichlet("displacement", mesh.Nodes_Conditions(conditionZ=lambda z : z<-50), [2], ["z"])
else:
    simu.add_dirichlet("displacement", noeudsGauche, [0,0], ["x","y"])
    if dim == 2:
        simu.add_lineLoad("displacement", noeudsDroit, [-800/h], ["y"])
    else:
        simu.add_surfLoad("displacement", noeudsDroit, [-800/(h*h)], ["x"])

simu.Assemblage_u()
simu.Solve_u()

simu.Resume()

# Affichage.Plot_ElementsMaillage(mesh, nodes=noeudsDroit, dimElem =2)
Affichage.Plot_BoundaryConditions(simu)

# Affichage.Plot_Maillage(simu, deformation=True)

Affichage.Plot_Result(simu, "Sxy")

Affichage.Plot_Result(simu, "Sxx")
Affichage.Plot_Result(simu, "Svm", affichageMaillage=True, valeursAuxNoeuds=True, deformation=True)


plt.show()
