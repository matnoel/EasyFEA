import matplotlib.pyplot as plt

from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simulations
from Mesh import ElemType
import Materials
import Dossier

option = 2
dim = 3
N = 5

dictOptions = {
    1 : "CPEF",
    2 : "EQU",
    3 : "TEF2"
}

interface = Interface_Gmsh(affichageGmsh=False, gmshVerbosity=False)

coef = 1
E=210000 # MPa
v=0.3


if option == 1:
    dim = 3
    h=1
    fichier = Dossier.Join([Dossier.GetPath(), "3Dmodels", "CPEF.stp"])
    mesh = interface.Mesh_Importation3D(fichier, 10)

    noeuds134 = mesh.Nodes_Tag(['S134'])
    Affichage.Plot_Elements(mesh, noeuds134)
    plt.show()
    
elif option ==  2:

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

    listObjetsInter.extend([Domain(Point(x=h,y=h/2-h*0.1), Point(x=h*2.1,y=h/2+h*0.1), isCreux=False, taille=h/N)])    

    if dim == 2:
        mesh = interface.Mesh_From_Points_2D(listPoint, elemType=ElemType.QUAD8, geomObjectsInDomain=listObjetsInter, tailleElement=h/N)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_From_Points_3D(listPoint, extrude=[0,0,h], nCouches=3, elemType=ElemType.HEXA8, interieursList=listObjetsInter, tailleElement=h/N)


        noeudsS3 = mesh.Nodes_Tag(["S9","S15","S14","S21"])
        Affichage.Plot_Elements(mesh, noeudsS3)
        # plt.show()

    noeudsGauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0)
    noeudsDroit = mesh.Nodes_Conditions(conditionX=lambda x: x == L)
    
elif option == 3:

    # dim = 2 # sinon problème dans importation maillage

    coef = 1e6
    E=15000*coef # Pa
    v=0.25
    g=9.81

    h = 180 #m
    taille = h/N

    pt1 = Point()
    pt2 = Point(x=h)
    pt3 = Point(y=h)

    ro = 2400 # kg m-3
    w = 1000 # kg m-3

    listPoint = [pt1, pt2, pt3]
    
    if dim == 2:
        mesh = interface.Mesh_From_Points_2D(listPoint, elemType=ElemType.TRI3, geomObjectsInDomain=[], tailleElement=taille)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_From_Points_3D(listPoint, extrude=[0,0,2*h], nCouches=10, elemType=ElemType.PRISM6, interieursList=[], tailleElement=taille)

    noeudsBas = mesh.Nodes_Line(Line(pt1, pt2))
    noeudsGauche = mesh.Nodes_Line(Line(pt1, pt3))

# Affichage.Plot_Maillage(mesh)
# Affichage.Plot_Model(mesh, showId=False)
# plt.show()

comportement = Materials.Elas_Isot(dim, contraintesPlanes=True, epaisseur=h, E=E, v=v)

materiau = Materials.Create_Materiau(comportement)

simu = Simulations.Create_Simu(mesh, materiau)

if option == 1:
    simu.add_dirichlet(mesh.Nodes_Conditions(conditionZ=lambda z : z==0), [0,0,0], ['x','y','z'])
    simu.add_dirichlet(mesh.Nodes_Conditions(conditionZ=lambda z : z<-50), [2], ["z"])

elif option == 2:
    if dim == 2:
        simu.add_dirichlet(noeudsGauche, [0,0], ["x","y"])
        simu.add_lineLoad(noeudsDroit, [-8000/h], ["y"])
    else:
        simu.add_dirichlet(noeudsGauche, [0,0,0], ["x","y","z"])
        simu.add_surfLoad(noeudsDroit, [-800/(h*h)], ["y"])

elif option == 3:
    if dim == 2:
        simu.add_dirichlet(noeudsBas, [0,0], ["x","y"])
    else:
        simu.add_dirichlet(noeudsBas, [0,0,0], ["x","y","z"])

    simu.add_volumeLoad(mesh.nodes, [-ro*g], ["y"], description="[-ro*g]")
    simu.add_surfLoad(noeudsGauche, [lambda x,y,z : w*g*(h-y)], ["x"], description="[w*g*(h-y)]")

simu.Solve()

simu.Save_Iteration()
# PostTraitement.Save_Simulation_in_Paraview(Dossier.NewFile("gmsh test",results=True), simu)
simu.Resultats_Resume()

# Affichage.Plot_ElementsMaillage(mesh, nodes=noeudsDroit, dimElem =2)
Affichage.Plot_BoundaryConditions(simu)

# Affichage.Plot_Maillage(simu, deformation=True)
Affichage.Plot_Result(simu, "Sxx", valeursAuxNoeuds=True, coef=1/coef)
Affichage.Plot_Result(simu, "Syy", valeursAuxNoeuds=True, coef=1/coef)
Affichage.Plot_Result(simu, "Sxy", valeursAuxNoeuds=True, coef=1/coef)
Affichage.Plot_Result(simu, "Svm", affichageMaillage=False, valeursAuxNoeuds=True, deformation=True,coef=1/coef)


plt.show()
