from enum import Enum
import matplotlib.pyplot as plt
import os

from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Affichage
import Simulations
from Mesh import ElemType
import Materials
import Folder
import TicTac

# import gmsh

dim = 2
N = 15

class SimulationType(str, Enum):
    CPEF = "CPEF",
    EQUERRE = "EQUERRE",
    TEF2 = "TEF2"

simulationType = SimulationType.TEF2

interface = Interface_Gmsh(affichageGmsh=False, gmshVerbosity=False)

coef = 1
E=210000 # MPa
v=0.3

folder = Folder.New_File("gmshTest", results=True)
if not os.path.exists(folder):
    os.makedirs(folder)

if simulationType == SimulationType.CPEF:
    dim = 3
    h=1
    fichier = Folder.Join([Folder.Get_Path(), "3Dmodels", "CPEF.stp"])
    mesh = interface.Mesh_Import_part3D(fichier, 5)

    noeuds134 = mesh.Nodes_Tags(['S134'])
    Affichage.Plot_Elements(mesh, noeuds134)
    plt.show()
    
elif simulationType == SimulationType.EQUERRE:

    L = 120 #mm
    h = L*0.3

    pt1 = Point(isOpen=True, r=0)
    pt2 = Point(x=L)
    pt3 = Point(x=L,y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)

    # crack = Line(pt1, Point(h/3, h/3), h/10, isOpen=False)
    crack = Line(pt1, Point(h/3, h/3), h/10, isOpen=False)

    listPoint = PointsList([pt1, pt2, pt3, pt4, pt5, pt6], h/N)
    # listPoint = PointsList([pt1, pt2, pt3, pt7], h/N)

    listObjetsInter = [Circle(Point(x=h/2, y=h*(i+1)), h/4, meshSize=h/N, isCreux=True) for i in range(3)]

    listObjetsInter.extend([Domain(Point(x=h,y=h/2-h*0.1), Point(x=h*2.1,y=h/2+h*0.1), isCreux=False, meshSize=h/N)])    

    if dim == 2:
        mesh = interface.Mesh_2D(listPoint, elemType=ElemType.QUAD4, inclusions=listObjetsInter, cracks=[], folder=folder)

        # Affichage.Plot_Noeuds(mesh, mesh.Nodes_Line(crack), showId=True)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_3D(listPoint, extrude=[0,0,h], nCouches=3, elemType=ElemType.HEXA8, inclusions=listObjetsInter, folder=folder)


        noeudsS3 = mesh.Nodes_Tags(["S9","S15","S14","S21"])
        Affichage.Plot_Elements(mesh, noeudsS3)
        # plt.show()

    noeudsGauche = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    noeudsDroit = mesh.Nodes_Conditions(lambda x,y,z: x == L)
    
elif simulationType == SimulationType.TEF2:

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

    listPoint = PointsList([pt1, pt2, pt3], taille)
    
    if dim == 2:
        mesh = interface.Mesh_2D(listPoint, elemType=ElemType.TRI3, inclusions=[], folder=folder)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_3D(listPoint, extrude=[0,0,2*h], nCouches=10, elemType=ElemType.TETRA4, inclusions=[], folder=folder)

    noeudsBas = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    noeudsGauche = mesh.Nodes_Conditions(lambda x,y,z: x==0)

Affichage.Plot_Mesh(mesh)
Affichage.Plot_Model(mesh, showId=False)
# plt.show()

comportement = Materials.Elas_Isot(dim, contraintesPlanes=True, epaisseur=h, E=E, v=v)

simu = Simulations.Simu_Displacement(mesh, comportement)

if simulationType == SimulationType.CPEF:
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: z==0), [0,0,0], ['x','y','z'])
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: z<-50), [2], ["z"])

elif simulationType == SimulationType.EQUERRE:
    if dim == 2:
        simu.add_dirichlet(noeudsGauche, [0,0], ["x","y"])
        simu.add_lineLoad(noeudsDroit, [-800/h], ["y"])
    else:
        simu.add_dirichlet(noeudsGauche, [0,0,0], ["x","y","z"])
        simu.add_surfLoad(noeudsDroit, [-800/(h*h)], ["y"])

elif simulationType == SimulationType.TEF2:
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
Affichage.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/coef)
Affichage.Plot_Result(simu, "Syy", nodeValues=True, coef=1/coef)
Affichage.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/coef)
Affichage.Plot_Result(simu, "Svm", plotMesh=False, nodeValues=True, deformation=True,coef=1/coef)

# TicTac.Tic.Plot_History()

plt.show()
