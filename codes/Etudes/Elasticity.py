from enum import Enum
import matplotlib.pyplot as plt
import os

from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Display
import Simulations
from Mesh import ElemType
import Materials
import TicTac
import Folder

dim = 2
N = 50*1.5 if dim == 2 else 20

class SimulationType(str, Enum):
    CPEF = "CPEF",
    EQUERRE = "EQUERRE",
    TEF2 = "TEF2"

simulationType = SimulationType.EQUERRE

interface = Interface_Gmsh(affichageGmsh=False, gmshVerbosity=False)

coef = 1
E=210000 # MPa
v=0.3

folder = Folder.New_File(f"Elasticity{dim}D", results=True)
if not os.path.exists(folder):
    os.makedirs(folder)

if simulationType == SimulationType.CPEF:
    dim = 3
    h=1
    fichier = Folder.Join([Folder.Get_Path(), "3Dmodels", "CPEF.stp"])
    mesh = interface.Mesh_Import_part(fichier, 5, ElemType.TETRA4)

    noeuds134 = mesh.Nodes_Tags(['S134'])

    comportement = Materials.Elas_Isot(dim, E=E, v=v)
    simu = Simulations.Simu_Displacement(mesh, comportement)

    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: z==0), [0,0,0], ['x','y','z'])
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: z<-50), [2], ["z"])
    
elif simulationType == SimulationType.EQUERRE:

    L = 120 #mm
    h = L * 0.3

    pt1 = Point(isOpen=True, r=-10)
    pt2 = Point(x=L)
    pt3 = Point(x=L,y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)

    crack = Line(Point(100, h, isOpen=True), Point(100-h/3, h*0.9, isOpen=True), h/N, isOpen=True)
    crack2 = Line(crack.pt2, Point(100-h/2, h*0.9, isOpen=True), h/N, isOpen=True)

    cracks = [crack, crack2]
    cracks = []

    listPoint = PointsList([pt1, pt2, pt3, pt4, pt5, pt6], h/N)
    # listPoint = PointsList([pt1, pt2, pt3, pt7], h/N)

    inclusions = [Circle(Point(x=h/2, y=h*(i+1)), h/4, meshSize=h/N, isCreux=True) for i in range(3)]

    inclusions.extend([Domain(Point(x=h,y=h/2-h*0.1), Point(x=h*2.1,y=h/2+h*0.1), isCreux=False, meshSize=h/N)])

    if dim == 2:
        mesh = interface.Mesh_2D(listPoint, inclusions, ElemType.TRI3, cracks)
    elif dim == 3:        
        mesh = interface.Mesh_3D(listPoint, inclusions, extrude=[0,0,-h], nCouches=4, elemType=ElemType.HEXA20)

    noeudsGauche = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    noeudsDroit = mesh.Nodes_Conditions(lambda x,y,z: x == L)

    comportement = Materials.Elas_Isot(dim, planeStress=True, thickness=h, E=E, v=v)
    simu = Simulations.Simu_Displacement(mesh, comportement)

    if dim == 2:
        simu.add_dirichlet(noeudsGauche, [0,0], ["x","y"])
        simu.add_lineLoad(noeudsDroit, [-800/h], ["y"])
    else:
        simu.add_dirichlet(noeudsGauche, [0,0,0], ["x","y","z"])
        simu.add_surfLoad(noeudsDroit, [-800/(h*h)], ["y"])
    
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
        mesh = interface.Mesh_2D(listPoint, [], ElemType.TRI6)
    elif dim == 3:
        # ["TETRA4", "HEXA8", "PRISM6"]
        mesh = interface.Mesh_3D(listPoint, [], extrude=[0,0,2*h], nCouches=10, elemType=ElemType.PRISM15)

    noeudsBas = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    noeudsGauche = mesh.Nodes_Conditions(lambda x,y,z: x==0)

    comportement = Materials.Elas_Isot(dim, planeStress=False, thickness=h, E=E, v=v)
    simu = Simulations.Simu_Displacement(mesh, comportement)

    if dim == 2:
        simu.add_dirichlet(noeudsBas, [0,0], ["x","y"])
    else:
        simu.add_dirichlet(noeudsBas, [0,0,0], ["x","y","z"])

    simu.add_volumeLoad(mesh.nodes, [-ro*g], ["y"], description="[-ro*g]")
    simu.add_surfLoad(noeudsGauche, [lambda x,y,z : w*g*(h-y)], ["x"], description="[w*g*(h-y)]")

if dim == 3:
    print(f"\nVolume = {mesh.volume:3f}")
else:
    print(f"\nVolume = {mesh.area*comportement.thickness:3f}")

Display.Plot_Mesh(mesh, folder=folder)
# Display.Plot_Model(mesh, showId=True)

simu.Solve()
simu.Save_Iter()

# Display.Plot_Elements(mesh, nodes=noeudsDroit, dimElem=2)
# ddlsF = Simulations.BoundaryCondition.Get_ddls_noeuds(dim, "displacement", noeudsDroit, ["y"])
# fr = np.sum(simu.Get_K_C_M_F()[0][ddlsF,:] @ simu.displacement)

# import PostProcessing
# PostProcessing.Make_Paraview(folder, simu)

print(simu)

Display.Plot_BoundaryConditions(simu)

# Display.Plot_Mesh(simu, deformation=True)
# Display.Plot_Result(simu, "Sxx", nodeValues=True, coef=1/coef)
# Display.Plot_Result(simu, "Syy", nodeValues=True, coef=1/coef)
# Display.Plot_Result(simu, "Sxy", nodeValues=True, coef=1/coef)
Display.Plot_Result(simu, "Svm", plotMesh=False, nodeValues=True, coef=1/coef, folder=folder)
# Display.Plot_Result(simu, "ux")

TicTac.Tic.Plot_History()


plt.show()
