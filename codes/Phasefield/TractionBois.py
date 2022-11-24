import Interface_Gmsh
import Simulations
import Materials
import Affichage
from Geom import np, Point, Domain, Circle
import Folder
import pandas as pd

folder = Folder.New_File("TractionBois", results=True)

pathData = Folder.Join([Folder.Get_Path(), "data", "TractionBois", 'data.xlsx'])

df = pd.read_excel(pathData)

forces = df["Force"].values #N
displacements = df["Dep"].values #mm


plt = Affichage.plt

L = 105
H = 70

h = H/2

r3 = 3

ep = 1


l = 20
a = 20
c = 20

d1 = 25.68
d2 = 40.85

a1 = 2.5 *np.pi/180
a2 = 20 *np.pi/180

Gc = 300*1e-6 # J/mm2
l0 = H/100

p0F = Point(x=0, y=ep/2)
p1F = Point(x=l, y=0)
p2F = Point(x=0, y=-ep/2)
p1 = Point(x=0, y=-h)
p2 = Point(x=a, y=-h)
p3 = Point(x=a+d1*np.sin(a1)+r3, y=-h+d1*np.cos(a1))
p4 = Point(x=L-c-d2*np.cos(a2), y=-h+d2*np.sin(a2))
p5 = Point(x=L-c, y=-h)
p6 = Point(x=L, y=-h)
p7 = Point(x=L, y=h)
p8 = Point(x=L-c, y=h)
p9 = Point(x=L-c-d2*np.cos(a2), y=h-d2*np.sin(a2))
p10 = Point(x=a+d1*np.sin(a1)+r3, y=h-d1*np.cos(a1))
p11 = Point(x=a, y=h)
p12 = Point(x=0, y=h)

listPoint = [p0F, p1F, p2F, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]


c1 = Circle(Point(a/2, -h+7.5), 5, l0/2, isCreux=True)
c2 = Circle(Point(L-c/2, -h+7.5), 5, l0/2, isCreux=True)
c3 = Circle(Point(L-c/2, h-7.5), 5, l0/2, isCreux=True)
c4 = Circle(Point(a/2, h-7.5), 5, l0/2, isCreux=True)

geomObjectsInDomain = [c1, c2, c3, c4]

# fig, ax = plt.subplots()
# [ax.scatter(point.x, point.y, label=p) for p, point in enumerate(listPoint)]
# ax.autoscale()
# ax.legend()
# plt.show()


interface = Interface_Gmsh.Interface_Gmsh(False, False)

taille = l0*5
zone = 6*ep
refineDomain = Domain(Point(l-zone, -zone), Point(L, zone), taille=l0/2)
mesh = interface.Mesh_From_Points_2D(listPoint, tailleElement=taille, refineGeom=refineDomain, geomObjectsInDomain=geomObjectsInDomain)

Affichage.Plot_Maillage(mesh)
# Affichage.Plot_Model(mesh)
plt.show()


# MATERIAU

# El=11580*1e6
El=12e9
Et=500*1e6
Gl=450*1e6
vl=0.02
vt=0.44
v=0
comportement = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=True, epaisseur=12.5, axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

splits = Materials.PhaseField_Model.SplitType
reg = Materials.PhaseField_Model.RegularizationType

pfm = Materials.PhaseField_Model(comportement, splits.Zhang, reg.AT2, Gc, l0)

materiau = Materials.Create_Materiau(pfm)

simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)

noeudsHaut = mesh.Nodes_Tag(["L31","L30"])
noeudsBas = mesh.Nodes_Tag(["L16","L17"])


ax = Affichage.Plot_Noeuds(mesh, noeudsHaut)
Affichage.Plot_Noeuds(mesh, noeudsBas, ax=ax)
# plt.show()

fig, ax, cb = Affichage.Plot_Result(simu, "damage")

for iter, dep in enumerate(displacements):

    simu.Bc_Init()

    simu.add_dirichlet(noeudsBas, [0,0], ["x","y"])
    simu.add_dirichlet(noeudsHaut, [np.abs(dep)], ["y"])

    # Affichage.Plot_BoundaryConditions(simu)
    # plt.show()

    simu.Solve()

    simu.Resultats_Set_Resume_Iteration(iter, dep, "mm", dep/displacements[-1],True)

    cb.remove()
    fig, ax, cb = Affichage.Plot_Result(simu, "damage", ax=ax)
    plt.pause(1e-12)
