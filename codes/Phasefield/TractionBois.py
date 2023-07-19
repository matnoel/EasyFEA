import Interface_Gmsh
import Simulations
import Materials
import Display
import PostTraitement
from Geom import np, Point, Domain, Circle, PointsList, Line
import Folder

plt = Display.plt
import pandas as pd


Display.Clear()

folder = Folder.New_File("TractionBois", results=True)

useSmallCrack = True

pathData = "/Users/matnoel/Library/CloudStorage/OneDrive-Personal/__Doctorat/Essais/TractionBois/data.xlsx"

df = pd.read_excel(pathData)

forces = df["Force"].values #N
displacements = df["Dep"].values #mm

filtre = displacements < 0.05

forces = forces[filtre]
displacements = displacements[filtre]

plt.figure()
plt.plot(displacements, forces)

L = 105
H = 70

h = H/2

r3 = 3

epFissure = 0.5
lFissure = 20
a = 20
c = 20

d1 = 25.68
d2 = 40.85

alpha1 = 2.5 * np.pi/180
alpha2 = 20 * np.pi/180
alpha3 = 34 * np.pi/180

betha = (np.pi - alpha3 - (np.pi/2-alpha1))/2
d = r3/np.tan(betha)

l0 = H/70
tailleFin = l0/2
tailleGros = l0*3

p0 = Point(x=0, y=-epFissure/2)
p1 = Point(x=0, y=-h)
p2 = Point(x=a, y=-h)
p3 = Point(x=a+(d1+d)*np.sin(alpha1), y=-h+(d1+d)*np.cos(alpha1), r=r3)
p4 = Point(x=L-c-d2*np.cos(alpha2), y=-h+d2*np.sin(alpha2))
p5 = Point(x=L-c, y=-h)
p6 = Point(x=L, y=-h)
p7 = Point(x=L, y=h)
p8 = Point(x=L-c, y=h)
p9 = Point(x=L-c-d2*np.cos(alpha2), y=h-d2*np.sin(alpha2))
p10 = Point(x=a+(d1+d)*np.sin(alpha1), y=h-(d1+d)*np.cos(alpha1), r=r3)
p11 = Point(x=a, y=h)
p12 = Point(x=0, y=h)
p13 = Point(x=0, y=epFissure/2)
p14 = Point(x=lFissure, y=epFissure/2, r=epFissure/2)
p15 = Point(x=lFissure, y=-epFissure/2, r=epFissure/2)

if useSmallCrack:
    p0 = Point(isOpen=True)
    listPoint = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    crack = Line(p0, Point(x=lFissure), tailleFin, isOpen=True)
    cracks = [crack]
else:
    listPoint = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]
    cracks = []

points = PointsList(listPoint, tailleGros)

diam = 5
r = diam/2
c1 = Circle(Point(a/2, -h+7.5), diam, tailleFin, isCreux=True)
c2 = Circle(Point(L-c/2, -h+7.5), diam, tailleFin, isCreux=True)
c3 = Circle(Point(L-c/2, h-7.5), diam, tailleFin, isCreux=True)
c4 = Circle(Point(a/2, h-7.5), diam, tailleFin, isCreux=True)

geomObjectsInDomain = [c1, c2, c3, c4]

interface = Interface_Gmsh.Interface_Gmsh(False, False)

zone = 10
refineDomain = Domain(Point(lFissure-zone, -zone), Point(L, zone), meshSize=tailleFin)
mesh = interface.Mesh_2D(points, refineGeom=refineDomain, inclusions=geomObjectsInDomain, elemType="TRI3", cracks=cracks)

# Affichage.Plot_Mesh(mesh)
Display.Plot_Model(mesh)
# plt.show()


# MATERIAU

# El=11580*1e6
# Gc = 300*1e6 # J/mm2
Gc = 1*1e-1 # J/mm2
El=12000
Et=500
# Et=50
Gl=450
vl=0.02
vt=0.44
v=0
comportement = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=True, epaisseur=12.5, axis_l=np.array([1,0,0]), axis_t=np.array([0,1,0]))

splits = Materials.PhaseField_Model.SplitType
reg = Materials.PhaseField_Model.RegularizationType

a1 = np.array([1,0])
M1 = np.einsum("i,j->ij", a1, a1)

a2 = np.array([0,1])
M2 = np.einsum("i,j->ij", a2, a2)

coef = El/Et
A = np.eye(2) + coef * M1 + 0 * M2

pfm = Materials.PhaseField_Model(comportement, splits.Zhang, reg.AT2, Gc, l0, A=A)

simu = Simulations.Simu_PhaseField(mesh, pfm, verbosity=False)

noeudsHaut = mesh.Nodes_Circle(c4)
noeudsHaut = noeudsHaut[np.where(mesh.coordoGlob[noeudsHaut,1]>=c4.center.y)] 

noeudsBas = mesh.Nodes_Circle(c1)
noeudsBas = noeudsBas[np.where(mesh.coordoGlob[noeudsBas,1]<=c1.center.y)]

noeudPoint = mesh.Nodes_Point(c1.center - [0, diam/2])

# if len(cracks) > 0: Affichage.Plot_Nodes(mesh, mesh.Nodes_Line(cracks[0]), showId=True)


if useSmallCrack:
    noeudsBord = mesh.Nodes_Tags(mesh.Get_list_groupElem(1)[0].nodeTags)
    noeudsCrack = mesh.Nodes_Line(crack)
    noeudsBord = list(set(noeudsBord) - set(noeudsCrack))
else:
    noeudsBord = mesh.Nodes_Tags([f"L{i}" for i in range(15)])

# Affichage.Plot_Nodes(mesh, noeudsBord)

def Chargement(force: float):
    simu.Bc_Init()

    SIG = force/(np.pi*r**2/2)
    
    simu.add_dirichlet(noeudPoint, [0], ["x"])

    
    simu.add_dirichlet(noeudsBas, [0], ["y"])
    simu.add_surfLoad(noeudsHaut, [lambda x,y,z: SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])

    # SIG *= 1/2
    # simu.add_surfLoad(noeudsHaut, [lambda x,y,z: SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])
    # simu.add_surfLoad(noeudsBas, [lambda x,y,z: -SIG*(y-c4.center.y)/r * np.abs((y-c4.center.y)/r)], ["y"])

Chargement(0)

Display.Plot_BoundaryConditions(simu)
# plt.show()

fig_Damage, ax_Damage, cb_Damage = Display.Plot_Result(simu, "damage")

# for iter, force, dep in zip(range(len(forces)), forces, displacements):

nf = 100
for iter, force in enumerate(np.linspace(0, 35, nf)):

    Chargement(force)

    # simu.Solve(1e-1, maxIter=50, convOption=1)
    simu.Solve(1e-0, maxIter=50, convOption=2)

    simu.Save_Iteration()

    depNum = np.max(simu.displacement[noeudsHaut])

    # ecart = np.abs(depNum-dep)/dep
    # print(ecart)

    # Affichage.Plot_Result(simu, "Syy")
    # plt.show()

    pourcent = iter/nf

    simu.Resultats_Set_Resume_Iteration(iter, force, "N", pourcent, True)

    cb_Damage.remove()
    fig_Damage, ax_Damage, cb_Damage = Display.Plot_Result(simu, "damage", ax=ax_Damage)
    plt.pause(1e-12)

    if np.max(simu.damage[noeudsBord]) >= 0.95:
        break

PostTraitement.Make_Paraview(folder, simu)