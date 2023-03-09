import Affichage
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Materials
import Simulations

plt = Affichage.plt

r = 0

p0 = Point(0, r=r)
p1 = Point(1, r=r)
p2 = Point(1, 1, r=r)
p3 = Point(0, 1, r=r)
pts = [p0, p1, p2, p3]

meshSize = 1/20

points = PointsList(pts, meshSize, isCreux=False)

circle = Circle(Point(1/2, 1/2), 1/3, meshSize, isCreux=False)

gmshInterface = Interface_Gmsh(False, False)

mesh = gmshInterface.Mesh_2D(points, inclusions=[circle])

# Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

elementsCircle = mesh.Elements_Tags(["S0"])
elementsDomain = mesh.Elements_Tags(["S1"])

E = np.zeros_like(mesh.groupElem.elements, dtype=float)
v = np.zeros_like(mesh.groupElem.elements, dtype=float)

Affichage.Plot_Nodes(mesh, mesh.Nodes_Tags(["S0"]))
Affichage.Plot_Elements(mesh, mesh.Nodes_Tags(["S1"]))

E[elementsCircle] = 400000
E[elementsDomain] = 210000

v[elementsCircle] = 0.3
v[elementsDomain] = 0.3

comp = Materials.Elas_Isot(2, E, v)

simu = Simulations.Simu_Displacement(mesh, comp)

nodesY0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodesY1 = mesh.Nodes_Conditions(lambda x,y,z: y==1)

simu.add_dirichlet(nodesY0, [0,0], ["x","y"])
# simu.add_dirichlet(nodesY1, [0,-0.01], ["x","y"])
simu.add_surfLoad(nodesY1, [-1], ["y"])

simu.Solve()

Affichage.Plot_Result(simu, "uy", plotMesh=True)
Affichage.Plot_Result(simu, "Syy", plotMesh=True)
Affichage.Plot_Result(simu, E, nodeValues=False, plotMesh=True)

nodesInCircle = mesh.Nodes_Circle(circle)


plt.show()