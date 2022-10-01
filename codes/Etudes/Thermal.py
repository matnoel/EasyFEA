import matplotlib.pyplot as plt
import Affichage
import Interface_Gmsh
from Geom import Circle, Domain, Line, Point
import Materials
import Simu

Affichage.Clear()

dim=3

a = 1
if dim == 2:
    domain = Domain(Point(), Point(a, a))
else:
    domain = Domain(Point(), Point(a, a))
circle = Circle(Point(a/2, a/2), diam=a/3, isCreux=False, taille=a/50)
interfaceGmsh = Interface_Gmsh.Interface_Gmsh(False, False, True)

if dim == 2:
    # mesh = interfaceGmsh.Rectangle_2D(domain, "QUAD4")
    mesh = interfaceGmsh.PlaqueAvecCercle2D(domain, circle, "TRI6")
else:
    mesh = interfaceGmsh.PlaqueAvecCercle3D(domain, circle, [0,0,a], 4)

thermalModel = Materials.ThermalModel(dim=dim, k=1, ep=1)

materiau = Materials.Materiau(thermalModel, verbosity=False)

simu = Simu.Simu(mesh , materiau, False)

noeuds0 = mesh.Nodes_Conditions(lambda x: x == 0)
noeudsL = mesh.Nodes_Conditions(lambda x: x == a)
if dim == 2:
    noeudsCircle = mesh.Nodes_Circle(circle)
else:
    noeudsCircle = mesh.Nodes_Cylindre(circle, [0,0,a])

simu.add_dirichlet("thermal", noeuds0, [0], [""])
simu.add_dirichlet("thermal", noeudsL, [1], [""])

# simu.add_volumeLoad("thermal", noeudsCircle, [1], [""])

simu.Assemblage_t()

t = simu.Solve_t()
# Affichage.Plot_NoeudsMaillage(mesh, noeuds=noeudsCircle)
Affichage.Plot_ElementsMaillage(mesh, noeuds=noeudsCircle, dimElem=3)
Affichage.Plot_Result(simu, "thermal", affichageMaillage=True, valeursAuxNoeuds=True)

if dim == 3:
    print(f"Volume : {mesh.volume:.3}")

plt.show()

pass