"""Hydraulic dam"""

import Folder
import Display
from Interface_Gmsh import Interface_Gmsh, ElemType, Point, PointsList, Circle, Domain
import Simulations
import Materials

import matplotlib.pyplot as plt
import numpy as np

Display.Clear()

# Define dimension and mesh size parameters
dim = 2
N = 20 if dim == 2 else 10

coef = 1e6
E = 15000*coef  # Pa (Young's modulus)
v = 0.25          # Poisson's ratio

g = 9.81   # m/s^2 (acceleration due to gravity)
ro = 2400  # kg/m^3 (density)
w = 1000   # kg/m^3 (density)

h = 180  # m (thickness)
thickness = 2*h

# --------------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------------

pt1 = Point()
pt2 = Point(x=h)
pt3 = Point(y=h)
contour = PointsList([pt1, pt2, pt3], h/N)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, [], ElemType.TRI6)
    print(f"err area = {np.abs(mesh.area - h**2/2):.3e}")
elif dim == 3:
    mesh = Interface_Gmsh().Mesh_3D(contour, [], [0, 0, -thickness], 3, ElemType.PRISM15)
    print(f"error volume = {np.abs(mesh.volume - h**2/2 * thickness):.3e}")

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)

# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------

material = Materials.Elas_Isot(dim, E, v, planeStress=False, thickness=thickness)
simu = Simulations.Simu_Displacement(mesh, material)

simu.add_dirichlet(nodesY0, [0]*dim, simu.Get_directions())
simu.add_surfLoad(nodesX0, [lambda x, y, z: w*g*(h - y)], ["x"], description="[w*g*(h-y)]")
simu.add_volumeLoad(mesh.nodes, [-ro*g], ["y"], description="[-ro*g]")

sol = simu.Solve()
simu.Save_Iter()

# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------
print(simu)

Display.Plot_Model(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Mesh(simu, h/10/np.abs(sol.max()))
Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1/coef, nColors=20)

Simulations.Tic.Plot_History(details=True)

plt.show()