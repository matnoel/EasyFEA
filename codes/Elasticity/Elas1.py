"""Bending beam"""

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


# Define material properties
E = 210000  # MPa (Young's modulus)
v = 0.3     # Poisson's ratio
coef = 1

L = 120 # mm
h = 13
load = 800

# --------------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------------

meshSize = h/10

domain = Domain(Point(), Point(L,h), meshSize)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(domain, [], ElemType.QUAD4, isOrganised=True)
else:
    mesh = Interface_Gmsh().Mesh_3D(domain, [], [0,0,-h], 4, ElemType.PRISM6, isOrganised=True)

nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------

material = Materials.Elas_Isot(dim, E, v, planeStress=True, thickness=h)
simu = Simulations.Simu_Displacement(mesh, material)

simu.add_dirichlet(nodesX0, [0]*dim, simu.Get_directions())
# # is the same as
# if dim == 2:
#     simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])    
# else:
#     simu.add_dirichlet(nodesX0, [0, 0, 0], ["x", "y", "z"]) 

simu.add_surfLoad(nodesXL, [-load/h**2], ["y"])

sol = simu.Solve()
simu.Save_Iter()

# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------
print(simu)

Display.Plot_Model(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Mesh(simu, h/2/np.abs(sol).max())
Display.Plot_Result(simu, "uy", nodeValues=True, coef=1/coef, nColors=20)

Simulations.Tic.Plot_History(details=False)

plt.show()