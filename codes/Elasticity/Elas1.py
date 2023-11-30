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
dim = 3
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

import gmsh
import sys

gmsh.initialize(sys.argv)



def Set_to_elements(values: np.ndarray):

    values_e: np.ndarray = values[mesh.connect]

    if len(values_e.shape) == 3:
        values_e = np.transpose(values_e, (0,2,1))

    return values_e.reshape((mesh.Ne, -1))

elements_e = Set_to_elements(mesh.coordo)

all_e = elements_e.copy()

for i in range(simu.Niter):

    simu.Set_Iter(i)

    values_e = Set_to_elements(simu.Result('uy'))

    all_e = np.concatenate((all_e, values_e), 1)

def GetType(elemType: str):
    # point "P", line "L", triangle "T", quadrangle
    # "Q", tetrahedron "S", hexahedron "H", prism "I" and pyramid "Y"). (See `x4.py'
    # for a tutorial on model-based views.)
    if 'POINT' in elemType:
        return 'P'
    elif 'SEG' in elemType:
        return 'L'
    elif 'TRI' in elemType:
        return 'T'
    elif 'QUAD' in elemType:
        return 'Q'
    elif 'TETRA' in elemType:
        return 'S'
    elif 'HEXA' in elemType:
        return 'H'
    elif 'PRISM' in elemType:
        return 'I'

view = gmsh.view.add("Svm")
gmsh.view.addListData(view, f"S{GetType(mesh.elemType)}", mesh.Ne, list(np.reshape(all_e, -1)))
gmsh.view.option.setNumber(view, "IntervalsType", 3)
gmsh.view.option.setNumber(view, "NbIso", 10)


gmsh.view.addModelData


rgb = (0,0,0)
gmsh.view.option.setColor(view, 'Triangles', *rgb)
gmsh.view.option.setColor(view, 'Quadrangles', *rgb)
gmsh.view.option.setColor(view, 'Tetrahedra', *rgb)
gmsh.view.option.setColor(view, 'Hexahedra', *rgb)
gmsh.view.option.setColor(view, 'Prisms', *rgb)
 
 
 
 
gmsh.view.option.setNumber(view, "ShowElement", 1)
 
 
gmsh.view.option.setNumber(view, "Axes", 1) # 0 5
# gmsh.view.option.setColor()






gmsh.fltk.run()


pass














Display.Plot_Model(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Mesh(simu, h/2/np.abs(sol).max())
Display.Plot_Result(simu, "uy", nodeValues=True, coef=1/coef, nColors=20)

Simulations.Tic.Plot_History(details=False)

plt.show()