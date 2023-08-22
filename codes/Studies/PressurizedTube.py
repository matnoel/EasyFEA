import numpy as np

import Display
import Materials
import Simulations
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, Line, Circle, CircleArc, Contour
import Folder

Display.Clear()

dim = 2
model = 1 # symmetric
# model = 2 # total

folder = Folder.New_File(f"PressurizedTube{dim}D", results=True)

openCrack = True

r = 10
e = 5

sig = 5 # bar
sig *= 1e-1 # 1 bar = 0.1 MPa 

meshSize = e/5
thickness = 100

center = Point()

if model == 1:    
    p1 = Point(r,0)
    p2 = Point(e+r,0)
    p3 = Point(0,e+r)
    p4 = Point(0,r)

    line1 = Line(p1, p2, meshSize)
    line2 = CircleArc(p2, center, p3, meshSize)
    line3 = Line(p3, p4, meshSize)
    line4 = CircleArc(p4, center, p1, meshSize)

    contour = Contour([line1, line2, line3, line4])
    inclusions = []
elif model == 2:
    
    contour = Circle(center, (r+e)*2, meshSize)
    inclusions = [Circle(center, 2*r, meshSize, isHollow=True)]

extrude = [0,0,-thickness]

l = e/4
p = r+e/2
alpha = np.pi/3
pc1 = Point((p-l/2)*np.cos(alpha), (p-l/2)*np.sin(alpha))
pc2 = Point((p+l/2)*np.cos(alpha), (p+l/2)*np.sin(alpha))

crack = Line(pc1, pc2, meshSize/6, isOpen=openCrack)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, inclusions, elemType=ElemType.TRI6, cracks=[crack])
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, inclusions, extrude, 10, ElemType.PRISM6, cracks=[crack])

material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=False, thickness=thickness)

simu = Simulations.Simu_Displacement(mesh, material)

if model == 1:
    nodes_x0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
    simu.add_dirichlet(nodes_x0, [0], ['x'])
    simu.add_dirichlet(nodes_y0, [0], ['y'])

nodes_load = mesh.Nodes_Cylinder(Circle(center, r*2), extrude)

def FuncEval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Evaluation de la fonction sig vect_n\n
    x,y,z (ep)"""

    # Gauss point coordinates in form (Ne, nPg, 3)
    coord = np.zeros((x.shape[0], x.shape[1], 3))
    coord[:,:,0] = x
    coord[:,:,1] = y
    coord[:,:,2] = 0

    # Construction of the normal vector to the inner surface
    vect = coord - center.coordo
    vectN = np.einsum('npi,np->npi', vect, 1/np.linalg.norm(vect, axis=2))
   
    loads = sig * vectN

    return loads

funcEvalX = lambda x,y,z: FuncEval(x,y,z)[:,:,0]
funcEvalY = lambda x,y,z: FuncEval(x,y,z)[:,:,1]

simu.add_surfLoad(nodes_load, [funcEvalX, funcEvalY], ['x','y'])

simu.Solve()
simu.Save_Iter()

factorDef = r/5 / simu.Get_Result('amplitude').max()

# Display.Plot_Model(mesh, alpha=0)
Display.Plot_BoundaryConditions(simu)

Display.Plot_Result(simu, 'Svm', nColors=10, nodeValues=True, deformation=True, factorDef=factorDef, plotMesh=True)
Display.Plot_Result(simu, 'ux', nColors=10, nodeValues=True)
Display.Plot_Result(simu, 'uy', nColors=10, nodeValues=True)

print(simu)

Display.plt.show()