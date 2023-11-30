import Display
import Materials
import Simulations
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, Line, Circle, CircleArc, Contour
import Folder
np = Display.np

Display.Clear()

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
dim = 2
isSymmetric = True

folder = Folder.New_File(f"PressurizedTube{dim}D", results=True)

openCrack = True

r = 10
e = 5

sig = 5 # bar
sig *= 1e-1 # 1 bar = 0.1 MPa 

meshSize = e/5
thickness = 100

# --------------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------------
center = Point()
if isSymmetric:
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
else:    
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

# --------------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------------
material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=False, thickness=thickness)
simu = Simulations.Simu_Displacement(mesh, material)

if isSymmetric:
    nodes_x0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
    simu.add_dirichlet(nodes_x0, [0], ['x'])
    simu.add_dirichlet(nodes_y0, [0], ['y'])

def Eval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Evaluation of the sig vect_n function\n
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

EvalX = lambda x,y,z: Eval(x,y,z)[:,:,0]
EvalY = lambda x,y,z: Eval(x,y,z)[:,:,1]
nodes_load = mesh.Nodes_Cylinder(Circle(center, r*2), extrude)
simu.add_surfLoad(nodes_load, [EvalX, EvalY], ['x','y'])

simu.Solve()
simu.Save_Iter()

# --------------------------------------------------------------------------------------------
# PostProcessing
# --------------------------------------------------------------------------------------------
factorDef = r/5 / simu.Result('amplitude').max()
# factorDef = 1
Display.Plot_BoundaryConditions(simu)
Display.Plot_Mesh(simu, deformFactor=factorDef)
Display.Plot_Result(simu, 'ux', nColors=10, nodeValues=True)
Display.Plot_Result(simu, 'uy', nColors=10, nodeValues=True)
Display.Plot_Result(simu, 'Svm', nColors=10, nodeValues=True, deformFactor=factorDef, plotMesh=True)

print(simu)

Display.plt.show()