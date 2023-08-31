import Display
from Interface_Gmsh import Interface_Gmsh, GroupElem, ElemType
from Geom import Point, Line, Circle, PointsList, Domain, Contour
import Simulations
import Materials

Display.Clear()

L = 1
meshSize = L/5
openCrack = True

contour = Domain(Point(), Point(L, L), meshSize)
circle = Circle(Point(L/2,L/2), L/3, meshSize, False)
inclusions = [circle]

refine1 = Domain(Point(0, L), Point(L, L*0.8), meshSize/5)
# refine2 = Circle(Point(L/5,L/5),L/2, meshSize_2)
refine2 = Circle(circle.center, L/2, meshSize/7)
refine3 = Circle(Point(L/10,L/10), L/2, meshSize/10)
refineGeom = [refine1, refine2, refine3]

def DoMesh(dim, elemType):
    if dim == 2:
        mesh = Interface_Gmsh().Mesh_2D(contour, inclusions, elemType, refineGeoms=[refineGeom])
    elif dim == 3:
        nLayers = 3
        mesh = Interface_Gmsh().Mesh_3D(contour, inclusions, [0, 0, -L], nLayers, elemType, refineGeoms=[refineGeom])

    Display.Plot_Mesh(mesh)

    material = Materials.Elas_Isot(dim)
    simu = Simulations.Simu_Displacement(mesh, material)

    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==0), [0]*dim, simu.Get_directions())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==L), [3e-2], ['y'])
    simu.Solve()
    Display.Plot_Result(simu, 'uy', True, 1, plotMesh=True)

    pass

# [DoMesh(2, elemType) for elemType in GroupElem.get_Types2D()]

# [DoMesh(3, elemType) for elemType in [ElemType.TETRA4, ElemType.TETRA10]]
[DoMesh(3, elemType) for elemType in [ElemType.PRISM6]]

Display.plt.show()