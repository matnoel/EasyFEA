import Display
from Interface_Gmsh import Interface_Gmsh, GroupElem, ElemType
from Geom import Point, Line, Circle, PointsList, Domain, Contour
import Simulations
import Materials

Display.Clear()

L = 1
openCrack = True

contour = Domain(Point(), Point(L, L))

def DoMesh(dim, elemType):
    if dim == 2:
        mesh = Interface_Gmsh().Mesh_2D(contour, [], elemType, cracks)
    elif dim == 3:
        # WARNING :
        # 2D cracks only works with TETRA4 and TETRA10
        # 2D cracks only works with nLayers = 1
        nLayers = 1
        mesh = Interface_Gmsh().Mesh_3D(contour, [], [0, 0, L], nLayers, elemType, cracks)

    material = Materials.Elas_Isot(dim)
    simu = Simulations.Simu_Displacement(mesh, material)

    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==0), [0]*dim, simu.Get_directions())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==L), [3e-2], ['y'])
    simu.Solve()
    Display.Plot_Result(simu, 'uy', True, 1, plotMesh=True)

    return mesh

crack1 = Line(Point(L/4,L/2), Point(3*L/4,L/2), isOpen=openCrack)
crack2 = Line(Point(0,L/3, isOpen=openCrack), Point(L/2,L/3), isOpen=openCrack)
crack3 = Line(Point(0,2*L/3, isOpen=openCrack), Point(L/2,2*L/3), isOpen=openCrack)
crack4 = Line(Point(0,4*L/5), Point(L,4*L/5), isOpen=False)
cracks = [crack1, crack2, crack3, crack4]
meshes2D = [DoMesh(2, elemType) for elemType in GroupElem.get_Types2D()]
Display.Plot_Model(meshes2D[0], alpha=0)

line1 = Line(Point(L/4, L/2), Point(3*L/4, L/2), isOpen=openCrack)
line2 = Line(line1.pt2, line1.pt2+[0,0.08,L])
line3 = Line(line2.pt2, line1.pt1+[0,0.08,L], isOpen=openCrack)
line4 = Line(line3.pt2, line1.pt1)
cracks = [Contour([line1, line2, line3, line4], openCrack)]
meshes3D = [DoMesh(3, elemType) for elemType in [ElemType.TETRA4, ElemType.TETRA10]]
Display.Plot_Model(meshes3D[0], alpha=0)

Display.plt.show()