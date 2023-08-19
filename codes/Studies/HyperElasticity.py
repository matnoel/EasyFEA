from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, PointsList
import Display
import Simulations
import Materials
import Folder
import PostProcessing
from Mesh import Get_new_mesh

Display.Clear()

dim = 2
makeParaview = False
useHyperElastic = True # reactualized lagrangian

folder = Folder.New_File(f"HyperElasticity{dim}D", results=True)

L = 250
thickness = 50
w = 50

meshSize = L/20

sigMax = 80000/(w*thickness)
uMax = 50

p1 = Point(0,0)
p2 = Point(L,0)
p3 = Point(L,L, r=50)
p4 = Point(2*L-w,L)
p5 = Point(2*L,L)
p6 = Point(2*L,2*L)
p7 = Point(2*L-w,2*L)
p8 = Point(0,2*L)

contour = PointsList([p1,p2,p3,p4,p5,p6,p7,p8], meshSize)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, [], ElemType.TRI6)
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, [], [0,0,-thickness], 3, ElemType.PRISM6)

nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
# nodes_Load = mesh.Nodes_Conditions(lambda x,y,z: (y==2*L) & (x>=2*L-30))
nodes_Load = mesh.Nodes_Conditions(lambda x,y,z: x==2*L)

material = Materials.Elas_Isot(dim, E=210000, v=0.25, planeStress=True, thickness=thickness)

simu = Simulations.Simu_Displacement(mesh, material)

N = 5
iter = 0

while iter < N:

    iter += 1

    print(f"{iter/N*100:2.2f} %", end='\r')

    simu.Bc_Init()
    simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_directions())
    # simu.add_dirichlet(nodes_Load, [uMax*iter/N], ['y'])
    simu.add_surfLoad(nodes_Load, [sigMax*iter/N], ['y'])

    simu.Solve()

    simu.Save_Iter()

    if useHyperElastic and iter != N:
        # update the nodes coordinates

        newMesh = Get_new_mesh(simu.mesh, simu.Results_displacement_matrix())

        simu.mesh = newMesh

        pass

Display.Plot_Mesh(mesh)
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, 'ux')
Display.Plot_Result(simu, 'uy')
Display.Plot_Result(simu, 'Svm', nodeValues=False)
Display.Plot_Result(simu, 'Evm', nodeValues=False)

print(f"max(Evm) = {simu.Get_Result('Evm').max()*100:.2f} %")

print(simu)

if makeParaview:
    PostProcessing.Make_Paraview(folder, simu, elementsResult=['Strain'])


Display.plt.show()