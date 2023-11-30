import Display
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, PointsList, Line, Domain, Circle, normalize_vect
import Materials
import Simulations
import Folder
import PostProcessing

plt = Display.plt
np = Display.np

Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
solve = False
test = True
optimMesh = True

pltIter = True
pltLoad = True
makeMovie = False
makeParaview = False

# geom
dim = 2
L = 250 # mm
ep = 100
l0 = 5

# material
E = 2e4 # MPa
v = 0.18

# phase field
split = "AnisotStress"
regu = "AT2"
Gc = 130 # J/m2
Gc *= 1000/1e6 #mJ/mm2
tolConv = 1e-2
convOption = 2

# loading
adaptLoad = True
uMax = 1.2 # mm
inc0 = 1.2/200
inc1 = inc0/2

# folder
name = "L_Shape_Benchmark"
if dim == 3:
    name += '_3D'
folder = Folder.New_File(name, results=True)

# ----------------------------------------------
# Mesh
# ----------------------------------------------
nL = L//l0

if test:
    hC = l0/2
else:
    hC = 0.5
    # hC = 0.25

p1 = Point()
p2 = Point(L,0)
p3 = Point(L,L)
p4 = Point(2*L-30,L)
p5 = Point(2*L,L)
p6 = Point(2*L,2*L)
p7 = Point(0,2*L)

if optimMesh:
    # hauteur zone rafinée
    h = 100
    refineDomain = Domain(Point(0,L-h/3), Point(L+h/3,L+h), hC)
    hD = hC*5
else:
    refineDomain = None
    hD = hC

contour = PointsList([p1,p2,p3,p4,p5,p6,p7], hD)

circle = Circle(p5, 100)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, [], ElemType.TRI3, refineGeoms=[refineDomain])
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, [], [0,0,-ep], 3, ElemType.HEXA8, refineGeoms=[refineDomain])

Display.Plot_Mesh(mesh)
# Display.Plot_Model(mesh)

nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodes_load = mesh.Nodes_Conditions(lambda x,y,z: (y==L) & (x>=2*L-30))
node3 = mesh.Nodes_Point(p3); node4 = mesh.Nodes_Point(p4)
nodes_circle = mesh.Nodes_Cylinder(circle, [0,0,ep])
nodes_edges = mesh.Nodes_Conditions(lambda x,y,z: (x==0) | (x==L) | (y==L)| (y==0))

# ----------------------------------------------
# Simulation
# ----------------------------------------------
material = Materials.Elas_Isot(dim, E, v, True, ep)
pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0)

folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "CP", tolConv, "", test, optimMesh, nL=nL)

if solve:

    simu = Simulations.Simu_PhaseField(mesh, pfm)
    
    dofsY_load = simu.Bc_dofs_nodes(nodes_load, ['y'])

    if adaptLoad:
        def Add_Dep(x,y,z):
            """Function that projects displacement in the right direction to adapt movement to surface inclination."""
            
            # recovers displacement
            dep = simu.Results_displacement_matrix()
            # new mesh coordinates
            newCoordo = simu.mesh.coordo + dep

            vectBord = newCoordo[node4] - newCoordo[node3]
            vectBord = normalize_vect(vectBord)

            vectDep = np.cross([0,0,1], vectBord)
            vectDep = normalize_vect(vectDep)

            displ = ud * vectDep[0,:2]

            return displ
        
        loadX = lambda x,y,z: Add_Dep(x,y,z)[0]
        loadY = lambda x,y,z: Add_Dep(x,y,z)[1]
    
    if pltIter:
        __, axIter, cb = Display.Plot_Result(simu, 'damage')

        axLoad = plt.subplots()[1]
        axLoad.set_xlabel('displacement [mm]')
        axLoad.set_ylabel('load [kN]')

    displacement = []
    load = []
    ud = - inc0
    iter = -1

    while ud <= uMax:
        
        # update displacement
        iter += 1
        ud += inc0 if simu.damage.max() < 0.6 else inc1
        
        # update boundary conditions
        simu.Bc_Init()
        simu.add_dirichlet(nodes_circle, [0], ['d'], "damage")
        simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_directions())       
        
        if adaptLoad:
            simu.add_dirichlet(nodes_load, [loadX, loadY], ['x','y'])
        else:            
            simu.add_dirichlet(nodes_load, [ud], ['y'])

        # solve
        u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

        # calc load
        fr = np.sum(Kglob[dofsY_load,:] @ u)

        # save load and displacement
        displacement.append(ud)
        load.append(fr)

        # print iter
        simu.Results_Set_Iteration_Summary(iter, ud, "mm", ud/uMax, True)

        # save iteration
        simu.Save_Iter()

        if pltIter:
            plt.figure(axIter.figure)
            cb.remove()
            cb = Display.Plot_Result(simu, 'damage', ax=axIter)[2]
            plt.pause(1e-12)

            plt.figure(axLoad.figure)
            axLoad.scatter(ud, fr/1000, c='black')            
            plt.pause(1e-12)

        if not convergence or np.max(d[nodes_edges]) >= 1:
            # stop simulation if damage occurs on edges or convergence has not been reached
            break
    
    # save load and displacement
    displacement = np.array(displacement)
    load = np.array(load)
    PostProcessing.Save_Load_Displacement(load, displacement, folderSimu)

    # save the simulation
    simu.Save(folderSimu)

    PostProcessing.Tic.Plot_History(folderSimu, True)    

else:

    simu = Simulations.Load_Simu(folderSimu)
    mesh = simu.mesh

load, displacement = PostProcessing.Load_Load_Displacement(folderSimu)

# ----------------------------------------------
# PostProcessing
# ----------------------------------------------
Display.Plot_BoundaryConditions(simu, folderSimu)

Display.Plot_Result(simu, 'damage', folder=folderSimu)


axLoad = plt.subplots()[1]
axLoad.set_xlabel('displacement [mm]')
axLoad.set_ylabel('load [kN]')
axLoad.plot(displacement, load/1000, c="blue")
Display.Save_fig(folderSimu, "forcedep")

Display.Plot_Iter_Summary(simu, folderSimu)

if makeMovie:
    depMax = simu.Result("amplitude").max()
    facteur = 10*depMax
    PostProcessing.Make_Movie(folderSimu, 'damage', simu, deformation=True, factorDef=facteur, plotMesh=False)

if makeParaview:
    PostProcessing.Make_Paraview(folderSimu, simu)

plt.show()