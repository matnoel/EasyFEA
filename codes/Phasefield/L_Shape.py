import Display
from Interface_Gmsh import Interface_Gmsh
from Geom import Point, PointsList, Line, Domain, Circle, normalize_vect
import Materials
import Simulations
import Folder
import PostTraitement

plt = Display.plt
np = Display.np

pltIter = True
pltLoad = True

makeMovie = False
makeParaview = False

doSimu = True

Display.Clear()

dim = 2

name = "L_Shape_Benchmark" if dim == 2 else "L_Shape_Benchmark_3D"

folder = Folder.New_File(name, results=True)

# ----------------------------------------------
# Config
# ----------------------------------------------

test = True
optimMesh = True

L = 250
ep = 100

# l0 = 10
l0 = 5

split = "Zhang"
regu = "AT2"

tolConv = 1e-0
convOption = 2

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
    mesh = Interface_Gmsh().Mesh_2D(contour, [], "TRI3", refineGeom=refineDomain)
    directions = ["x","y"]
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, [], [0,0,ep], 3, "HEXA8", refineGeom=refineDomain)
    directions = ["x","y","z"]

Display.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

nodesEnca = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodesLoad = mesh.Nodes_Conditions(lambda x,y,z: (y==L) & (x>=2*L-30))
node3 = mesh.Nodes_Point(p3); node4 = mesh.Nodes_Point(p4)
nodesCircle = mesh.Nodes_Cylindre(circle, [0,0,ep])

ddlsY_Load = Simulations.BoundaryCondition.Get_dofs_nodes(dim, "displacement", nodesLoad, ['y'])

# ----------------------------------------------
# Material
# ----------------------------------------------
E = 2e4 # MPa
v = 0.18
Gc = 130 # J/m2
Gc *= 1000/1e6

comportement = Materials.Elas_Isot(dim, E, v, True, ep)

pfm = Materials.PhaseField_Model(comportement, split, regu, Gc, l0)

folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "CP", tolConv, "", test, optimMesh, nL=nL)

if doSimu:

    simu = Simulations.Simu_PhaseField(mesh, pfm)

    def Add_Dep(x,y,z):
        """Fonction qui projete le déplacement dans la bonne direction"""
        
        # récupère le déplacement
        dep = simu.Resultats_matrice_displacement()
        # nouvelles coordonées du maillage
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

    uMax = 1.2 # mm

    inc0 = 1.2/200
    inc1 = inc0/2

    ud = - inc0

    iter = -1

    displacement = []
    load = []

    while ud <= uMax:

        iter += 1

        ud += inc0 if simu.damage.max() < 0.6 else inc1

        simu.Bc_Init()
        simu.add_dirichlet(nodesCircle, [0], ['d'], "damage")
        simu.add_dirichlet(nodesEnca, [0]*dim, directions)       
        
        simu.add_dirichlet(nodesLoad, [ud], ['y'])
        # simu.add_dirichlet(nodesLoad, [loadX, loadY], ['x','y'])

        # Affichage.Plot_BoundaryConditions(simu)        

        u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

        fr = np.sum(Kglob[ddlsY_Load,:] @ u)

        displacement.append(ud)
        load.append(fr)

        simu.Resultats_Set_Resume_Iteration(iter, ud, "mm", ud/uMax, True)

        simu.Save_Iteration()

        if pltIter:
            plt.figure(axIter.figure)
            cb.remove()
            cb = Display.Plot_Result(simu, 'damage', ax=axIter)[2]
            plt.pause(1e-12)

            plt.figure(axLoad.figure)
            axLoad.scatter(ud, fr/1000, c='black')            
            plt.pause(1e-12)

        if not convergence:
            break

    displacement = np.array(displacement)
    load = np.array(load)

    PostTraitement.Save_Load_Displacement(load, displacement, folderSimu)

    simu.Save(folderSimu)

    PostTraitement.Tic.Plot_History(folderSimu, True)    

else:

    simu = Simulations.Load_Simu(folderSimu)
    mesh = simu.mesh

load, displacement = PostTraitement.Load_Load_Displacement(folderSimu)

# ----------------------------------------------
# PostTraitement
# ----------------------------------------------

Display.Plot_BoundaryConditions(simu, folderSimu)

Display.Plot_Result(simu, 'damage', folder=folderSimu)


axLoad = plt.subplots()[1]
axLoad.set_xlabel('displacement [mm]')
axLoad.set_ylabel('load [kN]')
axLoad.plot(displacement, load/1000, c="blue")
Display.Save_fig(folderSimu, "forcedep")

Display.Plot_ResumeIter(simu, folderSimu)

if makeMovie:
    depMax = simu.Get_Resultat("amplitude").max()
    facteur = 10*depMax
    PostTraitement.Make_Movie(folderSimu, 'damage', simu, deformation=True, facteurDef=facteur, plotMesh=False)

if makeParaview:
    PostTraitement.Make_Paraview(folderSimu, simu)
    

plt.show()

pass