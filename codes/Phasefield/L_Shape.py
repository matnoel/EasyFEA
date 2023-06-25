import Affichage
from Interface_Gmsh import Interface_Gmsh
from Geom import Point, PointsList, Line, Domain, Circle
import Materials
import Simulations
import Folder
import PostTraitement

plt = Affichage.plt
np = Affichage.np

pltIter = True
doSimu = True

Affichage.Clear()

dim = 3

folder = Folder.New_File(f"L_Shape_Benchmark_{dim}D", results=True)

test = True
optimMesh = True

L = 250
ep = 100

# l0 = 10
l0 = 5

if test:
    hC = l0/2
else:
    hC = 0.5
    hC = 0.25

nL = L/hC

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
    refineDomain = Domain(Point(0,L-h/3), Point(L+h/2,L+h), hC)
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

Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

nodesEnca = mesh.Nodes_Conditions(lambda x,y,z: y==0)
nodesLoad = mesh.Nodes_Conditions(lambda x,y,z: (y==L) & (x>=2*L-30))
# nodesLoad = mesh.Nodes_Point(p4)
nodesCircle = mesh.Nodes_Cylindre(circle, [0,0,ep])

ddlsY_Load = Simulations.BoundaryCondition.Get_ddls_noeuds(dim, "displacement", nodesLoad, ['y'])

E = 2e4 # MPa
v = 0.18
Gc = 130 # J/m2
Gc *= 1000/1e6

tolConv = 1e-0
convOption = 2

comportement = Materials.Elas_Isot(dim, E, v, True, ep)

pfm = Materials.PhaseField_Model(comportement, "Amor", "AT2", Gc, l0)

simu = Simulations.Simu_PhaseField(mesh, pfm)

folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "CP", tolConv, "", test, optimMesh, nL=nL)


if doSimu:

    if pltIter:
        __, axIter, cb = Affichage.Plot_Result(simu, 'damage')

        axLoad = plt.subplots()[1]
        axLoad.set_xlabel('displacement [mm]')
        axLoad.set_ylabel('load [kN]')

    # loads = np.linspace(0,25,500) * 1e3
    # for iter, load in enumerate(loads):

    uMax = 1.2 # mm

    inc0 = 1.2/200
    inc1 = inc0/2

    ud = - inc0

    iter = -1

    while ud <= uMax:

        iter += 1

        ud += inc0 if simu.damage.max() < 0.6 else inc1

        simu.Bc_Init()
        simu.add_dirichlet(nodesCircle, [0], ['d'], "damage")
        simu.add_dirichlet(nodesEnca, [0]*dim, directions)
        simu.add_dirichlet(nodesLoad, [ud], ['y'])

        # Affichage.Plot_BoundaryConditions(simu)

        # simu.add_lineLoad(nodesLoad, [load/30], ['y'])
        # simu.add_neumann(nodePoint, [load], ['y'])

        u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

        fr = np.sum(Kglob[ddlsY_Load,:] @ u)

        simu.Resultats_Set_Resume_Iteration(iter, ud, "mm", ud/uMax, True)

        simu.Save_Iteration()

        if pltIter:
            plt.figure(axIter.figure)
            cb.remove()
            cb = Affichage.Plot_Result(simu, 'damage', ax=axIter)[2]
            plt.pause(1e-12)

            plt.figure(axLoad.figure)
            axLoad.scatter(ud, fr/1000, c='black')            
            plt.pause(1e-12)

        if not convergence:
            break

        pass

    simu.Save(folderSimu)

else:

    simu = Simulations.Load_Simu(folderSimu)
    mesh = simu.mesh


    simu.Update_iter(-1)

    depMax = simu.Get_Resultat("amplitude").max()

    facteur = 10*depMax

    PostTraitement.Make_Movie(folderSimu, 'damage', simu, deformation=True, facteurDef=facteur)


PostTraitement.Make_Paraview(folderSimu, simu)

Affichage.Plot_Result(simu, 'damage', folder=folderSimu)

Affichage.Plot_BoundaryConditions(simu, folderSimu)


plt.show()

pass