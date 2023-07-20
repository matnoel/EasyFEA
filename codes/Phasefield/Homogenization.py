from Interface_Gmsh import Interface_Gmsh
import Display
import Materials
import Simulations
import TicTac
import BoundaryConditions
import Geom

np = Display.np
plt = Display.plt

# CONFIGURATION

useLagrange = False
test = False

pltIter = True

L = 1

E = 1
v = 0.3

Gc = 1
l_0 = L/20

meshSize = l_0/2 if test else l_0/6

# MESH

pt1 = Geom.Point(-L/2, -L/2)
pt2 = Geom.Point(L/2, -L/2)
pt3 = Geom.Point(L/2, L/2)
pt4 = Geom.Point(-L/2, L/2)
points = Geom.PointsList([pt1, pt2, pt3, pt4], meshSize)

circle = Geom.Circle(Geom.Point(), L*0.5, meshSize, isCreux=True)

gmshInterface = Interface_Gmsh()

mesh = gmshInterface.Mesh_2D(points, [circle], "TRI3")

Display.Plot_Model(mesh)
ax = Display.Plot_Mesh(mesh)

noeudsCoins = mesh.Nodes_Tags(["P0", "P1", "P2", "P3"])
noeudsBords = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

listnodes0 = []
listnodes1 = []

for line1, line2 in zip(["L0", "L1"], ["L2", "L3"]):

    nodes0 = mesh.Nodes_Tags([line1])
    nodes1 = mesh.Nodes_Tags([line2])

    if line1 == "L1":
        nodes0 = nodes0[np.argsort(mesh.coordo[nodes0, 0])]
        nodes1 = nodes1[np.argsort(mesh.coordo[nodes1, 0])]
    elif line1 == "L2":
        nodes0 = nodes0[np.argsort(mesh.coordo[nodes0, 1])]
        nodes1 = nodes1[np.argsort(mesh.coordo[nodes1, 1])]    

    nodes0 = nodes0[1:-1]
    nodes1 = nodes1[1:-1]

    # [ax.scatter(mesh.coordo[[n1, n2], 0], mesh.coordo[[n1, n2], 1]) for n1, n2 in zip(nodes1, nodes2)]

    listnodes0.extend(nodes0)
    listnodes1.extend(nodes1)

# SIMU

comp = Materials.Elas_Isot(2, E, v, False, 1)
pfm = Materials.PhaseField_Model(comp, "Zhang", "AT2", Gc, l_0)
simu = Simulations.Simu_PhaseField(mesh, pfm)

r2 = np.sqrt(2)
coefMax = 5
E11 = np.array([[1, 0],[0, 0]]) * coefMax
E22 = np.array([[0, 0],[0, 1]]) * coefMax
E12 = np.array([[0, 1/r2],[1/r2, 0]]) * coefMax*2

axLoad = plt.subplots()[1]

def Add_Lagrange(Eps: np.ndarray, n0, n1):
    nodes = np.array([n0, n1])
    for direction in ["x", "y"]:
        ddls = BoundaryConditions.BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes, [direction])                   
        
        values = Eps @ [mesh.coordo[n0,0]-mesh.coordo[n1,0], mesh.coordo[n0,1]-mesh.coordo[n1,1]]
        value = values[0] if direction == "x" else values[1]

        condition = BoundaryConditions.LagrangeCondition("displacement", nodes, ddls, [direction], [value], [1, -1])
        simu._Bc_Add_Lagrange(condition)

def Solve(Eij: np.ndarray):

    coefs = np.linspace(0, 1.2, 60)

    if pltIter:
        figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True)

    for i, coef in enumerate(coefs):

        simu.Bc_Init()

        Eps = Eij * coef

        func_ux = lambda x,y,z: (Eps.dot([x, y]))[0]
        func_uy = lambda x,y,z: (Eps.dot([x, y]))[1]

        if useLagrange:
            simu.add_dirichlet(noeudsCoins, [func_ux, func_uy], ["x","y"], description="KUBC")
            
            [Add_Lagrange(Eps, n0, n1) for n0, n1 in zip(listnodes0, listnodes1)]
            lagrangeConditions = simu.Bc_Lagrange

        else:
            simu.add_dirichlet(noeudsBords, [func_ux, func_uy], ["x","y"], description="KUBC")

        u, d, Kglob, convergence = simu.Solve(tolConv=1e-0, maxIter=50)

        simu.Save_Iteration()

        Sxy_e = simu.Get_Resultat("Sxy", nodeValues=False)

        Sxy = np.mean(Sxy_e)

        axLoad.scatter(coef, Sxy, c='black')
        plt.figure(axLoad.figure)
        plt.pause(1e-12)


        simu.Resultats_Set_Resume_Iteration(i, 0, "µm", i/len(coefs), True)

        if pltIter:
            cb.remove()
            figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)
            axIter.set_title(f"{i} {axIter.get_title()}")

            plt.figure(figIter)

            plt.pause(1e-12)

        print(i, end='\r')

        if True in (d[noeudsBords] >= 0.98): break




    pass



    
Solve(E12)

TicTac.Tic().Plot_History()

plt.show()