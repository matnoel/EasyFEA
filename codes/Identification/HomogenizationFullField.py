from Interface_Gmsh import Interface_Gmsh, ElemType
import Geom
import Display
import Simulations
import BoundaryCondition
import Materials

np = Materials.np
plt = Display.plt

Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
# use Periodic boundary conditions ?
usePER = True

L = 120 # mm
h = 13
b = 13

nL = 40 # number of inclusions following x
nH = 4 # number of inclusions following y
isHollow = True # hollow inclusions

# c = 13/2
cL = L/(2*nL)
cH = h/(2*nH)

E = 210000
v = 0.3

load = 800

# ----------------------------------------------
# Mesh
# ----------------------------------------------
elemType = ElemType.TRI3
meshSize = h/20

pt1 = Geom.Point()
pt2 = Geom.Point(L,0)
pt3 = Geom.Point(L,h)
pt4 = Geom.Point(0,h)

domain = Geom.Domain(pt1, pt2, meshSize)

inclusions = []
for i in range(nL):
    x = cL + cL*(2*i)
    for j in range(nH):
        y = cH + cH*(2*j)

        ptd1 = Geom.Point(x-cL/2, y-cH/2)
        ptd2 = Geom.Point(x+cL/2, y+cH/2)

        inclusion = Geom.Domain(ptd1, ptd2, meshSize, isHollow)

        inclusions.append(inclusion)

interfaceGmsh = Interface_Gmsh(False)

inclusion = Geom.Domain(ptd1, ptd2, meshSize)
area_inclusion = interfaceGmsh.Mesh_2D(inclusion).area

points = Geom.PointsList([pt1, pt2, pt3, pt4], meshSize)

# mesh with inclusions
mesh_inclusions = interfaceGmsh.Mesh_2D(points, inclusions, elemType)

# mesh without inclusions
mesh = interfaceGmsh.Mesh_2D(points, [], elemType)

ptI1 = Geom.Point(-cL,-cH)
ptI2 = Geom.Point(cL,-cH)
ptI3 = Geom.Point(cL, cH)
ptI4 = Geom.Point(-cL, cH)

pointsI = Geom.PointsList([ptI1, ptI2, ptI3, ptI4], meshSize/4)

mesh_VER = interfaceGmsh.Mesh_2D(pointsI, [Geom.Domain(Geom.Point(-cL/2,-cH/2), Geom.Point(cL/2, cH/2),
                                                       meshSize/4, isHollow)], elemType)
area_VER = mesh_VER.area

Display.Plot_Mesh(mesh_inclusions)
Display.Plot_Mesh(mesh)
axVer = Display.Plot_Mesh(mesh_VER)

n1 = mesh_VER.Nodes_Point(ptI1)
n2 = mesh_VER.Nodes_Point(ptI2)
n3 = mesh_VER.Nodes_Point(ptI3)
n4 = mesh_VER.Nodes_Point(ptI4)

coins = [mesh_VER.coordo[n].reshape(-1) for n in [n1, n2, n3, n4]]

calc_vect = lambda n0, n1: (mesh_VER.coordo[n1, :] - mesh_VER.coordo[n0, :]).reshape(-1)
vect_i = np.array([calc_vect(n1, n2), calc_vect(n2, n3), calc_vect(n3, n4), calc_vect(n4, n1)])

nodes_edges = []

for l, line in enumerate(["L0", "L1", "L2", "L3"]):

    nodes = mesh_VER.Nodes_Tags([line])

    vect_j = mesh_VER.coordo[nodes] - coins[l]

    proj = vect_j @ vect_i[l]

    idxSort = np.argsort(proj)

    nodes = nodes[idxSort]

    if l >= 2:
        nodes = nodes[::-1]

    nodes = nodes[1:-1]

    # [axVer.scatter(mesh_VER.coordo[n, 0], mesh_VER.coordo[n, 1], c='black') for n in nodes]

    nodes_edges.append(nodes)

list_pairedNodes = []

# pour chaque pair de bord on assemble dans un tuple
for p in range(2):

    pairedNodes = (nodes_edges[p], nodes_edges[p+1*2])

    # [axVer.scatter(mesh_VER.coordo[n, 0], mesh_VER.coordo[n, 1], c='black') for n in pairedNodes[0]]
    # [axVer.scatter(mesh_VER.coordo[n, 0], mesh_VER.coordo[n, 1], c='black') for n in pairedNodes[1]]

    list_pairedNodes.append(pairedNodes)

# plt.show()

# ----------------------------------------------
# Material
# ----------------------------------------------

# elastic behavior of the beam
material_inclsuion = Materials.Elas_Isot(2, E=E, v=v, planeStress=True, thickness=b)
CMandel = material_inclsuion.C

material = Materials.Elas_Anisot(2, CMandel, np.array([1,0,0]), useVoigtNotation=False)
testC = np.linalg.norm(material_inclsuion.C-material.C)/np.linalg.norm(material_inclsuion.C)
assert testC < 1e-12, "the matrices are different"

# ----------------------------------------------
# Homogenization
# ----------------------------------------------
simu_inclusions = Simulations.Simu_Displacement(mesh_inclusions, material_inclsuion)
simu_VER = Simulations.Simu_Displacement(mesh_VER, material_inclsuion)
simu = Simulations.Simu_Displacement(mesh, material)

r2 = np.sqrt(2)
E11 = np.array([[1, 0],[0, 0]])
E22 = np.array([[0, 0],[0, 1]])
E12 = np.array([[0, 1/r2],[1/r2, 0]])

if usePER:
    nodes_border = mesh_VER.Nodes_Tags(["P0","P1","P2","P3"])
else:
    nodes_border = mesh_VER.Nodes_Tags(["L0", "L1", "L2", "L3"])

def Calc_ukl(Ekl: np.ndarray):

    simu_VER.Bc_Init()

    func_ux = lambda x, y, z: Ekl.dot([x, y])[0]
    func_uy = lambda x, y, z: Ekl.dot([x, y])[1]
    simu_VER.add_dirichlet(nodes_border, [func_ux, func_uy], ["x","y"])

    if usePER:

        for pairedNodes in list_pairedNodes:

            for n0, n1 in zip(pairedNodes[0], pairedNodes[1]):
                
                nodes = np.array([n0, n1])

                axVer.scatter(mesh_VER.coordo[nodes, 0],mesh_VER.coordo[nodes, 1], marker='+', c='red')

                for direction in ["x", "y"]:
                    dofs = BoundaryCondition.BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes, [direction])                   
                    
                    values = Ekl @ [mesh_VER.coordo[n0,0]-mesh_VER.coordo[n1,0], mesh_VER.coordo[n0,1]-mesh_VER.coordo[n1,1]]
                    value = values[0] if direction == "x" else values[1]

                    # value = 0

                    condition = BoundaryCondition.LagrangeCondition("displacement", nodes, dofs, [direction], [value], [1, -1])
                    simu_VER._Bc_Add_Lagrange(condition)

    ukl = simu_VER.Solve()

    simu_VER.Save_Iter()

    # Display.Plot_Result(simu_VER, "ux", deformation=False)
    # Display.Plot_Result(simu_VER, "uy", deformation=False)
    # Display.Plot_Result(simu_VER, "Exx")
    # Display.Plot_Result(simu_VER, "Eyy")
    # Display.Plot_Result(simu_VER, "Exy")

    return ukl

u11 = Calc_ukl(E11)
u22 = Calc_ukl(E22)
u12 = Calc_ukl(E12)

u11_e = mesh_VER.Locates_sol_e(u11)
u22_e = mesh_VER.Locates_sol_e(u22)
u12_e = mesh_VER.Locates_sol_e(u12)

U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

matrixType = "rigi"
jacobien_e_pg = mesh_VER.Get_jacobian_e_pg(matrixType)
poids_pg = mesh_VER.Get_weight_pg(matrixType)
B_e_pg = mesh_VER.Get_B_e_pg(matrixType)

C_hom = np.einsum('ep,p,ij,epjk,ekl->il', jacobien_e_pg, poids_pg, CMandel, B_e_pg, U_e, optimize='optimal') * 1/mesh_VER.area

if isHollow:
    coef = (1 - area_inclusion/area_VER)
    C_hom *= coef

# print(np.linalg.eigvals(C_hom))

# ----------------------------------------------
# Comparison
# ----------------------------------------------
def Simulation(simu: Simulations._Simu, title=""):

    simu.Bc_Init()

    simu.Need_Update()

    simu.add_dirichlet(simu.mesh.Nodes_Tags(['L3']), [0,0], ['x', 'y'])
    simu.add_surfLoad(simu.mesh.Nodes_Tags(['L1']), [-load/(b*h)], ['y'])

    simu.Solve()

    # Display.Plot_BoundaryConditions(simu)
    Display.Plot_Result(simu, "uy", title=f"{title} uy")
    # Display.Plot_Result(simu, "Eyy")

    print(f"{title}: dy = {np.max(simu.Get_Result('uy')[simu.mesh.Nodes_Point(Geom.Point(L,0))]):.3f}")

Simulation(simu_inclusions, "inclusions")
Simulation(simu, "non hom")

testSym = np.linalg.norm(C_hom.T - C_hom)/np.linalg.norm(C_hom)

if testSym >= 1e-12 and testSym <= 1e-7:
    C_hom = 1/2 * (C_hom.T + C_hom)

material.Set_C(C_hom, False)
Simulation(simu, "hom")

# ax = Display.Plot_Result(simu, "uy")[1]
# Display.Plot_Result(simuInclusions, "uy", ax=ax)

plt.show()