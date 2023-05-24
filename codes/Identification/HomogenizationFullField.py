from Interface_Gmsh import Interface_Gmsh
import Geom
import Affichage
import Simulations
import BoundaryCondition
import Materials

np = Materials.np
plt = Affichage.plt

Affichage.Clear()

useLagrange = True

# ----------------------------------------------
# Options
# ----------------------------------------------

L = 120 # mm
h = 13
b = 13

nL = 40 # nombre d'inclusion suivant L
nH = 4 # nombre d'inclusion suivant h

# c = 13/2
cL = L/(2*nL)
cH = h/(2*nH)

E = 210000
v = 0.3

load = 800

# ----------------------------------------------
# Mesh
# ----------------------------------------------
elemType = "QUAD4"
meshSize = h/20

pt1 = Geom.Point()
pt2 = Geom.Point(L,0)
pt3 = Geom.Point(L,h)
pt4 = Geom.Point(0,h)

domain = Geom.Domain(pt1, pt2, meshSize)

listGeomInDomain = []
for i in range(nL):
    x = cL + cL*(2*i)
    for j in range(nH):
        y = cH + cH*(2*j)

        ptd1 = Geom.Point(x-cL/2, y-cH/2)
        ptd2 = Geom.Point(x+cL/2, y+cH/2)

        inclusion = Geom.Domain(ptd1, ptd2, meshSize, isCreux=True)

        listGeomInDomain.append(inclusion)

interfaceGmsh = Interface_Gmsh(False)

inclusion = Geom.Domain(ptd1, ptd2, meshSize, isCreux=True)
surfaceInclu = interfaceGmsh.Mesh_2D(inclusion).aire

points = Geom.PointsList([pt1, pt2, pt3, pt4], meshSize)

# maillage avec les inclusions
meshInclusions = interfaceGmsh.Mesh_2D(points, listGeomInDomain, elemType)

# maillage sans les inclusions
mesh = interfaceGmsh.Mesh_2D(points, [], elemType)

ptI1 = Geom.Point(-cL,-cH)
ptI2 = Geom.Point(cL,-cH)
ptI3 = Geom.Point(cL, cH)
ptI4 = Geom.Point(-cL, cH)

pointsI = Geom.PointsList([ptI1, ptI2, ptI3, ptI4], meshSize/4)

meshVER = interfaceGmsh.Mesh_2D(pointsI, [Geom.Domain(Geom.Point(-cL/2,-cH/2), Geom.Point(cL/2, cH/2), meshSize/4, isCreux=True)], elemType)

surfaceVer = meshVER.aire

coef = (1 - surfaceInclu/surfaceVer)
coef = 1

Affichage.Plot_Mesh(meshInclusions)
Affichage.Plot_Mesh(mesh)
axVer = Affichage.Plot_Mesh(meshVER)

# Verification que les elements 1D font tous la même taille
group1d = meshVER.Get_list_groupElem(1)[0]
l = np.einsum('ep,p->e', group1d.Get_jacobien_e_pg("rigi"), group1d.Get_gauss("rigi").poids)

n1 = meshVER.Nodes_Point(ptI1)
n2 = meshVER.Nodes_Point(ptI2)
n3 = meshVER.Nodes_Point(ptI3)
n4 = meshVER.Nodes_Point(ptI4)

coins = [meshVER.coordo[n].reshape(-1) for n in [n1, n2, n3, n4]]

calc_vect = lambda n0, n1: (meshVER.coordo[n1, :] - meshVER.coordo[n0, :]).reshape(-1)
vect_i = np.array([calc_vect(n1, n2), calc_vect(n2, n3), calc_vect(n3, n4), calc_vect(n4, n1)])

noeuds_bord = []

for l, line in enumerate(["L0", "L1", "L2", "L3"]):

    nodes = meshVER.Nodes_Tags([line])

    vect_j = meshVER.coordo[nodes] - coins[l]

    proj = vect_j @ vect_i[l]

    idxSort = np.argsort(proj)

    nodes = nodes[idxSort]

    if l >= 2:
        nodes = nodes[::-1]

    nodes = nodes[1:-1]

    [axVer.scatter(meshVER.coordo[n, 0], meshVER.coordo[n, 1], c='black') for n in nodes]

    noeuds_bord.append(nodes)

list_pairedNodes = []

# pour chaque pair de bord on assemble dans un tuple
for p in range(2):

    pairedNodes = (noeuds_bord[p], noeuds_bord[p+1*2])

    # [axVer.scatter(meshVER.coordo[n, 0], meshVER.coordo[n, 1], c='black') for n in pairedNodes[0]]
    # [axVer.scatter(meshVER.coordo[n, 0], meshVER.coordo[n, 1], c='black') for n in pairedNodes[1]]

    list_pairedNodes.append(pairedNodes)

# plt.show()

# ----------------------------------------------
# Materiau
# ----------------------------------------------

# comportement élastique de la poutre
compInclusions = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=b)
CMandel = compInclusions.C

comp = Materials.Elas_Anisot(2, CMandel, np.array([1,0,0]), useVoigtNotation=False)
testC = np.linalg.norm(compInclusions.C-comp.C)/np.linalg.norm(compInclusions.C)
assert testC < 1e-12, "les matrices sont différentes"

# ----------------------------------------------
# Homogenization
# ----------------------------------------------

simuInclusions = Simulations.Simu_Displacement(meshInclusions, compInclusions)
simuVER = Simulations.Simu_Displacement(meshVER, compInclusions)
simu = Simulations.Simu_Displacement(mesh, comp)

r2 = np.sqrt(2)
E11 = np.array([[1, 0],[0, 0]])
E22 = np.array([[0, 0],[0, 1]])
E12 = np.array([[0, 1/r2],[1/r2, 0]])

# u11 = np.einsum('ij,nj->ni', E11, meshInclusion.coordo[noeudsDuBord, 0:2])
# u22 = np.einsum('ij,nj->ni', E22, meshInclusion.coordo[noeudsDuBord, 0:2])
# u12 = np.einsum('ij,nj->ni', E12, meshInclusion.coordo[noeudsDuBord, 0:2])

if useLagrange:
    noeudsDuBord = meshVER.Nodes_Tags(["P0","P1","P2","P3"])
else:
    noeudsDuBord = meshVER.Nodes_Tags(["L0", "L1", "L2", "L3"])

def CalcDisplacement(Ekl: np.ndarray):

    simuVER.Bc_Init()    

    simuVER.add_dirichlet(noeudsDuBord, [lambda x, y, z: Ekl.dot([x, y])[0], lambda x, y, z: Ekl.dot([x, y])[1]], ["x","y"])

    if useLagrange:

        for pairedNodes in list_pairedNodes:

            for n0, n1 in zip(pairedNodes[0], pairedNodes[1]):
                
                nodes = np.array([n0, n1])

                axVer.scatter(meshVER.coordo[nodes, 0],meshVER.coordo[nodes, 1], marker='+', c='red')

                for direction in ["x", "y"]:
                    ddls = BoundaryCondition.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, [direction])                   
                    
                    values = Ekl @ [meshVER.coordo[n0,0]-meshVER.coordo[n1,0], meshVER.coordo[n0,1]-meshVER.coordo[n1,1]]
                    value = values[0] if direction == "x" else values[1]

                    # value = 0

                    condition = BoundaryCondition.LagrangeCondition("displacement", nodes, ddls, [direction], [value], [1, -1])
                    simuVER._Bc_Add_Lagrange(condition)

    ukl = simuVER.Solve()

    simuVER.Save_Iteration()

    # Affichage.Plot_Result(simuVER, "ux", deformation=False)
    # Affichage.Plot_Result(simuVER, "uy", deformation=False)

    # Affichage.Plot_Result(simuVER, "Exx")
    # Affichage.Plot_Result(simuVER, "Eyy")
    # Affichage.Plot_Result(simuVER, "Exy")

    return ukl

u11 = CalcDisplacement(E11)
u22 = CalcDisplacement(E22)
u12 = CalcDisplacement(E12)

u11_e = meshVER.Localises_sol_e(u11)
u22_e = meshVER.Localises_sol_e(u22)
u12_e = meshVER.Localises_sol_e(u12)

U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

matriceType = "rigi"
jacobien_e_pg = meshVER.Get_jacobien_e_pg(matriceType)
poids_pg = meshVER.Get_poid_pg(matriceType)
B_e_pg = meshVER.Get_B_dep_e_pg(matriceType)

C_hom = np.einsum('ep,p,ij,epjk,ekl->il', jacobien_e_pg, poids_pg, CMandel, B_e_pg, U_e, optimize='optimal') * 1/meshVER.aire

C_hom *= coef

# print(np.linalg.eigvals(C_hom))

# ----------------------------------------------
# Comparaison
# ----------------------------------------------

def Simulation(simu: Simulations.Simu, title=""):

    simu.Bc_Init()

    simu.Need_Update()

    simu.add_dirichlet(simu.mesh.Nodes_Tags(['L3']), [0,0], ['x', 'y'])
    simu.add_surfLoad(simu.mesh.Nodes_Tags(['L1']), [-load/(b*h)], ['y'])

    simu.Solve()

    # Affichage.Plot_BoundaryConditions(simu)
    Affichage.Plot_Result(simu, "uy", title=f"{title} uy")
    # Affichage.Plot_Result(simu, "Eyy")

    print(f"{title}: dy={np.max(simu.Get_Resultat('uy')[simu.mesh.Nodes_Point(Geom.Point(L,0))])}")

Simulation(simuInclusions, "inclusions")
Simulation(simu, "non hom")

testSym = np.linalg.norm(C_hom.T - C_hom)/np.linalg.norm(C_hom)

if testSym >= 1e-12 and testSym <= 1e-7:
    C_hom = 1/2 * (C_hom.T + C_hom)

comp.Set_C(C_hom, False)
Simulation(simu, "hom")

# ax = Affichage.Plot_Result(simu, "uy")[1]
# Affichage.Plot_Result(simuInclusions, "uy", ax=ax)

plt.show()