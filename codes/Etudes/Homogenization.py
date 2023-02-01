from Interface_Gmsh import Interface_Gmsh
import Geom
import Affichage
import Simulations
import Materials

np = Materials.np
plt = Affichage.plt

Affichage.Clear()

# ----------------------------------------------
# Options
# ----------------------------------------------

L = 120 # mm
h = 13
b = 13

nL = 20 # bombre inclusion suivant L
nH = 3

# c = 13/2
cL = L/(2*nL)
cH = h/(2*nH)

E = 210000
v = 0.3

load = 800

# ----------------------------------------------
# Mesh
# ----------------------------------------------
elemType = "TRI6"
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

        domain = Geom.Domain(ptd1, ptd2, meshSize, isCreux=True)

        listGeomInDomain.append(domain)

interfaceGmsh = Interface_Gmsh(False)


# maillage avec les inclusions
meshInclusions = interfaceGmsh.Mesh_From_Points_2D([pt1, pt2, pt3, pt4], elemType, inclusions=listGeomInDomain, tailleElement=meshSize)

# maillage sans les inclusions
mesh = interfaceGmsh.Mesh_From_Points_2D([pt1, pt2, pt3, pt4], elemType, tailleElement=meshSize)


ptI1 = Geom.Point(-cL,-cH)
ptI2 = Geom.Point(cL,-cH)
ptI3 = Geom.Point(cL, cH)
ptI4 = Geom.Point(-cL, cH)

meshVER = interfaceGmsh.Mesh_From_Points_2D([ptI1, ptI2, ptI3, ptI4], elemType, inclusions=[Geom.Domain(Geom.Point(-cL/2,-cH/2), Geom.Point(cL/2, cH/2), meshSize, isCreux=True)], tailleElement=meshSize)

Affichage.Plot_Mesh(meshInclusions)
Affichage.Plot_Mesh(mesh)
Affichage.Plot_Mesh(meshVER)

# plt.show()

# ----------------------------------------------
# Materiau
# ----------------------------------------------

# comportement élastique de la poutre
compInclusions = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=b)
CMandel = compInclusions.C
materiauInclusions = Materials.Create_Materiau(compInclusions)

comp = Materials.Elas_Anisot(2, CMandel, np.array([1,0,0]), useVoigtNotation=False)
testC = np.linalg.norm(compInclusions.C-comp.C)/np.linalg.norm(compInclusions.C)
assert testC < 1e-12, "les matrices sont différentes"
materiau = Materials.Create_Materiau(comp)

# ----------------------------------------------
# Homogenization
# ----------------------------------------------

simuInclusions = Simulations.Create_Simu(meshInclusions, materiauInclusions)
simuVER = Simulations.Create_Simu(meshVER, materiauInclusions)
simu = Simulations.Create_Simu(mesh, materiau)

noeudsDuBord = meshVER.Nodes_Tag(["L1", "L2", "L3", "L4"])
ddlsNoeudsDubord = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeudsDuBord, ["x","y"])
ddlsX = ddlsNoeudsDubord.reshape(-1, 2)[:,0]
ddlsY = ddlsNoeudsDubord.reshape(-1, 2)[:,1]
Affichage.Plot_Nodes(meshVER, noeudsDuBord)

r2 = np.sqrt(2)
E11 = np.array([[1, 0],[0, 0]])
E22 = np.array([[0, 0],[0, 1]])
E12 = np.array([[0, 1/r2],[1/r2, 0]])

# u11 = np.einsum('ij,nj->ni', E11, meshInclusion.coordo[noeudsDuBord, 0:2])
# u22 = np.einsum('ij,nj->ni', E22, meshInclusion.coordo[noeudsDuBord, 0:2])
# u12 = np.einsum('ij,nj->ni', E12, meshInclusion.coordo[noeudsDuBord, 0:2])

def CalcDisplacement(Ekl: np.ndarray):

    simuVER.Bc_Init()

    simuVER.add_dirichlet(noeudsDuBord, [lambda x, y, z: Ekl.dot([x, y])[0], lambda x, y, z: Ekl.dot([x, y])[1]], ["x","y"])

    ukl = simuVER.Solve()

    simuVER.Save_Iteration()

    # Affichage.Plot_Result(simuInclusion, "Exx")
    # Affichage.Plot_Result(simuInclusion, "Eyy")
    # Affichage.Plot_Result(simuInclusion, "Exy")

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

# print(np.linalg.eigvals(C_hom))

# ----------------------------------------------
# Comparaison
# ----------------------------------------------

def Simulation(simu: Simulations.Simu, title=""):

    simu.Bc_Init()

    simu.Matrices_Need_Update()

    simu.add_dirichlet(simu.mesh.Nodes_Tag(['L4']), [0,0], ['x', 'y'])
    simu.add_surfLoad(simu.mesh.Nodes_Tag(['L2']), [-load/(b*h)], ['y'])

    simu.Solve()

    # Affichage.Plot_BoundaryConditions(simu)
    Affichage.Plot_Result(simu, "uy", title=f"{title} dy")
    # Affichage.Plot_Result(simu, "Eyy")

    print(f"{title}: dy={np.max(simu.Get_Resultat('dy')[simu.mesh.Nodes_Point(Geom.Point(L,0))])}")

Simulation(simuInclusions, "inclusions")
Simulation(simu, "non hom")
comp.Update(C_hom, False)
Simulation(simu, "hom")

# ax = Affichage.Plot_Result(simu, "uy")[1]
# Affichage.Plot_Result(simuInclusions, "uy", ax=ax)

plt.show()