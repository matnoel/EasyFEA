import numpy as np
import matplotlib.pyplot as plt

import Simulations
import Affichage
import Interface_Gmsh
import Materials
import Geom

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

mat = "bois"

pltVerif = False

L=45
h=90
b=20
d=10
meshSize = h/40
f = 40
sig = f/L/b

# ----------------------------------------------
# Maillage
# ----------------------------------------------

gmshInterface = Interface_Gmsh.Interface_Gmsh()

pt1 = Geom.Point(0,0)
pt2 = Geom.Point(L, h)
domain = Geom.Domain(pt1, pt2, meshSize)
pC = Geom.Point(L/2, h/2)
circle = Geom.Circle(pC, d, meshSize, isCreux=True)

diam = d*0.5
pZone = pC + [diam/2,diam/2]
circleZone = Geom.Circle(pZone, diam)

mesh = gmshInterface.Mesh_Domain_Circle_2D(domain, circle, "TRI3")
xn = mesh.coordo[:,0]
yn = mesh.coordo[:,1]

# Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

nodesEdge = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodesLower = mesh.Nodes_Tags(["L0"])
nodesUpper = mesh.Nodes_Tags(["L2"])

nodesZone = mesh.Nodes_Circle(circleZone)
# Affichage.Plot_Nodes(mesh, nodesZone)

# Récupération des array pour l'intégration numérique
matriceType = "rigi" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)

# ----------------------------------------------
# Comportement
# ----------------------------------------------

if mat == "acier":
    E_exp, v_exp = 210000, 0.3
    comp = Materials.Elas_Isot(2, epaisseur=b, E=E_exp, v=v_exp)
elif mat == "bois":
    EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3

    comp = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), contraintesPlanes=True, epaisseur=b)

# ----------------------------------------------
# Simulation
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesLower, [0,0], ["x","y"])
simu.add_surfLoad(nodesUpper, [-sig], ["y"])

# Affichage.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()

# f_exp = simu._Apply_Neumann("displacement").toarray().reshape(-1)
f_exp = simu.Get_K_C_M_F()[0] @ u_exp

# Affichage.Plot_Result(simu, "Exx")
# Affichage.Plot_Result(simu, "Eyy")
# Affichage.Plot_Result(simu, "Exy")

ddls = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesUpper, ["y"])
forces = simu.Get_K_C_M_F()[0] @ u_exp
forceR = np.sum(f_exp[ddls])

# Affichage.Plot_Result(simu, forces[ddls], cmap="seismic")

# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NouvelleSection("Identification")

# Récupération des déformations aux elements
Eps_exp = simu._Calc_Epsilon_e_pg(u_exp, matriceType)
E11_exp = Eps_exp[:,:,0]
E22_exp = Eps_exp[:,:,1]
E12_exp = Eps_exp[:,:,2]

def Get_A_B_C_D_E(champVirtuel_x, champVirtuel_y, nodes=mesh.nodes,  pltSol=False):
    # Fonction qui renvoie les intégrales calculés

    # Calcul les déplacements associés aux champs virtuels.
    result = np.zeros((mesh.Nn, 2))
    result[nodes,0] = champVirtuel_x(xn[nodes], yn[nodes])
    result[nodes,1] = champVirtuel_y(xn[nodes], yn[nodes])
    u_n = result.reshape(-1)
    simu.set_u_n("displacement", u_n)

    # Calcul les déformations associées aux champs virtuels.
    Eps_e_pg = simu._Calc_Epsilon_e_pg(u_n, matriceType)
    E11_e_pg = Eps_e_pg[:,:,0]
    E22_e_pg = Eps_e_pg[:,:,1]
    E12_e_pg = Eps_e_pg[:,:,2]

    if pltSol:        
        Affichage.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, deformation=False, facteurDef=0.01)
        Affichage.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, deformation=False, facteurDef=0.01)
        # Affichage.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
        # Affichage.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
        # Affichage.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
    
    # Calcul des intégrales.
    A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E11_exp)
    B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_e_pg * E22_exp)
    C = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E22_exp + E22_e_pg * E11_exp)
    D = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_e_pg * E12_exp)

    E = np.sum(f_exp * u_n)

    return A, B, C, D, E

u1, v1 = lambda x, y: x, lambda x, y: 0

u2, v2 = lambda x, y: 0, lambda x, y: y

# u3, v3 = lambda x, y: x*(x-L), lambda x, y: y*(y-h)
# u3, v3 = lambda x, y: x, lambda x, y: y
u3, v3 = lambda x, y: x**2, lambda x, y: y**2

# u4, v4 = lambda x, y: y-pZone.y, lambda x, y: x-pZone.x
# u4, v4 = lambda x, y: y, lambda x, y: x
u4, v4 = lambda x, y: 1, lambda x, y: 1

A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False)
# A1,B1,C1,D1,E1 = 1, -1, 0, 0, 0
A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False)
A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False)

A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, nodesZone, pltSol=False)
# A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False)

systMat = np.array([[A1, B1, C1, D1],
                    [A2, B2, C2, D2],
                    [A3, B3, C3, D3],
                    [A4, B4, C4, D4]])

# c11 c22 c12 c33

cijIdentif = np.linalg.solve(systMat, [E1,E2,E3,E4])

cijMat = np.array([comp.C[0,0], comp.C[1,1], comp.C[0,1], comp.C[2,2]])

erreur = np.abs(cijIdentif - cijMat)/cijMat

print(erreur)

c11 = cijIdentif[0]
c22 = cijIdentif[1]
c12 = cijIdentif[2]
c33 = cijIdentif[3]

matC_Identif = np.array([[c11, c12, 0],
                         [c12, c22, 0],
                         [0, 0, c33]])


if mat == "acier":

    lamb = cijIdentif[2]
    mu = cijIdentif[3]/2

elif mat == "bois":

    matS = np.linalg.inv(matC_Identif)

    Et = 1/matS[0,0]
    El = 1/matS[1,1]
    Gl = 1/matS[2,2]/2
    vl = - matS[0,1] * El

    print(f"El = {El}")
    print(f"Et = {Et}")
    print(f"Gl = {Gl}")
    print(f"vl = {vl}")

plt.show()