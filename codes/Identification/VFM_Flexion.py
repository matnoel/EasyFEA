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

pltVerif = False

L=120
h=13
b=13
meshSize = h/1
p = 800

# ----------------------------------------------
# Maillage
# ----------------------------------------------

gmshInterface = Interface_Gmsh.Interface_Gmsh()

pt1 = Geom.Point(0,-h/2)
pt2 = Geom.Point(L, h/2)
domain = Geom.Domain(pt1, pt2, meshSize)

mesh = gmshInterface.Mesh_2D(domain, "TRI10")
xn = mesh.coordo[:,0]
yn = mesh.coordo[:,1]

nodesEdge = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodesX0 = mesh.Nodes_Tags(["L3"])
nodesXL = mesh.Nodes_Tags(["L1"])

# Récupération des array pour l'intégration numérique
matriceType = "rigi" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)

groupElem1D = mesh.Get_list_groupElem(1)[0]
jacob1D_e_pg = groupElem1D.Get_jacobien_e_pg(matriceType)
poid1D_pg = groupElem1D.Get_poid_pg(matriceType)

assembly1D_e = groupElem1D.Get_assembly_e(2)

# Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

# ----------------------------------------------
# Comportement
# ----------------------------------------------

E_exp, v_exp = 210000, 0.3 
comp = Materials.Elas_Isot(2, epaisseur=b, E=E_exp, v=v_exp)

# ----------------------------------------------
# Simulation
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesX0, [0,0], ["x","y"])
simu.add_surfLoad(nodesXL, [-p/b/h], ["y"])
# simu.add_surfLoad(nodesXL, [-p/b/h], ["x"])

# Affichage.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()
f_exp = simu._Apply_Neumann("displacement").toarray().reshape(-1)

forceR = np.sum(f_exp)


forces = simu.Get_K_C_M_F()[0] @ u_exp
ddls = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", mesh.nodes, ["y"])
# Affichage.Plot_Result(simu, fff[ddls], cmap="seismic")

f_exp_loc = f_exp[assembly1D_e]

# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NewSection("Identification")

# Récupération des déformations aux elements
Eps_exp = simu._Calc_Epsilon_e_pg(u_exp, matriceType)
E11_exp = Eps_exp[:,:,0]
E22_exp = Eps_exp[:,:,1]
E12_exp = Eps_exp[:,:,2]

# Affichage.Plot_Result(simu, "Exx", nodeValues=False)
# Affichage.Plot_Result(simu, "Eyy", nodeValues=False)
# Affichage.Plot_Result(simu, "Exy", nodeValues=False)

def Get_A_B_C_D_E(champVirtuel_x, champVirtuel_y, pltSol=False):
    # Fonction qui renvoie les intégrales calculés

    # Calcul les déplacements associés aux champs virtuels.
    result = np.zeros((mesh.Nn, 2))
    result[:,0] = champVirtuel_x(xn, yn)
    result[:,1] = champVirtuel_y(xn, yn)
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
        Affichage.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
        Affichage.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
        Affichage.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
    
    # Calcul des intégrales.
    A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E11_exp)
    B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_e_pg * E22_exp)
    C = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E22_exp + E22_e_pg * E11_exp)
    D = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_e_pg * E12_exp)

    E = np.sum(f_exp * u_n)
    # u_n_loc = u_n[assembly1D_e]
    # E = np.einsum('ep,p,ei->', jacob1D_e_pg, poid1D_pg, f_exp_loc * u_n_loc)
    # E = np.einsum('ep,p,ei->', jacob1D_e_pg, poid1D_pg, u_n_loc) * b

    return A, B, C, D, E

A1, B1, C1, D1, E1 = Get_A_B_C_D_E(lambda x, y: 0, lambda x, y: x, pltSol=False)
c33 = E1/D1/2
# muIdentif = - p * L /D1/2 # - E1/D1/2

mu = comp.get_mu()
lamb = comp.get_lambda()

A2,B2,C2,D2,E2 = Get_A_B_C_D_E(lambda x, y: x*y, lambda x, y: 0, pltSol=False)
c11 = - c33 * (D2-C2)/(A2+B2+C2)

systMat = np.array([[A1, B1, C1, D1],[A2, B2, C2, D2],[1,-1,0,0],[1,0,-1,-1]])

# c11 c22 c12 c33

# cijIdentif = np.linalg.solve(systMat, [-800*120,0,0,0])
cijIdentif = np.linalg.solve(systMat, [E1,E2,0,0])

cijMat = np.array([comp.C[0,0], comp.C[1,1], comp.C[0,1], comp.C[2,2]])

erreur = np.abs(cijIdentif - cijMat)/np.linalg.norm(cijMat)

print(erreur)

plt.show()