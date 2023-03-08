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
meshSize = h/20
p = 800

# ----------------------------------------------
# Maillage
# ----------------------------------------------

gmshInterface = Interface_Gmsh.Interface_Gmsh()

pt1 = Geom.Point()
pt2 = Geom.Point(L, h)
domain = Geom.Domain(pt1, pt2, meshSize)

mesh = gmshInterface.Mesh_Domain_2D(domain, "QUAD4", isOrganised=True)
xn = mesh.coordo[:,0]
yn = mesh.coordo[:,1]

nodesEdge = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodesX0 = mesh.Nodes_Tags(["L3"])
nodesXL = mesh.Nodes_Tags(["L1"])

# Récupération des array pour l'intégration numérique
matriceType = "masse" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)
aire_e = jacob2D_e_pg @ poid2D_pg

groupElem1D = mesh.Get_list_groupElem(1)[0]
jacob1D_e_pg = groupElem1D.Get_jacobien_e_pg(matriceType)
poid1D_pg = groupElem1D.Get_poid_pg(matriceType)
longueur_e = jacob1D_e_pg @ poid1D_pg

assembly1D_e = groupElem1D.Get_assembly_e(2)

# Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

# ----------------------------------------------
# Comportement
# ----------------------------------------------

E_exp, v_exp = 210000, 0.3 
comp = Materials.Elas_Isot(2, epaisseur=b)

# ----------------------------------------------
# Simulation
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesX0, [0,0], ["x","y"])
simu.add_surfLoad(nodesXL, [-p/b/h], ["y"])
# simu.add_surfLoad(nodesXL, [-p/b/h], ["x"])

# Affichage.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()
# f_exp = simu._Apply_Neumann("displacement").toarray().reshape(-1)
f_exp = simu.Get_K_C_M_F()[0] @ u_exp

f_exp_loc = f_exp[assembly1D_e]

# Affichage.Plot_Result(simu, "uy")
# simu.Resultats_Resume()

# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NouvelleSection("Identification")

# Récupération des déformations aux elements
Eps_exp_e = simu.Get_Resultat("Strain", nodeValues=False)
E11_exp_e = Eps_exp_e[:,0]
E22_exp_e = Eps_exp_e[:,1]
E12_exp_e = Eps_exp_e[:,2]

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

    # Calcul les déformations associées aux champs virtuels.
    simu.set_u_n("displacement", u_n)
    Eps_e = simu.Get_Resultat("Strain", nodeValues=False)
    E11_e = Eps_e[:,0]
    E22_e = Eps_e[:,1]
    E12_e = Eps_e[:,2]

    if pltSol:        
        Affichage.Plot_Result(simu, "ux", title=r"$u_x^*$")
        Affichage.Plot_Result(simu, "uy", title=r"$u_y^*$")
        Affichage.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False)
        Affichage.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False)
        Affichage.Plot_Result(simu, "Evm", title=r"$\epsilon_{xy}^*$", nodeValues=False)
    
    # Calcul des intégrales.
    A = b * np.einsum('ep,p,e->', jacob2D_e_pg, poid2D_pg, E11_e * E11_exp_e)
    B = b * np.einsum('ep,p,e->', jacob2D_e_pg, poid2D_pg, E22_e * E22_exp_e)
    C = b * np.einsum('ep,p,e->', jacob2D_e_pg, poid2D_pg, E11_e * E22_exp_e + E22_e * E11_exp_e)
    D = np.einsum('ep,p,e->', jacob2D_e_pg, poid2D_pg, E12_e * E12_exp_e)

    u_n_loc = u_n[assembly1D_e]
    F = np.einsum('ep,p,ei->', jacob1D_e_pg, poid1D_pg, f_exp_loc * u_n_loc) * b

    return A, B, C, D, F

# (lambda x, y: x, lambda x, y: 0) # Eps -> [1, 0, 0]
# (lambda x, y: 0, lambda x, y: y) # Eps -> [0, 1, 0]
# (lambda x, y: x*(x-L), lambda x, y: y*(y-h)) # Eps -> [1, 1, 0]
# (lambda x, y: y, lambda x, y: x) # Eps -> [0, 0, sqrt(2)]

# A, B, C, D, F = Get_A_B_C_D_E(lambda x, y: y*(y-h), lambda x, y: x*(x-L), pltSol=True)
A, B, C, D, F = Get_A_B_C_D_E(lambda x, y: y, lambda x, y: x, pltSol=True)
mu = F/D/2

mmm = comp.get_mu()

A1, B1, C1, D1, F1 = Get_A_B_C_D_E(lambda x, y: x, lambda x, y: 0)
A2, B2, C2, D2, F2 = Get_A_B_C_D_E(lambda x, y: 0, lambda x, y: x)
A3, B3, C3, D3, F3 = Get_A_B_C_D_E(lambda x, y: y, lambda x, y: x)

matrice = np.array([[A1+B1, C1, D1],[A2+B2, C2, D2],[A3+B3, C3, D]])
result = np.linalg.inv(matrice) @ np.array([F1, F2, F3])
cm = comp.C

matriceCoefs = np.array([cm[0,0], cm[0,1], cm[2,2]])

errreur = np.abs(result - matriceCoefs)/matriceCoefs

print(np.linalg.norm(result - matriceCoefs)/np.linalg.norm(matriceCoefs))


plt.show()