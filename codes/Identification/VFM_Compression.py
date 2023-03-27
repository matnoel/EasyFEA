import numpy as np
<<<<<<< Updated upstream
=======
from scipy import sparse
from scipy.sparse.linalg import spsolve

from scipy.linalg import lstsq
>>>>>>> Stashed changes
import matplotlib.pyplot as plt
import pandas as pd

import Simulations
import Affichage
import Interface_Gmsh
import Materials
import Geom
import Folder

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

folder = Folder.New_File("Identification", results=True)

mat = "bois" # "bois", "acier"

<<<<<<< Updated upstream
perturbations = np.linspace(0, 0.07, 15)
nTirage = 100

=======
perturbations = np.linspace(0, 0.02, 5)
nTirage = 100

useSpecVirtual = True

>>>>>>> Stashed changes
pltVerif = False

L=45
h=90
b=20
d=10
<<<<<<< Updated upstream
meshSize = h/40
=======
# meshSize = h/40
meshSize = h/50
>>>>>>> Stashed changes
f = 40
sig = f/(L*b)

# ----------------------------------------------
# Maillage
# ----------------------------------------------

<<<<<<< Updated upstream
gmshInterface = Interface_Gmsh.Interface_Gmsh()
=======
gmshInterface = Interface_Gmsh.Interface_Gmsh(False)
>>>>>>> Stashed changes

pt1 = Geom.Point(0,0)
pt2 = Geom.Point(L, h)
domain = Geom.Domain(pt1, pt2, meshSize)
pC = Geom.Point(L/2, h/2)
circle = Geom.Circle(pC, d, meshSize, isCreux=True)

diam = d
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
<<<<<<< Updated upstream

ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesUpper, ["y"])

nodesZone = mesh.Nodes_Circle(circleZone)
# Affichage.Plot_Nodes(mesh, nodesZone)
# Affichage.Save_fig(folder, "vfm zone cisaillement")

# Récupération des array pour l'intégration numérique
matriceType = "rigi" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)

=======
nodesUpperLower = mesh.Nodes_Tags(["L0", "L2"])
nodesZone = mesh.Nodes_Circle(circleZone)

ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesUpper, ["y"])

# Affichage.Plot_Nodes(mesh, nodesZone)
# Affichage.Save_fig(folder, "vfm zone cisaillement")
# Affichage.Plot_Elements(mesh, nodesUpper, 1)

# Récupération des array pour l'intégration numérique
matriceType = "masse" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)

groupElem1D = mesh.Get_list_groupElem(1)[0]
elems1D = groupElem1D.Get_Elements_Nodes(nodesUpper)

assembly1D = groupElem1D.Get_assembly_e(2)[elems1D]
jacob1D_e_pg = groupElem1D.Get_jacobien_e_pg(matriceType)[elems1D]
poid1D_pg = groupElem1D.Get_poid_pg(matriceType)

>>>>>>> Stashed changes
# ----------------------------------------------
# Comportement
# ----------------------------------------------

if mat == "acier":
    E_exp, v_exp = 210000, 0.3
    comp = Materials.Elas_Isot(2, epaisseur=b, E=E_exp, v=v_exp)

    dict_param = {
        "lambda" : comp.get_lambda(),
        "mu" : comp.get_mu()
    }

elif mat == "bois":
    EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3

    dict_param = {
        "EL" : EL_exp,
        "GL" : GL_exp,
        "ET" : ET_exp,
        "vL" : vL_exp
    }

    comp = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), contraintesPlanes=True, epaisseur=b)

# ----------------------------------------------
# Simulation
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesLower, [0], ["y"])
simu.add_dirichlet(mesh.Nodes_Tags((["P0"])), [0], ["x"])
simu.add_surfLoad(nodesUpper, [-sig], ["y"])

# Affichage.Plot_BoundaryConditions(simu)
# Affichage.Save_fig(folder, "Boundary Compression")

u_exp = simu.Solve()

<<<<<<< Updated upstream
f_exp = simu._Apply_Neumann("displacement").toarray().reshape(-1)
forceR2 = np.sum(f_exp[ddlsY_Upper])
# f_exp = simu.Get_K_C_M_F()[0] @ u_exp

# Affichage.Plot_Result(simu, "Exx")
# Affichage.Plot_Result(simu, "Eyy")
# Affichage.Plot_Result(simu, "Exy")

forces = simu.Get_K_C_M_F()[0] @ u_exp
forceR = np.sum(f_exp[ddlsY_Upper])

# Affichage.Plot_Result(simu, forces[ddls], cmap="seismic")

=======
>>>>>>> Stashed changes
# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NouvelleSection("Identification")

<<<<<<< Updated upstream
def Get_A_B_C_D_E(champVirtuel_x, champVirtuel_y, nodes=mesh.nodes,  pltSol=False):
    # Fonction qui renvoie les intégrales calculés
=======
def Get_A_B_C_D_E(champVirtuel_x, champVirtuel_y, nodes=mesh.nodes, pltSol=False, f=None):
    """Calcul des intégrales"""
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        Affichage.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, deformation=False, facteurDef=0.01)
        Affichage.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, deformation=False, facteurDef=0.01)
=======
        Affichage.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, colorbarIsClose=True)
        Affichage.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, colorbarIsClose=True)
>>>>>>> Stashed changes
        # Affichage.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
        # Affichage.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
        # Affichage.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
    
    # Calcul des intégrales.
    A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E11_exp)
    B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_e_pg * E22_exp)
    C = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E22_exp + E22_e_pg * E11_exp)
    D = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_e_pg * E12_exp)

<<<<<<< Updated upstream
    E = np.sum(f_exp_bruit * u_n)
=======
    if isinstance(f, float|int):
        uloc = u_n[assembly1D]
        E = np.einsum("ep,p,ei->", jacob1D_e_pg, poid1D_pg, uloc) * f * b
        pass
        E = np.sum(u_n[ddlsY_Upper]*f)
    else:
        E = np.sum(f_exp_bruit * u_n) 
>>>>>>> Stashed changes

    return A, B, C, D, E

list_dict_perturbation = []

for perturbation in perturbations:

    print(f"\nperturbation = {perturbation}")

    list_dict_tirage = []

    for tirage in range(nTirage):

        print(f"tirage = {tirage}", end='\r')

        # bruitage de la solution
        bruit = np.abs(u_exp).max() * (np.random.rand(u_exp.shape[0]) - 1/2) * perturbation
        u_exp_bruit = u_exp + bruit

        # Récupération des déformations aux elements
<<<<<<< Updated upstream
        Eps_exp = simu._Calc_Epsilon_e_pg(u_exp, matriceType)
=======
        Eps_exp = simu._Calc_Epsilon_e_pg(u_exp_bruit, matriceType)
>>>>>>> Stashed changes
        E11_exp = Eps_exp[:,:,0]
        E22_exp = Eps_exp[:,:,1]
        E12_exp = Eps_exp[:,:,2]

<<<<<<< Updated upstream
        f_exp_bruit = simu.Get_K_C_M_F()[0] @ u_exp_bruit
        # f_exp_bruit = f_exp.copy()

        fbruit = np.sum(f_exp_bruit[ddlsY_Upper])
        fbruit = -f

        u1, v1 = lambda x, y: x, lambda x, y: 0
        A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False)
        # A1,B1,C1,D1,E1 = 1, -1, 0, 0, 0
        E1 = 0

        u2, v2 = lambda x, y: 0, lambda x, y: y
        A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False)
        E2 = fbruit * h

        # u3, v3 = lambda x, y: x*(x-L), lambda x, y: y*(y-h)
        # u3, v3 = lambda x, y: x, lambda x, y: y
        u3, v3 = lambda x, y: x**2, lambda x, y: y**2
        A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False)
        E3 = fbruit * h**2

        # u4, v4 = lambda x, y: y-pZone.y, lambda x, y: x-pZone.x
        u4, v4 = lambda x, y: y, lambda x, y: x
        # u4, v4 = lambda x, y: 1, lambda x, y: 1
        A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, nodesZone, pltSol=False)
        # A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False)
        E4 = 0
=======
        # Attention, normalement on  a pas accès a cette information dans un essai réel !        
        f_exp_bruit = simu.Get_K_C_M_F()[0] @ u_exp_bruit

        # f_exp_bruit = f_exp.copy()       

        if useSpecVirtual:

            # ----------------------------------------------
            # Champs spéciaux
            # ----------------------------------------------

            # print("\n")

            dimConditions = nodesUpperLower.size * 2  + 4

            dimS2 = dimConditions/2
            
            # m=6
            # n=3            

            # p=6
            # q=3

            m=3
            n=3            

            p=3
            q=3

            

            sizeA = (m+1) * (n+1)
            sizeB = (p+1) * (q+1)
            sizeAB = sizeA + sizeB

            ecart = sizeAB - dimConditions

            mattIsSquare = ecart == 0
            # mattIsSquare = False

            pos_Aij = np.arange(sizeA)
            pos_Bij = np.arange(sizeA, sizeA+sizeB)

            lignes = []        
            colonnes = []
            values = []
            vectCond = []            

            def Add_Conditions(noeuds: np.ndarray, condU: float, condV: float, l=-1):

                # conditions u et v pour les noeuds renseignés
                
                for noeud in noeuds:
                    
                    # conditions u
                    l += 1                
                    c = -1
                    for i in range(m+1):
                        for j in range(n+1):
                            c += 1
                            lignes.append(l)
                            colonnes.append(pos_Aij[c])
                            values.append(xn[noeud]**i * yn[noeud]**j)                    
                            # values.append((xn[noeud]/L)**i * (yn[noeud]/h)**j)
                    vectCond.append(condU)

                    # conditions v
                    l += 1                
                    c = -1
                    for i in range(p+1):
                        for j in range(q+1):
                            c += 1
                            lignes.append(l)
                            colonnes.append(pos_Bij[c])
                            values.append(xn[noeud]**i * yn[noeud]**j)                    
                            # values.append((xn[noeud]/L)**i * (yn[noeud]/h)**j)
                    vectCond.append(condV)

                return l

            
            lastLigne = Add_Conditions(nodesLower, 0, 0)

            const = h
            lastLigne = Add_Conditions(nodesUpper, 0, const, lastLigne)

            condNodes = np.unique(lignes).size
            
            # conditions sur les déformations
            coord_e_p = mesh.groupElem.Get_coordo_e_p(matriceType)
            xn_e_g = coord_e_p[:,:,0]
            yn_e_g = coord_e_p[:,:,1]
            
            c = -1
            for i in range(m+1):
                for j in range(n+1):

                    c += 1                    

                    dx_ij_e_p = i*xn_e_g**(i-1) * yn_e_g**j
                    # dx_ij_e_p = i/L*(xn_e_g/L)**(i-1) * (yn_e_g/h)**j

                    dy_ij_e_p = xn_e_g**i * j*yn_e_g**(j-1)
                    # dy_ij_e_p = (xn_e_g/L)**i * j/h*(yn_e_g/h)**(j-1)

                    if i > 0:
                        lignes.append(lastLigne + 1)
                        colonnes.append(pos_Aij[c])
                        cond1_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_exp * dx_ij_e_p)
                        values.append(cond1_A)

                        lignes.append(lastLigne + 3)
                        colonnes.append(pos_Aij[c])
                        cond3_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_exp * dx_ij_e_p)
                        values.append(cond3_A)

                    if j > 0:
                        lignes.append(lastLigne + 4)
                        colonnes.append(pos_Aij[c])
                        cond4_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_exp * dy_ij_e_p)
                        values.append(cond4_A)

            c = -1
            for i in range(p+1):
                for j in range(q+1):

                    c += 1

                    dx_ij_e_p = i*xn_e_g**(i-1) * yn_e_g**j
                    # dx_ij_e_p = i/L*(xn_e_g/L)**(i-1) * (yn_e_g/h)**j

                    dy_ij_e_p = xn_e_g**i * j*yn_e_g**(j-1)
                    # dy_ij_e_p = (xn_e_g/L)**i * j/h*(yn_e_g/h)**(j-1)

                    if j > 0:
                        lignes.append(lastLigne + 2)
                        colonnes.append(pos_Bij[c])
                        cond2_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_exp * dy_ij_e_p)
                        values.append(cond2_B)

                        lignes.append(lastLigne + 3)
                        colonnes.append(pos_Bij[c])
                        cond3_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_exp * dy_ij_e_p)
                        values.append(cond3_B)  

                    if i > 0:
                        lignes.append(lastLigne + 4)
                        colonnes.append(pos_Bij[c])
                        cond4_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_exp * dx_ij_e_p)
                        values.append(cond4_B)

            
            matt = sparse.csr_matrix((values, (lignes, colonnes)), (condNodes + 4, sizeA+sizeB))
            # print(matt.shape)

            # plt.spy(matt.toarray())
            # plt.imshow(matt.toarray())
            # plt.colorbar()
            # plt.grid()

            def Get_u_v(condEps: np.ndarray):

                # print(matt.toarray())        
                conds = np.concatenate((vectCond, condEps))

                if mattIsSquare:
                    sol = spsolve(matt, conds)
                else:
                    sol = lstsq(matt.toarray(), conds)[0]
                
                ttt = matt @ sol
                tol = 1e-7

                testErr = np.linalg.norm(ttt - conds)/np.linalg.norm(conds)
                assert testErr <= tol, f"erreur = {testErr}"

                Aijs = sol[pos_Aij]
                Bijs = sol[pos_Bij]

                ue = "lambda x,y: "
                ve = "lambda x,y: "
                cc = -1
                for i in range(p+1):
                    for j in range(q+1):
                        cc += 1
                        if ue != "lambda x,y: ":
                            ue += " + "
                        # ue += f"A{i}{j} x**{i} * y**{j}"    
                        ue += f"{Aijs[cc]} * x**{i} * y**{j}"    
                        
                        if ve != "lambda x,y: ":
                            ve += " + "
                        # ve += f"B{i}{j} x**{i} * y**{j}"    
                        ve += f"{Bijs[cc]} * x**{i} * y**{j}"    

                funcU, funcV = tuple(map(eval, [ue, ve]))

                return funcU, funcV

            u1, v1 = Get_u_v([1,0,0,0])
            u2, v2 = Get_u_v([0,1,0,0])
            u3, v3 = Get_u_v([0,0,1,0])
            u4, v4 = Get_u_v([0,0,0,1])

            A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False, f=-f)
            A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False, f=-f)
            A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False, f=-f)
            A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False, f=-f)

            tt = (E1, E2, E3, E4)

            # E1 = E2 = E3 = E4 = - f * const

            dd = simu.displacement.reshape(-1,2)

            tol = 1e-9
            
            # verifie que le champ vaut 90 en haut suivant y
            testY_Haut_1 = np.linalg.norm(np.mean(dd[nodesUpper, 1]) - const)/const
            # verifie que le champ est constant
            testY_Haut_2 = np.std(dd[nodesUpper, 1])/const
            assert testY_Haut_1 <= tol and testY_Haut_2 <= tol  

            # verifie que le champ vaut 0 en bas suivant y
            testYBas_1 = np.linalg.norm(np.mean(dd[nodesLower, 1]))/const
            # verifie que le champ est constant
            testY_Bas_2 = np.std(dd[nodesLower, 1])/const
            assert testYBas_1 <= tol and testY_Bas_2 <= tol  

            # verifie que le champ vaut 0 en haut suivant x
            testX_1 = np.linalg.norm(np.mean(dd[nodesUpperLower, 0]))/const
            testX_2 = np.std(dd[nodesUpperLower, 0])/const
            assert testX_1 <= tol and testX_2 <= tol  

            pass

        else:            
            
            # ----------------------------------------------
            # Champ 1
            # ----------------------------------------------
            
            u1, v1 = lambda x, y: (y-h) * y * -(x-L/2), lambda x, y: 0 # tonneau
            # u1, v1 = lambda x, y: (y-h) * y, lambda x, y: 0 # tonneau

            # u1, v1 = lambda x, y: x, lambda x, y: 0
            # u1, v1 = lambda x, y: x**2, lambda x, y: 0
            
            A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False)        
            E1 = 0

            # ----------------------------------------------
            # Champ 2
            # ----------------------------------------------

            u2, v2 = lambda x, y: 0, lambda x, y: y
            # u2, v2 = lambda x, y: 0, lambda x, y: y**2
            
            A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False)
            E2 = - f * h        

            # ----------------------------------------------
            # Champ 3
            # ----------------------------------------------
            
            # u3, v3 = lambda x, y: (x**3/3)-(L*x**2/2), lambda x, y: (y**3/3)-(h*y**2/2)
            # u3, v3 = lambda x, y: (x**3/3)-(L*x**2/2), lambda x, y: 0
            # u3, v3 = lambda x, y: x*(L-x), lambda x, y: y*(y-h)
            # u3, v3 = lambda x, y: x*(L-x), lambda x, y: 0
            # u3, v3 = lambda x, y: y*(y-h), lambda x, y: x*(L-x)
            # u3, v3 = lambda x, y: y**2, lambda x, y: x**2

            u3, v3 = lambda x, y: y*(y-h), lambda x, y:  (x-L/2)**2 * y*(y-h)/h**2
            # u3, v3 = lambda x, y: (y-h) * y * -(x-L/2), lambda x, y:  (x-L/2)**2 * y*(y-h)/h**2


            A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False)
            E3=0

            # ----------------------------------------------
            # Champ 4
            # ----------------------------------------------

            # u4, v4 = lambda x, y: y-pZone.y, lambda x, y: x-pZone.x
            # u4, v4 = lambda x, y: y, lambda x, y: x
            u4, v4 = lambda x, y: 0, lambda x, y:  (x-L/2)**2 * y*(y-h)/h**2 + y
            # u4, v4 = lambda x, y: (x-L/2)**2 * y*(y-h)/h**2 + y, lambda x, y:  0
            
            # A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, nodesZone, pltSol=False)
            A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False)
            E4 = - f * h

        # ----------------------------------------------
        # Résolution
        # ----------------------------------------------
>>>>>>> Stashed changes

        systMat = np.array([[A1, B1, C1, D1],
                            [A2, B2, C2, D2],
                            [A3, B3, C3, D3],
                            [A4, B4, C4, D4]])

        # c11 c22 c12 c33

        cijIdentif = np.linalg.solve(systMat, [E1,E2,E3,E4])

        cijMat = np.array([comp.C[0,0], comp.C[1,1], comp.C[0,1], comp.C[2,2]])

        erreur = np.abs(cijIdentif - cijMat)/cijMat

        # print(erreur)

        c11 = cijIdentif[0]
        c22 = cijIdentif[1]
        c12 = cijIdentif[2]
        c33 = cijIdentif[3]

        matC_Identif = np.array([[c11, c12, 0],
                                [c12, c22, 0],
                                [0, 0, c33]])

        dict_tirage = {
            "tirage" : tirage
        }

        if mat == "acier":

            lamb = cijIdentif[2]
            mu = cijIdentif[3]/2

            dict_tirage["lambda"] = lamb
            dict_tirage["mu"] = mu

        elif mat == "bois":

            matS = np.linalg.inv(matC_Identif)

            Et = 1/matS[0,0]
            El = 1/matS[1,1]
            Gl = 1/matS[2,2]/2
            vl = - matS[0,1] * El

            dict_tirage["EL"] = El
            dict_tirage["GL"] = Gl
            dict_tirage["ET"] = Et
            dict_tirage["vL"] = vl

        list_dict_tirage.append(dict_tirage)

    df_tirage = pd.DataFrame(list_dict_tirage)

    dict_perturbation = {
        "perturbation" : perturbation,
    }

    if mat == "acier":
        dict_perturbation["lambda"] = df_tirage["lambda"].values
        dict_perturbation["mu"] = df_tirage["mu"].values
    elif mat == "bois":
        dict_perturbation["EL"] = df_tirage["EL"].values
        dict_perturbation["GL"] = df_tirage["GL"].values
        dict_perturbation["ET"] = df_tirage["ET"].values
        dict_perturbation["vL"] = df_tirage["vL"].values

    list_dict_perturbation.append(dict_perturbation)
    

df_pertubation = pd.DataFrame(list_dict_perturbation)

# ----------------------------------------------
# Affichage
# ----------------------------------------------

if mat == "acier":
    params = ["lambda","mu"]
elif mat == "bois":
    params = ["EL", "GL", "ET", "vL"]

borne = 0.95
bInf = 0.5 - (0.95/2)
bSup = 0.5 + (0.95/2)

print("\n")

for param in params:

    axParam = plt.subplots()[1]
    
    paramExp = dict_param[param]

    perturbations = df_pertubation["perturbation"]

    nPertu = perturbations.size
    values = np.zeros((nPertu, nTirage))
    for p in range(nPertu):
        values[p] = df_pertubation[param].values[p]
    values *= 1/paramExp

    mean = values.mean(axis=1)
    std = values.std(axis=1)

    paramInf, paramSup = tuple(np.quantile(values, (bInf, bSup), axis=1))

    axParam.plot(perturbations, [1]*nPertu, label=f"{param}_exp", c="black", ls='--')
    axParam.plot(perturbations, mean, label=f"{param}_moy")
    axParam.fill_between(perturbations, paramInf, paramSup, alpha=0.3, label=f"{borne*100} % ({nTirage} tirages)")
    axParam.set_xlabel("perturbations")
<<<<<<< Updated upstream
    axParam.set_ylabel(fr"$1 \ / \ {param}_{'{exp}'}$")
=======
    axParam.set_ylabel(fr"${param} \ / \ {param}_{'{exp}'}$")
>>>>>>> Stashed changes
    axParam.grid()
    axParam.legend(loc="upper left")
    
    Affichage.Save_fig(folder, "VFM_"+param, extension='pdf')

    print(f"{param} = {mean.mean()*paramExp}")




plt.show()