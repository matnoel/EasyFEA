import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import pandas as pd

import Simulations
import Affichage
from Interface_Gmsh import Interface_Gmsh
import Materials
import Geom
import Folder

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

folder = Folder.New_File("Identification", results=True)

mat = "bois" # "bois", "acier"

perturbations = np.linspace(0, 0.02, 5)
# perturbations = [0, 0.005]
nTirage = 100

useSpecVirtual = False

pltVerif = False

L=45
H=90

l = L/2
h = H/2

b=20
d=10

meshSize = H/70
f = 40
sig = f/(L*b)

# ----------------------------------------------
# Maillage
# ----------------------------------------------
pt1 = Geom.Point(-L/2, -H/2)
pt2 = Geom.Point(L/2, H/2)
domain = Geom.Domain(pt1, pt2, meshSize)
pC = Geom.Point(0, 0)
circle = Geom.Circle(pC, d, meshSize, isCreux=True)

diam = d
pZone = pC + [1*diam/2,1*diam/2]
# pZone = pC + [1.5*diam/2,0]
# pZone = pC + [0,0]
circleZone = Geom.Circle(pZone, diam, meshSize)

coordInter = Geom.Points_IntersectCircles(circle, circleZone)

pt1 = Geom.Point(*coordInter[0, :])
pt2 = Geom.Point(*coordInter[1, :])

circleArc1 = Geom.CircleArc(pt1, pZone, pt2, meshSize, coef=-1)
circleArc2 = Geom.CircleArc(pt2, pC, pt1, meshSize)
contour = Geom.Contour([circleArc1, circleArc2], isCreux=False)

mesh = Interface_Gmsh().Mesh_2D(domain, [circle, contour], "TRI6")
xn = mesh.coordo[:,0]
yn = mesh.coordo[:,1]

Affichage.Plot_Model(mesh)
Affichage.Plot_Mesh(mesh)

nodesEdge = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodesLower = mesh.Nodes_Tags(["L0"])
nodesUpper = mesh.Nodes_Tags(["L2"])

nodesUpperLower = mesh.Nodes_Tags(["L0", "L2"])
nodesZone = mesh.Nodes_Circle(circleZone)

ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesUpper, ["y"])

# Affichage.Plot_Nodes(mesh, nodesZone)
# Affichage.Save_fig(folder, "vfm zone cisaillement")
# Affichage.Plot_Elements(mesh, nodesUpper, 1)

# Récupération des array pour l'intégration numérique
matriceType = "rigi" 
jacob2D_e_pg = mesh.Get_jacobien_e_pg(matriceType)
poid2D_pg = mesh.Get_poid_pg(matriceType)

groupElem1D = mesh.Get_list_groupElem(1)[0]
elems1D = groupElem1D.Get_Elements_Nodes(nodesUpper)

assembly1D = groupElem1D.Get_assembly_e(2)[elems1D]
jacob1D_e_pg = groupElem1D.Get_jacobien_e_pg(matriceType)[elems1D]
poid1D_pg = groupElem1D.Get_poid_pg(matriceType)

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

# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NouvelleSection("Identification")

def Get_A_B_C_D_E(champVirtuel_x, champVirtuel_y, nodes=mesh.nodes, pltSol=False, f=None, pltEps=True):
    """Calcul des intégrales"""

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
        Affichage.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, colorbarIsClose=True)
        Affichage.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, colorbarIsClose=True)
        if pltEps:
            Affichage.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
            Affichage.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
            Affichage.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
    
    # Calcul des intégrales.
    A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E11_exp)
    B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_e_pg * E22_exp)
    C = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_e_pg * E22_exp + E22_e_pg * E11_exp)
    D = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_e_pg * E12_exp)

    if isinstance(f, float|int):
        uloc = u_n[assembly1D]
        E = np.einsum("ep,p,ei->", jacob1D_e_pg, poid1D_pg, uloc) * f * b
        pass
        E = np.sum(u_n[ddlsY_Upper]*f)
    else:
        E = np.sum(f_exp_bruit * u_n) 

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
        Eps_exp = simu._Calc_Epsilon_e_pg(u_exp_bruit, matriceType)
        E11_exp = Eps_exp[:,:,0]
        E22_exp = Eps_exp[:,:,1]
        E12_exp = Eps_exp[:,:,2]

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
            
            # m=8
            # n=5

            # p=8
            # q=5

            m=4
            n=3

            p=6
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

                # conditions u et v pour les noeuds renseignés:
                pass
                
                for noeud in noeuds:
                    
                    # conditions u
                    l += 1                
                    c = -1
                    for i in range(m+1):
                        for j in range(n+1):
                            c += 1
                            lignes.append(l)
                            colonnes.append(pos_Aij[c])
                            # values.append(xn[noeud]**i * yn[noeud]**j)                    
                            values.append((xn[noeud]/L)**i * (yn[noeud]/h)**j)
                    vectCond.append(condU)

                    # conditions v
                    l += 1                
                    c = -1
                    for i in range(p+1):
                        for j in range(q+1):
                            c += 1
                            lignes.append(l)
                            colonnes.append(pos_Bij[c])
                            # values.append(xn[noeud]**i * yn[noeud]**j)                    
                            values.append((xn[noeud]/L)**i * (yn[noeud]/h)**j)
                    vectCond.append(condV)

                return l

            
            lastLigne = Add_Conditions(nodesLower, 0, 0)

            const = H
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

                    # dx_ij_e_p = i*xn_e_g**(i-1) * yn_e_g**j
                    # dx_ij_e_p = i/L*(xn_e_g/L)**(i-1) * (yn_e_g/h)**j
                    dx_ij_e_p = i*(xn_e_g**(i-1)/l**i) * (yn_e_g/h)**j

                    # dy_ij_e_p = xn_e_g**i * j*yn_e_g**(j-1)
                    # dy_ij_e_p = (xn_e_g/L)**i * j/h*(yn_e_g/h)**(j-1)
                    dy_ij_e_p = (xn_e_g/l)**i * j*(yn_e_g**(j-1)/h**j)

                    if i > -1:
                        lignes.append(lastLigne + 1)
                        colonnes.append(pos_Aij[c])
                        cond1_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_exp * dx_ij_e_p)
                        values.append(cond1_A)

                        lignes.append(lastLigne + 3)
                        colonnes.append(pos_Aij[c])
                        cond3_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_exp * dx_ij_e_p)
                        values.append(cond3_A)

                    if j > -1:
                        lignes.append(lastLigne + 4)
                        colonnes.append(pos_Aij[c])
                        cond4_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_exp * dy_ij_e_p) * np.sqrt(2)/2
                        values.append(cond4_A)

            c = -1
            for i in range(p+1):
                for j in range(q+1):

                    c += 1

                    # dx_ij_e_p = i*xn_e_g**(i-1) * yn_e_g**j
                    # dx_ij_e_p = i/L*(xn_e_g/L)**(i-1) * (yn_e_g/h)**j
                    dx_ij_e_p = i*(xn_e_g**(i-1)/l**i) * (yn_e_g/h)**j

                    # dy_ij_e_p = xn_e_g**i * j*yn_e_g**(j-1)
                    # dy_ij_e_p = (xn_e_g/L)**i * j/h*(yn_e_g/h)**(j-1)
                    dy_ij_e_p = (xn_e_g/l)**i * j*(yn_e_g**(j-1)/h**j)

                    if j > -1:
                        lignes.append(lastLigne + 2)
                        colonnes.append(pos_Bij[c])
                        cond2_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E22_exp * dy_ij_e_p)
                        values.append(cond2_B)

                        lignes.append(lastLigne + 3)
                        colonnes.append(pos_Bij[c])
                        cond3_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E11_exp * dy_ij_e_p)
                        values.append(cond3_B)  

                    if i > -1:
                        lignes.append(lastLigne + 4)
                        colonnes.append(pos_Bij[c])
                        cond4_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, poid2D_pg, E12_exp * dx_ij_e_p) * np.sqrt(2)/2
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

            coord_e_n = np.mean(mesh.coordoGlob[mesh.connect], axis=1)
            x_e_n, y_e_n = coord_e_n[:,0], coord_e_n[:,1]

            ff = None
            # ff = -f            

            A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False, f=ff)
            A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False, f=ff)
            A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False, f=ff)
            A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False, f=ff)

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
            
            # # [Kim & al 2020]
            # u1 = lambda x, y: x/l  
            # v1 = lambda x, y: y/h
            # A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False)
            # E1 = -2*f

            # u1 = lambda x, y: x/l * (y-h)*(y+h)/h**2
            # u1 = lambda x, y: 0
            # v1 = lambda x, y: 0
            # u1 = lambda x, y: x/l

            u1 = lambda x, y: x/l
            v1 = lambda x, y: 0
            A1,B1,C1,D1,E1 = Get_A_B_C_D_E(u1, v1, pltSol=False)
            E1 = 0

            # axs = plt.subplots(ncols=2)[1]
            # Affichage.Plot_Result(simu, "Exy", nodeValues=False, ax=axs[0], title="num")
            # # rr = np.ones(mesh.Nn)
            # rr = np.ones(mesh.Nn)*1/2/l # 3*xn**2/l**3            
            # Affichage.Plot_Result(simu, rr, nodeValues=False, ax=axs[1], title="analytique")
            pass


            # ----------------------------------------------
            # Champ 2
            # ----------------------------------------------

            # # [Kim & al 2020]
            # u2 = lambda x, y: (x/l)**3  
            # v2 = lambda x, y: (y/h)**3
            # A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False)
            # E2 = -2*f
            
            # u2 = lambda x, y: (x/l)**3 * (y-h)*(y+h)/h**2

            u2 = lambda x, y: 0
            v2 = lambda x, y: y/h
            A2,B2,C2,D2,E2 = Get_A_B_C_D_E(u2, v2, pltSol=False)
            E2 = -2*f

            # ----------------------------------------------
            # Champ 3
            # ----------------------------------------------
            
            # # [Kim & al 2020]
            # u3 = lambda x,y: np.cos(np.pi/h*y) * np.sin(np.pi/l*x)
            # v3 = lambda x,y: y/h
            # A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False)
            # E3 = -2*f
            
            # u3 = lambda x,y: np.cos(np.pi/h*y) * np.sin(np.pi/l*x) * (y-h)*(y+h)/h**2

            u3 = lambda x,y: x/h
            v3 = lambda x, y: (y/h)**3
            A3,B3,C3,D3,E3 = Get_A_B_C_D_E(u3, v3, pltSol=False)
            # E3 = 0
            E3 = -2*f

            # ----------------------------------------------
            # Champ 4
            # ----------------------------------------------

            # # [Kim & al 2020]
            # u4 = lambda x,y: (y/h)**2 * (x/l)
            # v4 = lambda x,y: (y/h)**3
            # A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False)
            # E4 = -2*f

            # u4 = lambda x,y: (y/h)**2 * (x/l) * (y-h)*(y+h)/h**2
            # u4 = lambda x,y: x*y            
            # v4 = lambda x,y: 0
            # v4 = lambda x,y: x*y
            
            # u4 = lambda x,y: x*y            
            # v4 = lambda x,y: 0
            # A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False)
            # E4 = 0

            # Champ discontinu

            u4 = lambda x,y: 1
            v4 = lambda x,y: 1

            # u4 = lambda x,y: y/h
            # v4 = lambda x,y: x/h
            
            A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=True, nodes=nodesZone)
            E4 = 0
        
        # cc = -1
        # for u, v in zip([u1,u2,u3,u4], [v1,v2,v3,v4]):

        #     # opt = " Kim"
        #     opt = ""

        #     cc += 1

        #     # Calcul les déplacements associés aux champs virtuels.
        #     result = np.zeros((mesh.Nn, 2))
        #     result[:,0] = u(xn, yn)
        #     result[:,1] = v(xn, yn)
        #     u_n = result.reshape(-1)

        #     simu.set_u_n("displacement", u_n)
            
        #     ax = Affichage.Plot_Result(simu, "ux",title=" ", colorbarIsClose=True)[1]
        #     ax.axis("off")
        #     ax.set_title("$u_x^{" + f"*({cc+1})" + "}$")
        #     Affichage.Save_fig(folder, f"VFM_ux{cc+1}{opt}")

        #     ax = Affichage.Plot_Result(simu, "uy",title=" ", colorbarIsClose=True)[1]
        #     ax.axis("off")
        #     ax.set_title("$u_y^{" + f"*({cc+1})" + "}$")
        #     Affichage.Save_fig(folder, f"VFM_uy{cc+1}{opt}")

        #     ax = Affichage.Plot_Result(simu, "Exx",title=" ", colorbarIsClose=True)[1]
        #     ax.axis("off")
        #     ax.set_title("$\epsilon_{xx}^{" + f"*({cc+1})" + "}$")
        #     Affichage.Save_fig(folder, f"VFM_Exx{cc+1}{opt}")

        #     ax = Affichage.Plot_Result(simu, "Eyy", title=" ", colorbarIsClose=True)[1]
        #     ax.axis("off")
        #     ax.set_title("$\epsilon_{yy}^{" + f"*({cc+1})" + "}$")
        #     Affichage.Save_fig(folder, f"VFM_Eyy{cc+1}{opt}")

        #     ax = Affichage.Plot_Result(simu, "Exy",title=" ", colorbarIsClose=True)[1]
        #     ax.axis("off")
        #     ax.set_title("$\epsilon_{xy}^{" + f"*({cc+1})" + "}$")
        #     Affichage.Save_fig(folder, f"VFM_Exy{cc+1}{opt}")
        #     pass            
        

        # ----------------------------------------------
        # Résolution
        # ----------------------------------------------

        systMat = np.array([[A1, B1, C1, D1],
                            [A2, B2, C2, D2],
                            [A3, B3, C3, D3],
                            [A4, B4, C4, D4]])
        
        cond = np.linalg.norm(systMat) * np.linalg.norm(np.linalg.inv(systMat))

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

opt = " Kim"
# opt = ""

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
    axParam.set_ylabel(fr"${param} \ / \ {param}_{'{exp}'}$")
    axParam.grid()
    axParam.legend(loc="upper left")
    
    Affichage.Save_fig(folder, "VFM_"+param+opt, extension='pdf')

    print(f"{param} = {mean.mean()*paramExp}")




plt.show()