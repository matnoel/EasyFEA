"""Identify material properties in a compression test using the virtual field method (VFM).
WARNING: The current implementation has not been validated.
"""

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import lstsq
from typing import Union

from EasyFEA import (Display, Folder, plt, np, pd,
                     Geoms, Mesher, ElemType,
                     Materials, Simulations)

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    folder = Folder.New_File(Folder.Join("Identification","VFM"), results=True)

    mat = "wood" # "wood", "steel"

    noises = np.linspace(0, 0.02, 5)
    nRuns = 100

    useSpecVirtual = False

    pltVerif = False

    L=45
    H=90

    l = L/2
    h = H/2

    b=20
    diam=10

    meshSize = H/70
    f = 40
    sig = f/(L*b)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    pt1 = Geoms.Point(-L/2, -H/2)
    pt2 = Geoms.Point(L/2, H/2)
    domain = Geoms.Domain(pt1, pt2, meshSize)
    pC = Geoms.Point(0, 0)
    circle = Geoms.Circle(pC, diam, meshSize, isHollow=True)

    pZone = pC + [diam/2,diam/2]
    circleZone = Geoms.Circle(pZone, diam*1.2, meshSize)

    coordInter = Geoms.Points_Intersect_Circles(circle, circleZone)

    pt1 = Geoms.Point(*coordInter[0, :])
    pt2 = Geoms.Point(*coordInter[1, :])
    circleArc1 = Geoms.CircleArc(pt1, pt2, pZone, meshSize=meshSize, coef=-1)

    mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI3, cracks=[circleArc1])
    xn = mesh.coord[:,0]
    yn = mesh.coord[:,1]

    Display.Plot_Tags(mesh)
    ax = Display.Plot_Mesh(mesh)
    ax.scatter(*coordInter[0, :2], zorder=5)
    ax.scatter(*coordInter[1, :2], zorder=5)
    ax.scatter(*circleArc1.pt3.coord[:2], zorder=5)

    nodes_edges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    nodes_lower = mesh.Nodes_Tags(["L0"])
    nodes_upper = mesh.Nodes_Tags(["L2"])

    nodes_contact = mesh.Nodes_Tags(["L0", "L2"])
    nodes_zone = mesh.Nodes_Circle(circleZone)

    # Recovery of arrays for digital integration
    matrixType = "rigi" 
    jacob2D_e_pg = mesh.Get_jacobian_e_pg(matrixType)
    weight2D_pg = mesh.Get_weight_pg(matrixType)

    groupElem1D = mesh.Get_list_groupElem(1)[0]
    elems1D = groupElem1D.Get_Elements_Nodes(nodes_upper)

    assembly1D = groupElem1D.Get_assembly_e(2)[elems1D]
    jacob1D_e_pg = groupElem1D.Get_jacobian_e_pg(matrixType)[elems1D]
    weight1D_pg = groupElem1D.Get_weight_pg(matrixType)

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    if mat == "steel":
        E_exp, v_exp = 210000, 0.3
        material = Materials.Elas_Isot(2, thickness=b, E=E_exp, v=v_exp)

        dict_param = {
            "lambda" : material.get_lambda(),
            "mu" : material.get_mu()
        }

    elif mat == "wood":
        EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3

        dict_param = {
            "EL" : EL_exp,
            "GL" : GL_exp,
            "ET" : ET_exp,
            "vL" : vL_exp
        }

        material = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
        axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), planeStress=True, thickness=b)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodes_lower, [0], ["y"])
    simu.add_dirichlet(mesh.Nodes_Tags((["P0"])), [0], ["x"])
    simu.add_surfLoad(nodes_upper, [-sig], ["y"])

    # Display.Plot_BoundaryConditions(simu)
    # Display.Save_fig(folder, "Boundary Compression")

    u_exp = simu.Solve()

    dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

    # ----------------------------------------------
    # Identification
    # ----------------------------------------------
    Display.Section("Identification")

    def Get_A_B_C_D_E(virtualX, virtualY, nodes=mesh.nodes, pltSol=False, f=None, pltEps=True):
        """Calculating integrals constants"""

        # Calculates displacements associated with virtual fields.
        result = np.zeros((mesh.Nn, 2))
        result[nodes,0] = virtualX(xn[nodes], yn[nodes])
        result[nodes,1] = virtualY(xn[nodes], yn[nodes])
        u_n = result.ravel()
        simu._Set_u_n("elastic", u_n)

        # Calculates deformations associated with virtual fields.
        Eps_e_pg = simu._Calc_Epsilon_e_pg(u_n, matrixType)
        E11_e_pg = Eps_e_pg[:,:,0]
        E22_e_pg = Eps_e_pg[:,:,1]
        E12_e_pg = Eps_e_pg[:,:,2]

        if pltSol:        
            Display.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, colorbarIsClose=True)
            Display.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, colorbarIsClose=True)
            if pltEps:
                Display.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
                Display.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
                Display.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
        
        # Calculating integrals.
        A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E11_e_pg * E11_exp)
        B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E22_e_pg * E22_exp)
        C = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E11_e_pg * E22_exp + E22_e_pg * E11_exp)
        D = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E12_e_pg * E12_exp)

        if isinstance(f, Union[float,int]):
            uloc = u_n[assembly1D]
            E = np.einsum("ep,p,ei->", jacob1D_e_pg, weight1D_pg, uloc) * f * b
            pass
            E = np.sum(u_n[dofsY_upper]*f)
        else:
            E = np.sum(f_exp_noise * u_n) 

        return A, B, C, D, E

    list_dict_noises = []

    for noise in noises:

        print(f"\nnoise = {noise}")

        list_dict_tirage = []

        for run in range(nRuns):

            print(f"run = {run}", end='\r')

            u_noise = np.abs(u_exp).max() * (np.random.rand(u_exp.shape[0]) - 1/2) * noise
            u_exp_noise = u_exp + u_noise

            # Recovering element deformations
            Eps_exp = simu._Calc_Epsilon_e_pg(u_exp_noise, matrixType)
            E11_exp = Eps_exp[:,:,0]
            E22_exp = Eps_exp[:,:,1]
            E12_exp = Eps_exp[:,:,2]

            # This information is not normally available in a real test!
            f_exp_noise = simu.Get_K_C_M_F()[0] @ u_exp_noise        

            if useSpecVirtual:

                # ----------------------------------------------
                # Special fields
                # ----------------------------------------------
                # print("\n")

                dimConditions = nodes_contact.size * 2  + 4

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

                lines = []        
                columns = []
                values = []
                vectCond = []

                def Add_Conditions(nodes: np.ndarray, condU: float, condV: float, l=-1):
                    # u and v conditions for nodes with information:
                    
                    for node in nodes:
                        
                        # conditions u
                        l += 1                
                        c = -1
                        for i in range(m+1):
                            for j in range(n+1):
                                c += 1
                                lines.append(l)
                                columns.append(pos_Aij[c])
                                # values.append(xn[noeud]**i * yn[noeud]**j)                    
                                values.append((xn[node]/L)**i * (yn[node]/h)**j)
                        vectCond.append(condU)

                        # conditions v
                        l += 1                
                        c = -1
                        for i in range(p+1):
                            for j in range(q+1):
                                c += 1
                                lines.append(l)
                                columns.append(pos_Bij[c])
                                # values.append(xn[noeud]**i * yn[noeud]**j)                    
                                values.append((xn[node]/L)**i * (yn[node]/h)**j)
                        vectCond.append(condV)

                    return l

                
                lastLine = Add_Conditions(nodes_lower, 0, 0)

                const = H
                lastLine = Add_Conditions(nodes_upper, 0, const, lastLine)

                condNodes = len(list(set(lines)))
                
                # deformation conditions
                coord_e_p = mesh.groupElem.Get_GaussCoordinates_e_p(matrixType)
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
                            lines.append(lastLine + 1)
                            columns.append(pos_Aij[c])
                            cond1_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E11_exp * dx_ij_e_p)
                            values.append(cond1_A)

                            lines.append(lastLine + 3)
                            columns.append(pos_Aij[c])
                            cond3_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E22_exp * dx_ij_e_p)
                            values.append(cond3_A)

                        if j > -1:
                            lines.append(lastLine + 4)
                            columns.append(pos_Aij[c])
                            cond4_A = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E12_exp * dy_ij_e_p) * np.sqrt(2)/2
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
                            lines.append(lastLine + 2)
                            columns.append(pos_Bij[c])
                            cond2_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E22_exp * dy_ij_e_p)
                            values.append(cond2_B)

                            lines.append(lastLine + 3)
                            columns.append(pos_Bij[c])
                            cond3_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E11_exp * dy_ij_e_p)
                            values.append(cond3_B)  

                        if i > -1:
                            lines.append(lastLine + 4)
                            columns.append(pos_Bij[c])
                            cond4_B = b * np.einsum('ep,p,ep->', jacob2D_e_pg, weight2D_pg, E12_exp * dx_ij_e_p) * np.sqrt(2)/2
                            values.append(cond4_B)

                
                matt = sparse.csr_matrix((values, (lines, columns)), (condNodes + 4, sizeA+sizeB))
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

                coord_e_n = np.mean(mesh.coordGlob[mesh.connect], axis=1)
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
                testY_Haut_1 = np.linalg.norm(np.mean(dd[nodes_upper, 1]) - const)/const
                # verifie que le champ est constant
                testY_Haut_2 = np.std(dd[nodes_upper, 1])/const
                assert testY_Haut_1 <= tol and testY_Haut_2 <= tol  

                # verifie que le champ vaut 0 en bas suivant y
                testYBas_1 = np.linalg.norm(np.mean(dd[nodes_lower, 1]))/const
                # verifie que le champ est constant
                testY_Bas_2 = np.std(dd[nodes_lower, 1])/const
                assert testYBas_1 <= tol and testY_Bas_2 <= tol  

                # verifie que le champ vaut 0 en haut suivant x
                testX_1 = np.linalg.norm(np.mean(dd[nodes_contact, 0]))/const
                testX_2 = np.std(dd[nodes_contact, 0])/const
                assert testX_1 <= tol and testX_2 <= tol  

                pass

            else:            
                
                # ----------------------------------------------
                # Field 1
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
                # Display.Plot_Result(simu, "Exy", nodeValues=False, ax=axs[0], title="num")
                # # rr = np.ones(mesh.Nn)
                # rr = np.ones(mesh.Nn)*1/2/l # 3*xn**2/l**3            
                # Display.Plot_Result(simu, rr, nodeValues=False, ax=axs[1], title="analytique")
                pass


                # ----------------------------------------------
                # Field 2
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
                # Field 3
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
                # Field 4
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

                # Discontinuous field
                u4 = lambda x,y: 1
                v4 = lambda x,y: 1

                # u4 = lambda x,y: y/h
                # v4 = lambda x,y: x/h
                
                A4,B4,C4,D4,E4 = Get_A_B_C_D_E(u4, v4, pltSol=False, nodes=nodes_zone)
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
            #     u_n = result.ravel()

            #     simu.set_u_n("elastic", u_n)
                
            #     ax = Display.Plot_Result(simu, "ux",title=" ", colorbarIsClose=True)
            #     ax.axis("off")
            #     ax.set_title("$u_x^{" + f"*({cc+1})" + "}$")
            #     Display.Save_fig(folder, f"VFM_ux{cc+1}{opt}")

            #     ax = Display.Plot_Result(simu, "uy",title=" ", colorbarIsClose=True)
            #     ax.axis("off")
            #     ax.set_title("$u_y^{" + f"*({cc+1})" + "}$")
            #     Display.Save_fig(folder, f"VFM_uy{cc+1}{opt}")

            #     ax = Display.Plot_Result(simu, "Exx",title=" ", colorbarIsClose=True)
            #     ax.axis("off")
            #     ax.set_title("$\epsilon_{xx}^{" + f"*({cc+1})" + "}$")
            #     Display.Save_fig(folder, f"VFM_Exx{cc+1}{opt}")

            #     ax = Display.Plot_Result(simu, "Eyy", title=" ", colorbarIsClose=True)
            #     ax.axis("off")
            #     ax.set_title("$\epsilon_{yy}^{" + f"*({cc+1})" + "}$")
            #     Display.Save_fig(folder, f"VFM_Eyy{cc+1}{opt}")

            #     ax = Display.Plot_Result(simu, "Exy",title=" ", colorbarIsClose=True)
            #     ax.axis("off")
            #     ax.set_title("$\epsilon_{xy}^{" + f"*({cc+1})" + "}$")
            #     Display.Save_fig(folder, f"VFM_Exy{cc+1}{opt}")
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

            cijMat = np.array([material.C[0,0], material.C[1,1], material.C[0,1], material.C[2,2]])

            error = np.abs(cijIdentif - cijMat)/cijMat

            c11 = cijIdentif[0]
            c22 = cijIdentif[1]
            c12 = cijIdentif[2]
            c33 = cijIdentif[3]

            matC_Identif = np.array([[c11, c12, 0],
                                    [c12, c22, 0],
                                    [0, 0, c33]])

            dict_run = {
                "run" : run
            }

            if mat == "steel":

                lamb = cijIdentif[2]
                mu = cijIdentif[3]/2

                dict_run["lambda"] = lamb
                dict_run["mu"] = mu

            elif mat == "wood":

                matS = np.linalg.inv(matC_Identif)

                Et = 1/matS[0,0]
                El = 1/matS[1,1]
                Gl = 1/matS[2,2]/2
                vl = - matS[0,1] * El

                dict_run["EL"] = El
                dict_run["GL"] = Gl
                dict_run["ET"] = Et
                dict_run["vL"] = vl

            list_dict_tirage.append(dict_run)

        df_tirage = pd.DataFrame(list_dict_tirage)

        dict_perturbation = {
            "noise" : noise,
        }

        if mat == "steel":
            dict_perturbation["lambda"] = df_tirage["lambda"].values
            dict_perturbation["mu"] = df_tirage["mu"].values
        elif mat == "wood":
            dict_perturbation["EL"] = df_tirage["EL"].values
            dict_perturbation["GL"] = df_tirage["GL"].values
            dict_perturbation["ET"] = df_tirage["ET"].values
            dict_perturbation["vL"] = df_tirage["vL"].values

        list_dict_noises.append(dict_perturbation)
        

    df_pertubation = pd.DataFrame(list_dict_noises)

    # ----------------------------------------------
    # Display
    # ----------------------------------------------

    if mat == "steel":
        params = ["lambda","mu"]
    elif mat == "wood":
        params = ["EL", "GL", "ET", "vL"]

    borne = 0.95
    bInf = 0.5 - (0.95/2)
    bSup = 0.5 + (0.95/2)

    print("\n")

    # opt = " Kim"
    opt = ""

    for param in params:

        axParam = Display.init_Axes()
        
        paramExp = dict_param[param]

        noises = df_pertubation["noise"]

        nPertu = noises.size
        values = np.zeros((nPertu, nRuns))
        for p in range(nPertu):
            values[p] = df_pertubation[param].values[p]
        values *= 1/paramExp

        mean = values.mean(axis=1)
        std = values.std(axis=1)

        paramInf, paramSup = tuple(np.quantile(values, (bInf, bSup), axis=1))

        axParam.plot(noises, [1]*nPertu, label=f"{param}_exp", c="black", ls='--')
        axParam.plot(noises, mean, label=f"{param}_moy")
        axParam.fill_between(noises, paramInf, paramSup, alpha=0.3, label=f"{borne*100} % ({nRuns} runs)")
        axParam.set_xlabel("noises")
        axParam.set_ylabel(fr"${param} \ / \ {param}_{'{exp}'}$")
        axParam.grid()
        axParam.legend(loc="upper left")
        
        Display.Save_fig(folder, param+opt, extension='pdf')

        print(f"{param} = {mean.mean()*paramExp}")




    plt.show()