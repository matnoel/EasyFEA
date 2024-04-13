"""Identify material properties in a bending test using the virtual field method (VFM).
WARNING: The current implementation has not been validated.
"""

from EasyFEA import (Display, Folder, plt, np, pd,
                     Geoms, Mesher, ElemType,
                     Materials, Simulations)

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    pltVerif = False

    noise = 0.02

    L=120
    h=13
    b=13
    meshSize = h/1
    p = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    pt1 = Geoms.Point(0,-h/2)
    pt2 = Geoms.Point(L, h/2)
    domain = Geoms.Domain(pt1, pt2, meshSize)

    mesh = Mesher().Mesh_2D(domain,[],ElemType.TRI10)
    xn = mesh.coord[:,0]
    yn = mesh.coord[:,1]

    nodesEdge = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    nodesX0 = mesh.Nodes_Tags(["L3"])
    nodesXL = mesh.Nodes_Tags(["L1"])

    # Recovery of arrays for digital integration
    matrixType = "rigi" 
    jacob2D_e_pg = mesh.Get_jacobian_e_pg(matrixType)
    poid2D_pg = mesh.Get_weight_pg(matrixType)

    groupElem1D = mesh.Get_list_groupElem(1)[0]
    jacob1D_e_pg = groupElem1D.Get_jacobian_e_pg(matrixType)
    poid1D_pg = groupElem1D.Get_weight_pg(matrixType)

    assembly1D_e = groupElem1D.Get_assembly_e(2)

    # ----------------------------------------------
    # Material and simulation
    # ----------------------------------------------
    E_exp, v_exp = 210000, 0.3 
    material = Materials.Elas_Isot(2, thickness=b, E=E_exp, v=v_exp)

    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodesX0, [0,0], ["x","y"])
    simu.add_surfLoad(nodesXL, [-p/b/h], ["y"])
    # simu.add_surfLoad(nodesXL, [-p/b/h], ["x"])

    # Display.Plot_BoundaryConditions(simu)

    u_exp = simu.Solve()
    f_exp = simu._Solver_Apply_Neumann("elastic").toarray().ravel()

    forceR = np.sum(f_exp)

    forces = simu.Get_K_C_M_F()[0] @ u_exp
    dofs = simu.Bc_dofs_nodes(mesh.nodes, ["y"])

    f_exp_loc = f_exp[assembly1D_e]

    # ----------------------------------------------
    # Identification
    # ----------------------------------------------
    Display.Section("Identification")

    uMax = np.abs(u_exp).mean()
    u_noise = uMax * (np.random.rand(u_exp.shape[0]) - 1/2) * noise
    u_exp_noise = u_exp + u_noise

    # Recovering element deformations
    Eps_exp = simu._Calc_Epsilon_e_pg(u_exp_noise, matrixType)
    E11_exp = Eps_exp[:,:,0]
    E22_exp = Eps_exp[:,:,1]
    E12_exp = Eps_exp[:,:,2]

    # Display.Plot_Result(simu, "Exx", nodeValues=False)
    # Display.Plot_Result(simu, "Eyy", nodeValues=False)
    # Display.Plot_Result(simu, "Exy", nodeValues=False)

    def Get_A_B_C_D_E(virtualX, virtualY, pltSol=False):
        # Function that returns calculated integrals

        # Calculates displacements associated with virtual fields.
        result = np.zeros((mesh.Nn, 2))
        result[:,0] = virtualX(xn, yn)
        result[:,1] = virtualY(xn, yn)
        u_n = result.ravel()
        simu._Set_u_n("elastic", u_n)

        # Calculates deformations associated with virtual fields.
        Eps_e_pg = simu._Calc_Epsilon_e_pg(u_n, matrixType)
        E11_e_pg = Eps_e_pg[:,:,0]
        E22_e_pg = Eps_e_pg[:,:,1]
        E12_e_pg = Eps_e_pg[:,:,2]

        if pltSol:        
            Display.Plot_Result(simu, "ux", title=r"$u_x^*$", plotMesh=True, deformFactor=0.0)
            Display.Plot_Result(simu, "uy", title=r"$u_y^*$", plotMesh=True, deformFactor=0.0)
            Display.Plot_Result(simu, "Exx", title=r"$\epsilon_{xx}^*$", nodeValues=False, plotMesh=True)
            Display.Plot_Result(simu, "Eyy", title=r"$\epsilon_{yy}^*$", nodeValues=False, plotMesh=True)
            Display.Plot_Result(simu, "Exy", title=r"$\epsilon_{xy}^*$", nodeValues=False, plotMesh=True)
        
        # Calculating integrals.
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

    mu = material.get_mu()
    lamb = material.get_lambda()

    A2,B2,C2,D2,E2 = Get_A_B_C_D_E(lambda x, y: x*y, lambda x, y: 0, pltSol=False)
    c11 = - c33 * (D2-C2)/(A2+B2+C2)

    systMat = np.array([[A1, B1, C1, D1],[A2, B2, C2, D2],[1,-1,0,0],[1,0,-1,-1]])

    # c11 c22 c12 c33

    # cijIdentif = np.linalg.solve(systMat, [-800*120,0,0,0])
    cijIdentif = np.linalg.solve(systMat, [E1,E2,0,0])

    cijMat = np.array([material.C[0,0], material.C[1,1], material.C[0,1], material.C[2,2]])

    error = np.abs(cijIdentif - cijMat)/np.linalg.norm(cijMat)

    print(error)

    plt.show()