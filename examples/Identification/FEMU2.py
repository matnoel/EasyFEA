"""Material property identification in a compression test using a modified FEMU (Finite Element Updating Method)."""

from EasyFEA import (Display, Folder, plt, np, pd,
                     Geoms, Mesher, ElemType,
                     Materials, Simulations)

from scipy.optimize import least_squares

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    noises = np.linspace(0, 0.02, 4)
    nRuns = 10
    tol = 1e-10

    folder = Folder.New_File(Folder.Join("Identification","PlateWithHole"), results=True)

    pltVerif = False
    useRescale = True

    l = 45
    h = 90
    b = 20
    d = 10

    meshSize = l/15
    elemType = ElemType.TRI3

    mat = "wood" # "steel" "wood"

    f=40
    sig = f/(l*b)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    pt1 = Geoms.Point()
    pt2 = Geoms.Point(l, 0)
    pt3 = Geoms.Point(l, h)
    pt4 = Geoms.Point(0, h)
    points = Geoms.Points([pt1, pt2, pt3, pt4], meshSize)

    circle = Geoms.Circle(Geoms.Point(l/2, h/2), d, meshSize, isHollow=True)

    mesh = Mesher().Mesh_2D(points, [circle], elemType)

    nodes_contact = mesh.Nodes_Tags(["L0", "L2"])
    nodes_p0 = mesh.Nodes_Tags(["P0"])
    nodes_lower = mesh.Nodes_Tags(["L0"])
    nodes_upper = mesh.Nodes_Tags(["L2"])

    # Display.Plot_Mesh(mesh)
    # Display.Plot_Tags(mesh)
    # Display.Plot_Nodes(mesh, nodesX0)

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    tol0 = 1e-6
    bSup = np.inf

    if mat == "steel":
        E_exp, v_exp = 210000, 0.3
        material = Materials.Elas_Isot(2, thickness=b)

        dict_param = {
            "E" : E_exp,
            "v" : v_exp
        }

        Emax=300000
        vmax=0.49
        E0, v0 = Emax, vmax
        x0 = [E0, v0]
        
        material_FEMU = Materials.Elas_Isot(2, E0, v0, thickness=b)
        bounds=([tol0]*2, [bSup, vmax])

    elif mat == "wood":
        EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3

        dict_param = {
            "EL" : EL_exp,
            "GL" : GL_exp,
            "ET" : ET_exp,
            "vL" : vL_exp
        }
        
        EL0 = EL_exp * 10
        GL0 = GL_exp * 10
        ET0 = ET_exp * 10
        vL0 = vL_exp

        lb = [tol0]*4
        ub = (bSup, bSup, bSup, 0.5-tol0)
        bounds = (lb, ub)
        x0 = [EL0, GL0, ET0, vL0]

        material = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
        axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), planeStress=True, thickness=b)

        material_FEMU = Materials.Elas_IsotTrans(2, El=EL0, Et=ET0, Gl=GL0, vl=vL0, vt=0.3,
        axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), planeStress=True, thickness=b)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodes_lower, [0], ["y"])
    simu.add_dirichlet(nodes_p0, [0], ["x"])
    simu.add_surfLoad(nodes_upper, [-sig], ["y"])

    # Display.Plot_BoundaryConditions(simu)

    u_exp = simu.Solve()

    # Display.Plot_Result(simu, "uy")
    # Display.Plot_Result(simu, "Syy", coef=1/sig, nodeValues=False)
    # Display.Plot_Result(simu, np.linalg.norm(vectRand.reshape((mesh.Nn), 2), axis=1), title="bruit")
    # Display.Plot_Result(simu, u_exp.reshape((mesh.Nn,2))[:,1], title='uy bruit')

    dofsX = simu.Bc_dofs_nodes(nodes_contact, ["x"])
    dofsY = simu.Bc_dofs_nodes(nodes_contact, ["y"])

    assert nodes_contact.size*2 == (dofsX.size + dofsY.size)

    if useRescale:
        dofsX_lower = simu.Bc_dofs_nodes(nodes_lower, ["x"])
        dofsY_lower = simu.Bc_dofs_nodes(nodes_lower, ["y"])    
        
        dofsX_upper = simu.Bc_dofs_nodes(nodes_upper, ["x"])
        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])
        dofsXY_upper = simu.Bc_dofs_nodes(nodes_upper, ["x","y"])

    # ----------------------------------------------
    # Identification
    # ----------------------------------------------
    Display.Section("Identification")

    # WARNING: Identification does not work if the simulation uses an iterative solver !
    simu_FEMU = Simulations.ElasticSimu(mesh, material_FEMU, useIterativeSolvers=False)

    def func(x):
        # cost function

        # update params
        if mat == "steel":
            # x0 = [E0, v0]
            E = x[0]
            v = x[1]
            material_FEMU.E = E
            material_FEMU.v = v
        elif mat == "wood":
            # x0 = [EL0, GL0, ET0, vL0]
            material_FEMU.El = x[0]
            material_FEMU.Gl = x[1]
            material_FEMU.Et = x[2]
            material_FEMU.vl = x[3]

        u = simu_FEMU.Solve()
        
        diff = u - u_exp_noise
        diff = diff[dofsUnknow]

        return diff

    def Add_Dirichlet(nodes: np.ndarray, directions=["x","y"]):


        dofs = simu_FEMU.Bc_dofs_nodes(nodes, directions)
        nDim = len(directions)

        values = u_exp_noise[dofs]
        values = np.reshape(values, (-1,nDim))
        list_values = [values[:,d] for d in range(nDim)]

        simu_FEMU.add_dirichlet(nodes, list_values, directions)

    # dictionary list that will contain for the various disturbances the
    # properties identified
    list_dict_noise = []

    for noise in noises:

        print(f"\nnoise = {noise:.2e}")

        list_dict_run = []

        for run in range(nRuns):

            print(f"run = {run+1}", end='\r')
            
            uMax = np.abs(u_exp).mean()
            u_noise = uMax * (np.random.rand(u_exp.shape[0]) - 1/2) * noise        
            u_exp_noise = u_exp + u_noise

            if mat == "steel":
                material_FEMU.E = E0
                material_FEMU.v = v0
            elif mat == "wood":
                material_FEMU.El = EL0
                material_FEMU.Gl = GL0
                material_FEMU.Et = ET0
                material_FEMU.vl = vL0
            
            simu_FEMU.Bc_Init()
            
            if useRescale:
                # here, when the solution is noisy, the calculated force vector is no longer correct
                # identification by rescalating the vector can therefore not work
                # option 2 must be chosen
                # with option 2, we assume that the load distribution on the upper surface is homogeneous
                
                # optionRescale = 0 # dirichlet x and neumann on y nodes_upper
                optionRescale = 1 # dirichlet x,y low and neumann x,y nodes_upper
                # optionRescale = 2 # dirichlet x,y nodes_lower, dirichlet x haut and surfload y nodes_upper
                
                K = simu_FEMU.Get_K_C_M_F()[0]

                if optionRescale == 0:
                    # take only the following ddls y
                    f_dofs = K[dofsY_upper,:] @ u_exp_noise
                    f_r = - f_dofs.copy()                

                elif optionRescale == 1:
                    # take the dofs according to x and y
                    f_dofs = K[dofsXY_upper,:] @ u_exp_noise
                    f_dofs = f_dofs.reshape(-1,2)

                    f_r = np.sum(-f_dofs, 0)[1] # force following y

                if optionRescale in [0, 1]:

                    correct = f / np.sum(f_r)

                    f_dofs *= correct

                    verifF =  np.sum(f_r*correct) - f
                    assert np.abs(verifF) <= 1e-10

                if optionRescale == 0:
                    # apply the following displacements x to the surfaces in contact with the jaws
                    Add_Dirichlet(nodes_contact, ['x'])
                    # applies the following displacements to the lower surface y
                    Add_Dirichlet(nodes_lower, ['y'])
                    # applies the following corrected force vector to the upper surface y
                    simu_FEMU.add_neumann(nodes_upper, [f_dofs], ["y"])

                elif optionRescale == 1:
                    # applied to surfaces in contact with the bottom tray
                    Add_Dirichlet(nodes_lower, ['x','y'])
                    # applies the following corrected force vector to the upper surface y
                    simu_FEMU.add_neumann(nodes_upper, [f_dofs[:,0], f_dofs[:,1]], ["x","y"])

                elif optionRescale == 2:
                    # applied to surfaces in contact with the bottom tray
                    Add_Dirichlet(nodes_lower, ['x','y'])
                    # applies y displacements to top nodes
                    Add_Dirichlet(nodes_upper, ['x'])
                    # applies surface load
                    simu_FEMU.add_surfLoad(nodes_upper, [-sig], ["y"])                
            else:
                # apply dofs on edge
                Add_Dirichlet(nodes_contact, ['x','y'])

            dofsKnown, dofsUnknow = simu_FEMU.Bc_dofs_known_unknow(simu_FEMU.problemType)

            # res = least_squares(func, x0, bounds=bounds, verbose=2, ftol=tol, gtol=tol, xtol=tol, jac='3-point')
            res = least_squares(func, x0, bounds=bounds, verbose=0, ftol=tol, gtol=tol, xtol=tol)

            dict_run = {
                "run" : run
            }
            if mat == "steel":
                dict_run["E"]=res.x[0]
                dict_run["v"]=res.x[1]
            elif mat == "wood":
                dict_run["EL"]=res.x[0]
                dict_run["GL"]=res.x[1]
                dict_run["ET"]=res.x[2]
                dict_run["vL"]=res.x[3]

            list_dict_run.append(dict_run)

        df_run = pd.DataFrame(list_dict_run)

        dict_noise = {
            "noise" : noise,
        }
        if mat == "steel":
            dict_noise["E"] = df_run["E"].values
            dict_noise["v"] = df_run["v"].values
        elif mat == "wood":
            dict_noise["EL"] = df_run["EL"].values
            dict_noise["GL"] = df_run["GL"].values
            dict_noise["ET"] = df_run["ET"].values
            dict_noise["vL"] = df_run["vL"].values

        list_dict_noise.append(dict_noise)        

    df_noise = pd.DataFrame(list_dict_noise)

    # ----------------------------------------------
    # Display
    # ----------------------------------------------
    if mat == "steel":
        params = ["E","v"]
    elif mat == "wood":
        params = ["EL", "GL", "ET", "vL"]

    borne = 0.95
    bInf = 0.5 - (0.95/2)
    bSup = 0.5 + (0.95/2)

    for param in params:

        axParam = Display.init_Axes()
        
        paramExp = dict_param[param]
        
        nPertu = noises.size
        values = np.zeros((nPertu, nRuns))
        for p in range(nPertu):
            values[p] = df_noise[param].values[p]
        
        print(f"{param} = {values.mean()}")

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
        
        Display.Save_fig(folder, "FEMU_"+param, extension='pdf')

    diff_n = np.reshape(simu_FEMU.displacement - u_exp, (mesh.Nn, 2))

    err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp.reshape((mesh.Nn,2)), axis=1).max()
    # err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp)
    # err_n = np.linalg.norm(diff_n, axis=1)

    Display.Plot_Result(simu_FEMU, err_n, title=r"$\dfrac{\Vert u(p) - u_{exp} \Vert^2}{\Vert u_{exp} \Vert^2}$")

    # print(np.linalg.norm(diff_n)/np.linalg.norm(u_exp))

    plt.show()
