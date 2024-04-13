"""Material property identification in a biaxial test using a modified FEMU (Finite Element Updating Method)."""

from EasyFEA import (Display, Folder, plt, np, pd,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points, Circle

from scipy.optimize import least_squares

if __name__ == '__main__':

    Display.Clear()

    folder = Folder.New_File(Folder.Join("Identification","Biaxial"), results=True)

    # ----------------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------------
    useRescale = True
    noises = np.linspace(0, 0.02, 4)
    nRuns = 10
    tol = 1e-10

    L = 70 #mm
    h = 40
    r = -(L-h)/2
    ep = 0.5

    meshSize = h/10
    isHollow = True # circle is hollow

    pltMesh = True

    # ----------------------------------------------------------------------------
    # Mesh
    # ----------------------------------------------------------------------------
    pt1 = Point(-L/2, -L/2, r=r)
    pt2 = Point(L/2, -L/2, r=r)
    pt3 = Point(L/2, L/2, r=r)
    pt4 = Point(-L/2, L/2, r=r)

    contour = Points([pt1,pt2,pt3,pt4], meshSize)

    circle = Circle(Point(h/3, h/3), 10, meshSize, isHollow)

    mesh = Mesher().Mesh_2D(contour, [circle], ElemType.TRI3)

    if pltMesh:
        Display.Plot_Mesh(mesh)

    nodes_left = mesh.Nodes_Conditions(lambda x,y,z: x==-L/2)
    nodes_right = mesh.Nodes_Conditions(lambda x,y,z: x==L/2)
    nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==L/2)
    nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==-L/2)

    # ----------------------------------------------------------------------------
    # Material
    # ----------------------------------------------------------------------------
    tol0 = 1e-6
    bSup = np.inf

    EL = 3000
    ET = 6000
    GL = 1500
    vL = 0.25
    vT = 0.4

    axisL = np.array([0,1,0])
    axisT = np.array([1,0,0])

    dict_param = {
        "EL" : EL,
        "GL" : GL,
        "ET" : ET,
        "vL" : vL
    }

    El0 = EL * 10
    Gl0 = GL * 10
    Et0 = ET * 10
    vl0 = vL

    lb = [tol0]*4
    ub = (bSup, bSup, bSup, 0.5-tol0)
    bounds = (lb, ub)
    x0 = [El0, Gl0, Et0, vl0]

    material = Materials.Elas_IsotTrans(2, EL, ET, GL, vL, vT, axisL, axisT, True, ep)

    material_FEMU = Materials.Elas_IsotTrans(2, El0, Et0, Gl0, vl0, vT, axisL, axisT, True, ep)

    # ----------------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------------
    simu = Simulations.ElasticSimu(mesh, material)

    fexp = 1 # N
    simu.add_lineLoad(nodes_left, [-fexp/h], ['x'])
    simu.add_lineLoad(nodes_right, [fexp/h], ['x'])
    simu.add_lineLoad(nodes_upper, [fexp/h], ['y'])
    simu.add_lineLoad(nodes_lower, [-fexp/h], ['y'])

    # Display.Plot_BoundaryConditions(simu)

    u_exp = simu.Solve()
    simu.Save_Iter()

    # Display.Plot_Result(simu, "ux")
    # Display.Plot_Result(simu, "uy")
    # Display.Plot_Result(simu, "Sxx")
    # Display.Plot_Result(simu, "Syy")
    # Display.Plot_Result(simu, "Sxy")
    # Display.Plot_Result(simu, "Svm")

    # ----------------------------------------------------------------------------
    # Identification
    # ----------------------------------------------------------------------------
    Display.Section("Identification")

    # WARNING: Identification does not work if the simulation uses an iterative solver !
    simu_FEMU = Simulations.ElasticSimu(mesh, material_FEMU, useIterativeSolvers=False)

    def func(x):

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

            # noise on displacement
            uMax = np.abs(u_exp).mean()
            u_noise = uMax * (np.random.rand(u_exp.shape[0]) - 1/2) * noise
            u_exp_noise = u_exp + u_noise

            material_FEMU.El = El0
            material_FEMU.Gl = Gl0
            material_FEMU.Et = Et0
            material_FEMU.vl = vl0

            simu_FEMU.Bc_Init()

            Add_Dirichlet(nodes_left, ['x','y'])
            Add_Dirichlet(nodes_lower, ['x','y'])            
            
            # Add_Dirichlet(nodes_upper, ['x','y'])
            simu_FEMU.add_lineLoad(nodes_upper, [fexp/h], ["y"])

            # Add_Dirichlet(nodes_right, ['x','y'])
            simu_FEMU.add_lineLoad(nodes_right, [fexp/h], ["x"])

            if useRescale:
                # first solve the the simulation
                sol = simu_FEMU.Solve()
                K = simu_FEMU.Get_K_C_M_F()[0]
                # get the forces along x for nodes_right
                dofsX_right = simu.Bc_dofs_nodes(nodes_right, ['x'])
                fx_rigth = K[dofsX_right] @ sol
                # get the forces along y for nodes_upper
                dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ['y'])
                fy_upper = K[dofsY_upper] @ sol
                # calculate the scale factor along x y
                fxScale = np.sum(fx_rigth) / fexp
                fyScale = np.sum(fy_upper) / fexp
                # modified femu
                simu_FEMU.Bc_Init()
                Add_Dirichlet(nodes_left, ['x','y'])
                Add_Dirichlet(nodes_lower, ['x','y'])
                simu_FEMU.add_neumann(nodes_right, [fx_rigth*fxScale], ['x'])
                simu_FEMU.add_neumann(nodes_upper, [fy_upper*fyScale], ['y'])

            dofsKnown, dofsUnknow = simu_FEMU.Bc_dofs_known_unknow(simu_FEMU.problemType)        

            # res = least_squares(func, x0, bounds=bounds, verbose=2, ftol=tol, gtol=tol, xtol=tol, jac='3-point')
            res = least_squares(func, x0, bounds=bounds, verbose=0, ftol=tol, gtol=tol, xtol=tol)

            assert res.success, "Must converge"

            dict_run = {
                "run" : run
            }
            dict_run["EL"]=res.x[0]
            dict_run["GL"]=res.x[1]
            dict_run["ET"]=res.x[2]
            dict_run["vL"]=res.x[3]

            list_dict_run.append(dict_run)

        df_run = pd.DataFrame(list_dict_run)

        dict_noise = {
            "noise" : noise,
        }
        dict_noise["EL"] = df_run["EL"].values
        dict_noise["GL"] = df_run["GL"].values
        dict_noise["ET"] = df_run["ET"].values
        dict_noise["vL"] = df_run["vL"].values        

        list_dict_noise.append(dict_noise)
        
    Display.Plot_BoundaryConditions(simu_FEMU)

    df_noise = pd.DataFrame(list_dict_noise)

    # ----------------------------------------------------------------------------
    # Dislay
    # ----------------------------------------------------------------------------
    params = ["EL", "GL", "ET", "vL"]

    borne = 0.95
    bInf = 0.5 - (0.95/2)
    bSup = 0.5 + (0.95/2)

    print('\n')

    for param in params:

        axParam = Display.init_Axes()
        
        paramExp = dict_param[param]    

        nPertu = noises.size
        values = np.zeros((nPertu, nRuns))
        for p in range(nPertu):
            values[p] = df_noise[param].values[p]
        
        print(f"{param} = {values.mean():.2f}")

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

    err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp.reshape((mesh.Nn,2)), axis=1)
    # err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp)
    # err_n = np.linalg.norm(diff_n, axis=1)

    Display.Plot_Result(simu_FEMU, err_n, title=r"$\dfrac{\Vert u(p) - u^{exp} \Vert^2}{\Vert u^{exp} \Vert^2}$")

    # print(np.linalg.norm(diff_n)/np.linalg.norm(u_exp))


    plt.show()