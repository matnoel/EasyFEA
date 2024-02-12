import Display
from Interface_Gmsh import Mesher
from Geoms import Point, Points, Line, Domain, Circle
import Materials
import Simulations
import Folder
import PostProcessing

plt = Display.plt
np = Display.np

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    solve = True
    test = True
    optimMesh = True

    pltIter = False
    pltLoad = True
    makeMovie = True
    makeParaview = False

    # geom
    dim = 2
    useNotchCrack = False

    unit = 1e-3; # for mm [Guidault, Allix, Champaney, Cornuault, 2008, CMAME], [Miehe, Welschinger, Hofacker, 2010, IJNME], [Miehe, Hofacker, Welschinger, 2010, CMAME],[Passieux, Rethore, Gravouil, Baietto, 2013, CM]
    L = 8*unit # height
    L1 = 10*unit
    L2 = 9*unit
    ep = 0.5*unit
    nw = 0.05*unit # notch width mm
    diam = 0.5*unit # hole diameter
    
    e1 = 6*unit
    e2 = 1*unit

    # material
    # [Ambati, Gerasimov, De Lorenzis, 2015, CM]
    E = 20.8e9 # Pa
    v = 0.3

    # phase field
    split = "Miehe"
    regu = "AT2"
    Gc = 1e-3 # kN / mm
    Gc *= 1000*1000 # 1e3 N / m -> J/m2
    tolConv = 1e-0
    convOption = 2
    l0 = 0.025*unit # [Miehe, Welschinger, Hofacker, 2010, IJNME], [Miehe, Hofacker, Welschinger, 2010, CMAME], [Wu, Nguyen, 2018, JMPS], [Wu, Nguyen, Nguyen, Sutula, Bordas, Sinaie, 2019, AAM]
    l0 = L/120

    # loading
    if test:
        inc0 = 2e-3*unit
        Nt0 = 100        
        inc1 = 2e-4*unit
        Nt1 = 250
    else:
        inc0 = 1e-3*unit
        Nt0 = 200
        inc1 = 1e-4*unit
        Nt1 = 500
    # [Ambati, Gerasimov, De Lorenzis, 2015, CM]
    # du = 1e-3 mm during the first 200 time steps (up to u = 0.2 mm)
    # du = 1e-4 mm during the last  500 time steps (up to u = 0.25 mm)    

    disp1 = np.linspace(0, inc0*Nt0, Nt0)
    start = disp1[-1]
    disp2 = np.linspace(start, start+inc1*Nt1, Nt1)
    displacement = np.unique(np.concatenate([disp1, disp2]))

    name = "NotchedBeam_Benchmark"
    if dim == 3:
        name += '_3D'
    folder = Folder.New_File(name, results=True)

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------
    if test:
        hC = l0
    else:
        hC = l0/2

    nL = L/hC

    p0 = Point(0, L)
    p1 = Point(-L1, L)
    p2 = Point(-L1, 0)
    p3 = Point(-L2, 0)

    pC1 = Point(-e1-nw/2,0)
    pC2 = Point(-e1-nw/2,e2)
    pC3 = Point(-e1+nw/2,e2)
    pC4 = Point(-e1+nw/2,0)

    p4 = Point(L2, 0)
    p5 = Point(L1, 0)
    p6 = Point(L1, L)

    c1 = Circle(Point(-4*unit, 2.75*unit), diam, hC, True)
    c2 = Circle(Point(-4*unit, 4.75*unit), diam, hC, True)
    c3 = Circle(Point(-4*unit, 6.75*unit), diam, hC, True)

    if optimMesh:
        # zone rafinée
        z = e2 /2
        refineDomain = Domain(Point(-e1-z,0), Point(-4*unit+z,L), hC)
        # hD = hC*5
        # hD = 8*hC
        hD = 10*hC
    else:
        refineDomain = None
        # hD = 0.1*unit # 8 * hC -> 0.025*unit/2 * 8
        # hD = 0.2*unit # 8 * hC -> 0.025*unit/2 * 8

    if useNotchCrack:
        contour = Points([p0,p1,p2,p3,pC1,pC2,pC3,pC4,p4,p5,p6], hD)
        cracks = []
    else:
        contour = Points([p0,p1,p2,p3,p4,p5,p6], hD)
        cracks = [Line(Point(-e1,0, isOpen=True), Point(-e1,e2), hC, True)]

    inclusions = [c1, c2, c3]

    circlePos1 = Circle(p3, e2)
    circlePos2 = Circle(p4, e2)
    circlePos3 = Circle(p0, e2)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, inclusions, "TRI3", refineGeoms=[refineDomain], cracks=cracks)
    else:
        mesh = Mesher().Mesh_Extrude(contour, inclusions, [0,0,ep], [3], "HEXA8", refineGeoms=[refineDomain], cracks=cracks)

    Display.Plot_Mesh(mesh)
    # Display.Plot_Model(mesh)
    # Display.Plot_Nodes(mesh, mesh.Nodes_Line(cracks[0]), True)

    nodes_load = mesh.Nodes_Point(p0)
    nodes_fixed = np.concatenate([mesh.Nodes_Point(p3), mesh.Nodes_Point(p4)])

    nodes_c1 = mesh.Nodes_Cylinder(circlePos1, [0,0,ep])
    nodes_c2 = mesh.Nodes_Cylinder(circlePos2, [0,0,ep])
    nodes_c3 = mesh.Nodes_Cylinder(circlePos3, [0,0,ep])
    nodes_damage = np.concatenate([nodes_c1, nodes_c2, nodes_c3])



    # --------------------------------------------------------------------------------------------
    # Material
    # --------------------------------------------------------------------------------------------
    material = Materials.Elas_Isot(dim, E, v, False, ep)

    pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0)

    folderSimu = Folder.PhaseField_Folder(folder, "", pfm.split, pfm.regularization, "DP", tolConv, "", test, optimMesh, nL=nL)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    if solve:

        simu = Simulations.Simu_PhaseField(mesh, pfm)
        dofsY_load = simu.Bc_dofs_nodes(nodes_load, ['y'])
        
        if pltIter:
            __, axIter, cb = Display.Plot_Result(simu, 'damage')

            axLoad = plt.subplots()[1]
            axLoad.set_xlabel('displacement [mm]')
            axLoad.set_ylabel('load [kN]')    

        uMax = displacement[-1]
        load = []

        for iter, ud in enumerate(displacement):

            # add boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_damage, [0], ['d'], "damage")
            # simu.add_dirichlet(nodes_fixed, [0]*dim, simu.Get_directions())
            simu.add_dirichlet(nodes_fixed, [0], ['y'])
            simu.add_dirichlet(nodes_load, [-ud], ['y'])

            # solve
            u, d, Kglob, convergence = simu.Solve(tolConv, 500, convOption)

            # calc load
            fr = np.abs(np.sum(Kglob[dofsY_load,:] @ u))        
            load.append(fr)

            # print and save iter
            simu.Results_Set_Iteration_Summary(iter, ud*1e6, "µm", iter/displacement.size, True)        
            simu.Save_Iter()

            if pltIter:
                plt.figure(axIter.figure)
                cb.remove()
                cb = Display.Plot_Result(simu, 'damage', ax=axIter)[2]
                plt.pause(1e-12)

                plt.figure(axLoad.figure)
                axLoad.scatter(ud, fr, c='black')            
                plt.pause(1e-12)

            if not convergence:
                # stop if the simulation has not converged
                break
        
        # save load and displacement
        displacement = np.array(displacement)
        load = np.array(load)
        PostProcessing.Save_Load_Displacement(load, displacement, folderSimu)
        # save the simulation
        simu.Save(folderSimu)

        PostProcessing.Tic.Plot_History(folderSimu, True)    

    else:

        simu = Simulations.Load_Simu(folderSimu)
        mesh = simu.mesh

    load, displacement = PostProcessing.Load_Load_Displacement(folderSimu)

    # --------------------------------------------------------------------------------------------
    # PostProcessing
    # --------------------------------------------------------------------------------------------
    Display.Plot_BoundaryConditions(simu, folderSimu)

    Display.Plot_Result(simu, 'damage', folder=folderSimu)

    axLoad = plt.subplots()[1]
    axLoad.set_xlabel('displacement [mm]')
    axLoad.set_ylabel('load [kN]')
    axLoad.plot(displacement*1000, load/1000, c="blue")
    Display.Save_fig(folderSimu, "forcedep")

    Display.Plot_Iter_Summary(simu, folderSimu)

    if makeMovie:
        depMax = simu.Result("displacement_norm").max()
        facteur = 10*depMax
        PostProcessing.Make_Movie(folderSimu, 'damage', simu, deformation=False, factorDef=facteur, plotMesh=False)

    if makeParaview:
        PostProcessing.Make_Paraview(folderSimu, simu)

    plt.show()