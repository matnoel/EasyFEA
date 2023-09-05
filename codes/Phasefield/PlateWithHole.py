from TicTac import Tic
import Materials
from Geom import *
import Display as Display
from Interface_Gmsh import Interface_Gmsh, ElemType
from Mesh import Mesh, Calc_projector
import Simulations
import PostProcessing as PostProcessing
import Folder

import matplotlib.pyplot as plt

# Display.Clear()

# ----------------------------------------------
# Simulation
# ----------------------------------------------
dim = 2
problem = "Benchmark" # ["Benchmark", "FCBA"]

test = True
solve = True

# ----------------------------------------------
# Post processing
# ----------------------------------------------
plotMesh = False
plotIter = False
plotResult = True
plotEnergy = False
showFig = True

saveParaview = True; NParaview=300
makeMovie = False; NMovie = 200

# ----------------------------------------------
# Material
# ----------------------------------------------
materialType = "Elas_Isot" # ["Elas_Isot", "Elas_IsotTrans"]
solver = Materials.PhaseField_Model.SolverType.History # ["History", "HistoryDamage", "BoundConstrain"]
maxIter = 1000
tolConv = 1e-0

# ----------------------------------------------
# Mesh Option
# ----------------------------------------------
optimMesh = False
damagedNodes = []

# ----------------------------------------------
# Configurations
# ----------------------------------------------

# for tolConv in [1e-0, 1e-1, 1e-2]:
#     split = "Zhang"

# splits = ["Zhang"]
splits = ["Miehe"]
# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]

regus = ["AT2"] # ["AT1", "AT2"]
# regus = ["AT1", "AT2"]

Splits = []; Regus = []
for split in splits.copy():
    for regu in regus.copy():
        Splits.append(split)
        Regus.append(regu)

for split, regu in zip(Splits, Regus):

    # ----------------------------------------------
    # config options
    # ----------------------------------------------
    if "Benchmark" in problem:
        unitU = 'μm'
        unitF = 'kN/mm'
        unit = 1e3
        
        # geom
        L = 15e-3
        h = 30e-3
        thickness = 1 
        diam = 6e-3

        # material
        simpli2D = "DP" # ["CP","DP"]

        # phase field
        gc = 1.4
        nL = 0
        l0 = 0.12e-3

        # loading
        treshold = 0.6
        if materialType == "Elas_Isot":
            # u_max = 25e-6
            u_max = 35e-6
        else:
            u_max = 80e-6
        
        uinc0 = 8e-8; uinc1 = 2e-8
        listInc = [uinc0, uinc1]
        listTresh = [0, treshold]
        listOption = ["damage"]*len(listTresh)

    elif "FCBA" in problem:
        unitU = 'mm'
        unitF = 'kN'
        unit = 1e3

        # geom
        L = 4.5e-2
        h = 9e-2
        thickness = 2e-2
        diam = 1e-2
        r = diam/2

        # material
        simpli2D = "CP" # ["CP","DP"]
        
        # phase field
        
        gc = 0.07 # mJ/mm2
        # 1 J -> 1000 mJ
        # 1 m2 -> 1e6 mm2
        gc *= 1e-3 * 1e6 # J/m2

        # * 1000
        nL = 100
        # nL = 180
        l0 = L/nL

        # loading
        treshold = 0.2
        u_max = 1e-3
        
        uinc0 = 8e-6; uinc1 = 2e-6
        listInc = [uinc0, uinc1]
        listTresh = [0, treshold]
        listOption = ["damage"]*2

    # ----------------------------------------------
    # meshSize
    # ----------------------------------------------
    if test:
        if optimMesh:            
            clD = l0*4
            clC = l0
        else:
            if "Benchmark" in problem:
                clD = 0.25e-3
                clC = 0.12e-3
            else:
                clD = l0
                clC = l0
    else:        
        if optimMesh:
            clD = l0*4
            clC = l0/2
        else:
            clD = l0/2
            clC = l0/2

    # ----------------------------------------------
    # Elastic material
    # ----------------------------------------------
    if dim == 2 and simpli2D == "CP":
        planeStress = True
    else:
        planeStress = False

    if materialType == "Elas_Isot":
        E = 12e9
        v = 0.3
        material = Materials.Elas_Isot(dim, E, v, planeStress, thickness)

    elif materialType == "Elas_IsotTrans":        
        El = 12e9
        Et = 500*1e6
        Gl = 450*1e6
        vl = 0.02
        vt = 0.44
        v = 0
        axis_l = np.array([0,1,0])
        axis_t = np.array([1,0,0])
        material = Materials.Elas_IsotTrans(dim, El, Et, Gl, vl, vt,
                                            axis_l, axis_t, planeStress, thickness)

    # folder name
    if dim == 3:
        problem += "_3D"
    folderName = "PlateWithHole_" + problem
    folder = Folder.PhaseField_Folder(folderName, materialType, split, regu, simpli2D, tolConv, solver, test, optimMesh, nL=nL)
    
    if solve:

        # ----------------------------------------------
        # Mesh
        # ----------------------------------------------
        if optimMesh:
            refineZone = diam*1.5/2
            if split in ["Bourdin", "Amor"]:
                refineGeom = Domain(Point(0, h/2-refineZone), Point(L, h/2+refineZone), clC)
            else:
                refineGeom = Domain(Point(L/2-refineZone, 0), Point(L/2+refineZone, h), clC)
        else:
            refineGeom = None

        point = Point()
        domain = Domain(point, Point(L, h), clD)
        circle = Circle(Point(L/2, h/2), diam, clD, isHollow=True)

        if dim == 2:
            mesh = Interface_Gmsh().Mesh_2D(domain, [circle],
                                            ElemType.TRI3, refineGeoms=[refineGeom])
        elif dim == 3:
            mesh = Interface_Gmsh().Mesh_3D(domain, [circle], [0,0,thickness], 4,
                                            ElemType.HEXA8,refineGeoms=[refineGeom])
                    
        if plotMesh:
            Display.Plot_Mesh(mesh)
            plt.show()

        nodes_edges = mesh.Nodes_Tags(["L0","L1","L2","L3"])
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)        

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        pfm = Materials.PhaseField_Model(material, split, regu, gc, l0, solver=solver)
        simu = Simulations.Simu_PhaseField(mesh, pfm, verbosity=False)

        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])
        
        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        simu.Results_Set_Bc_Summary(u_max, listInc, listTresh, listOption)        

        def Loading(ud: float):
            """Boundary conditions"""
            
            # Get Nodes
            nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
            nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)            
            nodes_x0y0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
            nodes_y0z0 = mesh.Nodes_Conditions(lambda x,y,z: (y==0) & (z==0))

            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodes_x0y0, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])
            if dim == 3:
                simu.add_dirichlet(nodes_y0z0, [0], ["z"])
        
        # INIT
        displacement = []
        load = []
        ud = -uinc0
        iter = 0
        nDetect = 0

        if plotIter:
            figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True)

            arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)
            figLoad, axLoad = Display.Plot_Load_Displacement(arrayDisplacement*unit, arrayLoad/unit,
                                                             f'ud [{unitU}]', f'f [{unitF}]')

        while ud <= u_max:

            iter += 1
            if simu.damage.max() < treshold:
                ud += uinc0
            else:
                ud += uinc1

            if dim == 3 and iter > 1500:
                break
            
            Loading(ud)

            u, d, Kglob, convergence = simu.Solve(tolConv, maxIter)
            simu.Save_Iter()

            # stop if the simulation does not converge
            if not convergence: break            

            f = np.sum(Kglob[dofsY_upper, :] @ u)

            simu.Results_Set_Iteration_Summary(iter, ud*unit, unitU, ud/u_max, True)

            # Detection if the edges has been touched
            if np.any(d[nodes_edges] >= 0.98):
                nDetect += 1
                if nDetect == 10:                    
                    break

            displacement.append(ud)
            load.append(f)

            if plotIter:
                cb.remove()
                figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)
                plt.figure(figIter)
                plt.pause(1e-12)

                arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)
                axLoad = Display.Plot_Load_Displacement(arrayDisplacement*unit, arrayLoad/unit, f'ud [{unitU}]', f'f [{unitF}]')[1]
                plt.figure(axLoad.figure)
                plt.pause(1e-12)

        load = np.array(load)
        displacement = np.array(displacement)

        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        print()
        PostProcessing.Save_Load_Displacement(load, displacement, folder)
        simu.Save(folder)
            
    else:
        # ----------------------------------------------
        # Load
        # ----------------------------------------------
        load, displacement = PostProcessing.Load_Load_Displacement(folder)
        simu = Simulations.Load_Simu(folder)

    # ----------------------------------------------
    # Post processing
    # ---------------------------------------------
    if plotEnergy:
        Display.Plot_Energy(simu, load, displacement, Niter=400, folder=folder)

    if plotResult:
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Iter_Summary(simu, folder, None, None)
        Display.Plot_Load_Displacement(displacement*unit, load/unit, f'ud [{unitU}]', f'f [{unitF}]', folder)
        Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=True, folder=folder, filename="damage")

    if saveParaview:
        PostProcessing.Make_Paraview(folder, simu, Niter=NParaview)        

    if makeMovie:        
        PostProcessing.Make_Movie(folder, "damage", simu, Niter=NMovie, plotMesh=False, deformation=False, NiterFin=0, factorDef=1.5)

    if solve:
        Tic.Plot_History(folder, details=True)
    else:        
        Tic.Plot_History(details=True)

    if showFig:
        plt.show()
    
    Tic.Clear()
    plt.close('all')

    if solve:
        del simu
        del mesh
    else:        
        del simu