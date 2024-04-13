"""Performs damage simulation for a plate with a hole subjected to compression."""

from EasyFEA import (Display, Folder, plt, np, Tic,
                     Mesher, ElemType, Mesh,
                     Materials, Simulations,
                     Paraview_Interface)
from EasyFEA.Geoms import Point, Domain, Circle

import multiprocessing

# Display.Clear()

useParallel = False
nProcs = 4 # number of processes in parallel

# ----------------------------------------------
# Configurations
# ----------------------------------------------
dim = 2
problem = "Benchmark" # ["Benchmark", "FCBA"]

test = True
solve = True
optimMesh = True

# Post processing
plotMesh = False
plotIter = True
plotResult = True
plotEnergy = False
showFig = True

saveParaview = False
makeMovie = False

# Material
materialType = "Elas_Isot" # ["Elas_Isot", "Elas_IsotTrans"]
solver = Materials.PhaseField.SolverType.History # ["History", "HistoryDamage", "BoundConstrain"]
maxIter = 1000
tolConv = 1e-0

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
# splits = ["Zhang"]
# splits = ["AnisotStrain","AnisotStress","Zhang"]
splits = ["Miehe"]

regus = ["AT2"] # ["AT1", "AT2"]
# regus = ["AT1", "AT2"]

# ----------------------------------------------
# Mesh
# ----------------------------------------------
def DoMesh(L: float, h: float, diam: float, thickness: float, l0: float, split: str) -> Mesh:

    clC = l0 if test else l0/2
    if optimMesh:
        clD = l0*4
        
        refineZone = diam*1.5/2
        if split in ["Bourdin", "Amor"]:
            refineGeom = Domain(Point(0, h/2-refineZone), Point(L, h/2+refineZone), clC)
        else:
            refineGeom = Domain(Point(L/2-refineZone, 0), Point(L/2+refineZone, h), clC)
    else:
        # clD = l0*2 if test else l0/2
        clD = l0 if test else l0/2
        refineGeom = None

    point = Point()
    domain = Domain(point, Point(L, h), clD)
    circle = Circle(Point(L/2, h/2), diam, clD, isHollow=True)

    if dim == 2:
        mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI3, refineGeoms=[refineGeom])
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(domain, [circle], [0,0,thickness], [4], ElemType.HEXA8,refineGeoms=[refineGeom])

    return mesh

# ----------------------------------------------
# Simu
# ----------------------------------------------

def DoSimu(split: str, regu: str):

    if "Benchmark" in problem:
        unitU = 'μm'
        unitF = 'kN/mm'
        unit = 1e6
        
        # geom
        L = 15e-3
        h = 30e-3
        thickness = 1 
        diam = 6e-3

        # material
        planeStress = False

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
        planeStress = True
        
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

    # folder name
    folderName = "PlateWithHole_" + problem    
    if dim == 3:
        folderName += "_3D"
    simpli2D = "CP" if planeStress else "DP"
    folder = Folder.PhaseField_Folder(folderName, materialType, split, regu, simpli2D, tolConv, solver, test, optimMesh, nL=nL)    
    
    if solve:

        mesh = DoMesh(L, h, diam, thickness, l0, split)

        # Get Nodes
        nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)
        nodes_x0y0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
        nodes_y0z0 = mesh.Nodes_Conditions(lambda x,y,z: (y==0) & (z==0))
        nodes_edges = mesh.Nodes_Tags(["L0","L1","L2","L3"])
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)
        
        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        if materialType == "Elas_Isot":
            E = 12e9
            v = 0.3
            material = Materials.Elas_Isot(dim, E, v, planeStress, thickness)
        elif materialType == "Elas_IsotTrans":
            El = 15585.5*1e6 # 12e9, 15585.5*1e6
            Et = 209.22*1e6 #500*1e6, 209.22*1e6
            Gl = 640.61*1e6 #450*1e6, 640.61*1e6
            vl = 0.02 #0.02, 0.3
            vt = 0.44
            v = 0
            axis_l = np.array([0,1,0])
            axis_t = np.array([1,0,0])
            material = Materials.Elas_IsotTrans(dim, El, Et, Gl, vl, vt,
                                                axis_l, axis_t, planeStress, thickness)

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        pfm = Materials.PhaseField(material, split, regu, gc, l0, solver=solver)
        simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)
        
        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        simu.Results_Set_Bc_Summary(u_max, listInc, listTresh, listOption)

        def Loading(ud: float):
            """Boundary conditions"""            
            
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodes_x0y0, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])
            if dim == 3:
                simu.add_dirichlet(nodes_y0z0, [0], ["z"])
        
        # INIT
        displacement = []
        force = []
        ud = -uinc0
        iter = 0
        nDetect = 0

        if plotIter:
            axIter = Display.Plot_Result(simu, "damage", nodeValues=True)

            force = np.asarray(force)
            displacement = np.asarray(displacement)
            figLoad, axLoad = Display.Plot_Force_Displacement(force/unit, displacement*unit, f'ud [{unitU}]', f'f [{unitF}]')

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
            
            displacement = np.concatenate((displacement, [ud]))
            force = np.concatenate((force, [f]))

            if plotIter:
                Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)
                plt.figure(axIter.figure)
                plt.pause(1e-12)

                force = np.asarray(force)
                displacement = np.asarray(displacement)
                Display.Plot_Force_Displacement(force/unit, displacement*unit, f'ud [{unitU}]', f'f [{unitF}]', ax=axLoad)[1]
                plt.figure(axLoad.figure)
                plt.pause(1e-12)

        force = np.asarray(force)
        displacement = np.asarray(displacement)

        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        print()
        Simulations.Save_Force_Displacement(force, displacement, folder)
        simu.Save(folder)
            
    else:
        # ----------------------------------------------
        # Load
        # ----------------------------------------------
        simu: Simulations.PhaseFieldSimu = Simulations.Load_Simu(folder)
        force, displacement = Simulations.Load_Force_Displacement(folder)

    # ----------------------------------------------
    # Post processing
    # ---------------------------------------------
    if plotEnergy:
        Display.Plot_Energy(simu, force, displacement, N=400, folder=folder)

    if plotResult:
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Iter_Summary(simu, folder, None, None)
        Display.Plot_Force_Displacement(force/unit, displacement*unit, f'ud [{unitU}]', f'f [{unitF}]', folder)
        Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=True, folder=folder, filename="damage")

    if plotMesh:
        Display.Plot_Mesh(mesh)

    if saveParaview:
        Paraview_Interface.Make_Paraview(simu, folder)

    if makeMovie:        
        Display.Movie_Simu(folder, "damage", simu, Niter=200, plotMesh=False, NiterFin=0, deformFactor=1.5)

    if solve:
        Tic.Plot_History(folder, details=False)

    if showFig:
        plt.show()
    
    Tic.Clear()
    plt.close('all')

if __name__ == "__main__":
    
    # generates configs
    Splits = []; Regus = []
    for split in splits.copy():
        for regu in regus.copy():
            Splits.append(split)
            Regus.append(regu)

    if useParallel:
        items = [(split, regu) for split, regu in zip(Splits, Regus)]        
        with multiprocessing.Pool(nProcs) as pool:
            for result in pool.starmap(DoSimu, items):
                pass
    else:
        [DoSimu(split, regu) for split, regu in zip(Splits, Regus)]