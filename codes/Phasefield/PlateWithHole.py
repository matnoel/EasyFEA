from TicTac import Tic
import Materials
from BoundaryCondition import BoundaryCondition
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
problem = "Benchmark" # ["Benchmark", "FCBA"]
dim = 2
if dim == 3:
    problem += "_3D"

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

saveParaview = False; NParaview=300
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
optimMesh = True
updateMesh = False # non-validated implementation
damagedNodes = []

# ----------------------------------------------
# Configurations
# ----------------------------------------------

# for tolConv in [1e-0, 1e-1, 1e-2]:
#     split = "Zhang"

splits = ["Bourdin"]
# splits = ["He"]
# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]

regularisations = ["AT1"] # ["AT1", "AT2"]
# regularisations = ["AT1", "AT2"]

nSplit = len(splits)
nRegu = len(regularisations)

regularisations = regularisations * nSplit
splits = np.repeat(splits, nRegu)

for split, regu in zip(splits, regularisations):

    # ----------------------------------------------
    # config options
    # ----------------------------------------------
    if "Benchmark" in problem:
        
        # geom
        L = 15e-3
        h = 30e-3
        ep = 1 
        diam = 6e-3

        # material
        simpli2D = "DP" # ["CP","DP"]

        # phase field
        gc = 1.4
        nL = 0
        l0 = 0.12e-3

        # loading
        if materialType == "Elas_Isot":
            # u_max = 25e-6
            u_max = 35e-6    
        else:
            u_max = 80e-6
        inc0 = 8e-8; tresh0 = 0.6
        inc1 = 2e-8; tresh1 = 1
        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]
        listOption = ["damage"]*len(listTresh)

    elif "FCBA" in problem:
        # geom
        L = 4.5e-2
        h = 9e-2
        ep = 2e-2
        diam = 1e-2
        r = diam/2

        # material
        simpli2D = "CP" # ["CP","DP"]
        
        # phase field
        # gc = 3000
        gc = 0.07 * 1000
        nL = 100
        # nL = 180
        l0 = L/nL

        # loading
        u_max = 1e-3
        inc0 = 8e-6; tresh0 = 0.2
        inc1 = 2e-6; tresh1 = 0.6
        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]
        listOption = ["damage"]*2

    # ----------------------------------------------
    # meshSize
    # ----------------------------------------------
    if test:
        if optimMesh:            
            clD = l0*4
            clC = l0            
        elif updateMesh:
            clD = l0*4
            clC = l0
        else:
            if "Benchmark" in problem:
                clD = 0.25e-3
                clC = 0.12e-3
            else:
                clD = l0*2
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
    if dim == 2:
        if simpli2D == "CP":
            planeStress = True
        else:
            planeStress = False
    else:
        planeStress = False

    if materialType == "Elas_Isot":
        E = 12e9
        v = 0.3
        material = Materials.Elas_Isot(dim, E=E, v=v, planeStress=planeStress, thickness=ep)

    elif materialType == "Elas_IsotTrans":        
        El = 12e9
        Et = 500*1e6
        Gl = 450*1e6
        vl = 0.02
        vt = 0.44
        v = 0
        material = Materials.Elas_IsotTrans(dim, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, planeStress=planeStress, thickness=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

    # folder name
    folderName = "PlateWithHole_" + problem
    folder = Folder.PhaseField_Folder(folder=folderName, material=materialType, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solver=solver, test=test, closeCrack=False, nL=nL)
    
    if solve:

        # ----------------------------------------------
        # Meshing
        # ----------------------------------------------
        if optimMesh:
            refineZone = diam*1.5/2
            if split in ["Bourdin", "Amor"]:
                refineGeom = Domain(Point(y=h/2-refineZone, x=0), Point(y=h/2+refineZone, x=L), clC)
            else:
                refineGeom = Domain(Point(x=L/2-refineZone, y=0), Point(x=L/2+refineZone, y=h), clC)
        else:
            refineGeom = None

        def DoMesh(refineGeom=None) -> Mesh:

            point = Point()
            domain = Domain(point, Point(x=L, y=h), clD)
            circle = Circle(Point(x=L/2, y=h/2), diam, clD, isHollow=True)
            
            interfaceGmsh = Interface_Gmsh()

            if dim == 2:
                mesh = interfaceGmsh.Mesh_2D(domain, [circle], ElemType.TRI3, refineGeoms=[refineGeom])
            elif dim == 3:
                mesh = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,ep], 4, ElemType.HEXA8,refineGeoms=[refineGeom])

            return mesh
        
        mesh = DoMesh(refineGeom)
        
        if plotMesh:
            Display.Plot_Mesh(mesh)
            plt.show()        

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        pfm = Materials.PhaseField_Model(material, split, regu, gc, l0, solver=solver)

        simu = Simulations.Simu_PhaseField(mesh, pfm, verbosity=False)

        simu.Results_Set_Bc_Summary(u_max, listInc, listTresh, listOption)

        ud=0
        resol = 0
        nDetect = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True)

            arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)
            if "Benchmark" in problem:
                figLoad, axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]')
            elif "FCBA" in problem:
                figLoad, axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]')

        def Loading(ud: float):
            """Boundary conditions"""
            
            # Get Nodes
            nodes_Lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
            nodes_Upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)            
            nodes_X0Y0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
            nodes_Y0Z0 = mesh.Nodes_Conditions(lambda x,y,z: (y==0) & (z==0))

            simu.Bc_Init()
            simu.add_dirichlet(nodes_Lower, [0], ["y"])
            simu.add_dirichlet(nodes_X0Y0, [0], ["x"])
            simu.add_dirichlet(nodes_Upper, [-ud], ["y"])
            if dim == 3:
                simu.add_dirichlet(nodes_Y0Z0, [0], ["z"])                

        while ud <= u_max:

            resol += 1

            if dim == 3 and resol > 1500:
                break
            
            Loading(ud)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            # stop if the simulation does not converge
            if not convergence: break

            if updateMesh:

                meshSize_n = (clC-clD) * d + clD                

                # Display.Plot_Result(simu, meshSize_n)
                # nodes = np.where(d>=1e-3)[0]
                # test = [n for n in nodes if n not in damagedNodes]
                # if len(test) == 0: continue

                refineDomain = Interface_Gmsh().Create_posFile(simu.mesh.coordo, meshSize_n, folder)

                newMesh = DoMesh(refineDomain)

                if newMesh.Nn > simu.mesh.Nn:

                    # Display.Plot_Mesh(simu.mesh)
                    # Display.Plot_Mesh(newMesh)

                    proj = Calc_projector(simu.mesh, newMesh)

                    newU = np.zeros((newMesh.Nn, 2))

                    for i in range(dim):
                        newU[:,i] = proj @ u.reshape(-1,2)[:,i]

                    newD = proj @ d
                    
                    # Display.Plot_Result(simu.mesh, d, plotMesh=True)
                    # Display.Plot_Result(newMesh, newD, plotMesh=True)                    

                    # Display.Plot_Result(simu.mesh, u.reshape(-1,2)[:,0])
                    # Display.Plot_Result(newMesh, newU[:,0])

                    # plt.pause(1e-12)
                    # Tic.Plot_History()

                    simu.mesh = newMesh
                    mesh = newMesh
                    simu.set_u_n("displacement", newU.reshape(-1))
                    simu.set_u_n("damage", newD.reshape(-1))

                    # pass
                    # plt.close("all")

            simu.Save_Iter()

            nodes_Edges = simu.mesh.Nodes_Tags(["L0","L1","L2","L3"])
            nodes_Upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)
            dofsY_Upper = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_Upper, ["y"])

            f = np.sum(Kglob[dofsY_Upper, :] @ u)

            if "Benchmark" in problem:
                simu.Results_Set_Iteration_Summary(resol, ud*1e6, "µm", ud/u_max, True)
            elif "FCBA" in problem:
                simu.Results_Set_Iteration_Summary(resol, ud*1e3, "mm", 0, True)
           
            if d.max() < tresh0:
                ud += inc0
            else:
                ud += inc1

            # Detection if the edges has been touched
            if np.any(d[nodes_Edges] >= 0.98):
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

                if "Benchmark" in problem:
                    axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]', ax=axLoad)[1]
                elif "FCBA" in problem:
                    axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]', ax=axLoad)[1]

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

        if "Benchmark" in problem:
            Display.Plot_Load_Displacement(displacement*1e3, load*1e-6, 'ud [mm]', 'f [kN/mm]', folder)
        elif "FCBA" in problem:
            Display.Plot_Load_Displacement(displacement*1e3, load*1e-3, 'ud [mm]', 'f [kN]', folder)

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