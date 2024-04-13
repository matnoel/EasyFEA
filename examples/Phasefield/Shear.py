"""Performs a damage simulation for a plate subjected to shear."""

from EasyFEA import (Display, Folder, plt, np, Tic,
                     Mesher, ElemType, Mesh,
                     Materials, Simulations,
                     Paraview_Interface,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Points, Domain, Line, Contour

import multiprocessing

# Display.Clear()

useParallel = False
nProcs = 4 # number of processes in parallel

# ----------------------------------------------
# Configuration
# ----------------------------------------------
dim = 2
test = True
solve = True

# Mesh
openCrack = True
optimMesh = False

# phasefield
maxIter = 1000
tolConv = 1e-0 # 1e-1, 1e-2, 1e-3
pfmSolver = Materials.PhaseField.SolverType.History

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes sans bourdin
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
splits = ["Amor","Miehe"]

# regus = ["AT1", "AT2"]
regus = ["AT1"] # "AT1", "AT2" 

# PostProcessing
plotResult = False
showResult = False
plotMesh = True
plotEnergy = False
saveParaview = False; Nparaview=400
makeMovie = False

# ----------------------------------------------
# Mesh
# ----------------------------------------------    
L = 1e-3;  #m
l0 = 1e-5
thickness = 1 if dim == 2 else 0.1/1000

def DoMesh(split: str) -> Mesh:
    # meshSize
    clC = l0 if test else l0/2
    if optimMesh:
        # a coarser mesh can be used outside the refined zone
        clD = clC * 3
        # refines the mesh in the area where the crack will propagate
        gap = L*0.05
        h = L if split == "Bourdin" else L/2+gap
        refineDomain = Domain(Point(L/2-gap, 0), Point(L, h, thickness), clC)
    else:
        clD = clC
        refineDomain = None

    # geom
    pt1 = Point()
    pt2 = Point(L)
    pt3 = Point(L,L)
    pt4 = Point(0,L)
    contour = Points([pt1, pt2, pt3, pt4], clD)

    if dim == 2:
        ptC1 = Point(0,L/2, isOpen=openCrack)
        ptC2 = Point(L/2,L/2)
        cracks = [Line(ptC1, ptC2, clC, isOpen=openCrack)]
    if dim == 3:
        meshSize = clD if optimMesh else clC
        ptC1 = Point(0,L/2,0, isOpen=openCrack)
        ptC2 = Point(L/2,L/2, 0)
        ptC3 = Point(L/2,L/2, thickness)
        ptC4 = Point(0,L/2, thickness, isOpen=openCrack)
        l1 = Line(ptC1, ptC2, meshSize, openCrack)
        l2 = Line(ptC2, ptC3, meshSize, False)
        l3 = Line(ptC3, ptC4, meshSize, openCrack)
        l4 = Line(ptC4, ptC1, meshSize, openCrack)            
        cracks = [Contour([l1, l2, l3, l4], isOpen=openCrack)]
    
    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3, cracks, [refineDomain])
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,thickness], [], ElemType.TETRA4, cracks, [refineDomain])

    return mesh

# ----------------------------------------------
# Simu
# ----------------------------------------------
def DoSimu(split: str, regu: str):

    # Builds the path to the folder based on the problem data
    folderName = "Shear_Benchmark"
    if dim == 3:
        folderName += "_3D"
    folder = Folder.PhaseField_Folder(folderName, "Elas_Isot", split, regu, 'DP',
                                    tolConv, pfmSolver, test, optimMesh, not openCrack)

    if solve:
        
        mesh = DoMesh(split)        

        # Nodes recovery
        nodes_crack = mesh.Nodes_Conditions(lambda x,y,z: (y==L/2) & (x<=L/2))
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y == L)
        nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
        nodes_left = mesh.Nodes_Conditions(lambda x,y,z: (x == 0) & (y>0) & (y<L))
        nodes_right = mesh.Nodes_Conditions(lambda x,y,z: (x == L) & (y>0) & (y<L))

        # Builds edge nodes
        nodes_edges=[]
        for nodes in [nodes_lower,nodes_right,nodes_upper]:
            nodes_edges.extend(nodes)

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        material = Materials.Elas_Isot(dim, E=210e9, v=0.3,
                                        planeStress=False, thickness=thickness)
        Gc = 2.7e3 # J/m2
        pfm = Materials.PhaseField(material, split, regu, Gc, l0, pfmSolver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        u_inc = 5e-8 if test else 1e-8
        N = 400 if test else 2000

        loadings = np.linspace(u_inc, u_inc*N, N, endpoint=True)
        
        listInc = [u_inc]
        listThreshold = [loadings[-1]]
        optionTreshold = ["displacement"]

        def Loading(dep):
            """Boundary conditions"""

            simu.Bc_Init()

            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")            
            
            # Left and right travel conditions
            simu.add_dirichlet(nodes_left, [0],["y"])
            simu.add_dirichlet(nodes_right, [0],["y"])
            simu.add_dirichlet(nodes_upper, [dep,0], ["x","y"])
            simu.add_dirichlet(nodes_lower, [0]*dim, simu.Get_dofs())

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)
        simu.Results_Set_Bc_Summary(loadings[-1],listInc, listThreshold, optionTreshold)

        dofsX_upper = simu.Bc_dofs_nodes(nodes_upper, ["x"])

        tic = Tic()
        
        # INIT
        N = len(loadings)
        nDetect = 0
        displacement=[]
        force=[]        
        for iter, dep in enumerate(loadings):
            
            # apply new boundary conditions
            Loading(dep)

            # solve and save iter
            u, d, Kglob, converg = simu.Solve(tolConv, maxIter, convOption=1)
            simu.Save_Iter()

            # print iter solution
            simu.Results_Set_Iteration_Summary(iter, dep*1e6, "µm", iter/N, True)

            # If the solver has not converged, stop the simulation.
            if not converg: break            
            
            # resulting force on upper edge
            f = np.sum(Kglob[dofsX_upper, :] @ u)

            displacement.append(dep)
            force.append(f)

            # check for damaged edges
            if np.any(simu.damage[nodes_edges] >= 0.98):
                nDetect +=1
                if nDetect == 10:
                    # If the edge has been touched 10 times, stop the simulation
                    break

        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        print()
        Simulations.Save_Force_Displacement(force, displacement, folder)
        simu.Save(folder)        

        force = np.asarray(force)
        displacement = np.asarray(displacement)
    else:
        simu: Simulations.PhaseFieldSimu = Simulations.Load_Simu(folder)
        force, displacement = Simulations.Load_Force_Displacement(folder)

    # ----------------------------------------------
    # PostProcessing
    # ---------------------------------------------
    if plotResult:
        Display.Plot_Iter_Summary(simu, folder, None, None)
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Force_Displacement(force*1e-6, displacement*1e6, 'ud [µm]', 'f [kN/mm]', folder)
        Display.Plot_Result(simu, "damage", nodeValues=True, plotMesh=False, folder=folder, filename="damage")

    if plotMesh:
        # pvi.Plot_Mesh(simu.mesh).show()
        # pvi.Plot(simu, "ux", show_edges=True).show()
        Display.Plot_Mesh(simu.mesh)
            
    if saveParaview:
        Paraview_Interface.Make_Paraview(simu, folder, Nparaview)

    if makeMovie:
        # pvi.Plot_Mesh(simu.mesh).show()
        simu.Set_Iter(-1)
        nodes_upper = simu.mesh.Nodes_Conditions(lambda x,y,z: y==L)
        depMax = simu.Result('displacement_norm')[nodes_upper].max()
        deformFactor = L*.1/depMax
        
        def Func(plotter, n):

            simu.Set_Iter(n)

            grid = pvi._pvGrid(simu, 'damage', deformFactor)

            tresh = grid.threshold((0,0.8))
        
            pvi.Plot(tresh, 'damage', deformFactor, show_edges=True, plotter=plotter, clim=(0,1))
        
        pvi.Movie_func(Func, len(simu.results), folder, 'damage.mp4')
        
            
    if plotEnergy:
        Display.Plot_Energy(simu, N=400, folder=folder)

    Tic.Resume()

    if solve:
        Tic.Plot_History(folder, False)

    if showResult:
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