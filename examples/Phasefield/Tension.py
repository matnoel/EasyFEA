"""Performs a damage simulation for a plate subjected to tension."""

from EasyFEA import (Display, Folder, plt, np, Tic,
                     Mesher, ElemType, Mesh,
                     Materials, Simulations,
                     Paraview_Interface)
from EasyFEA.Geoms import Point, Points, Domain, Line, Contour

import multiprocessing

# Display.Clear()

useParallel = False
nProcs = 4 # number of processes in parallel

# ----------------------------------------------
# Configurations
# ----------------------------------------------
dim = 2
test = True
solve = True

# Mesh
openCrack = True
optimMesh = True

# material
materialType = "Elas_Isot" #  "Elas_Isot", "Elas_Anisot"

# phasefield
maxIter = 1000
tolConv = 1e-0 # 1e-1, 1e-2, 1e-3
pfmSolver = Materials.PhaseField.SolverType.History

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes sans bourdin
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
splits = ["Miehe"]

regus = ["AT1"] # "AT1", "AT2"
# regus = ["AT1", "AT2"]

thetas = [-70, -80, -90] # [-0, -10, -20, -30, -45, -60]
theta = -0 # default value

# PostProcessing
plotMesh = False
plotResult = True
showResult = True
plotEnergy = False
saveParaview = False; Nparaview=400
makeMovie = False

# ----------------------------------------------
# Mesh
# ----------------------------------------------    
L = 1e-3;  #m
l0 = 8.5e-6 if materialType == "Elas_Anisot" else 1e-5
thickness = 1 if dim == 2 else 0.1/1000

def DoMesh(materialType: str= "Elas_Isot") -> Mesh:

    # meshSize
    clC = l0 if test else l0/2
    if optimMesh:
        # a coarser mesh can be used outside the refined zone
        clD = clC * 3
        # refines the mesh in the area where the crack will propagate
        gap = L*0.05
        if materialType == "Elas_Isot":                
            refineDomain = Domain(Point(L/2-gap, L/2-gap), Point(L, L/2+gap, thickness), clC)
        else:                
            refineDomain = Domain(Point(L/2-gap, L/2-gap), Point(L, L*0.8, thickness), clC)
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
        cracks = [Contour([l1, l2, l3, l4])]

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI3, cracks, [refineDomain])
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,thickness], [3], ElemType.TETRA4, cracks, [refineDomain])

    return mesh

# ----------------------------------------------
# Simu
# ----------------------------------------------

def DoSimu(split: str, regu: str):

    # Builds the path to the folder based on the problem data
    folderName = "Tension_Benchmark"
    if dim == 3:
        folderName += "_3D"
    folder = Folder.PhaseField_Folder(folderName, materialType, split, regu, 'DP',
                                      tolConv, pfmSolver, test, optimMesh, not openCrack,
                                      theta=theta)
            
    if solve:

        mesh = DoMesh(materialType)

        # Nodes recovery
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y == L)
        nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
        nodes_left = mesh.Nodes_Conditions(lambda x,y,z: (x == 0) & (y>0) & (y<L))
        nodes_right = mesh.Nodes_Conditions(lambda x,y,z: (x == L) & (y>0) & (y<L))
        nodes_crack = mesh.Nodes_Conditions(lambda x,y,z: (y==L/2) & (x<=L/2))
        if openCrack:
            nodes_detect = mesh.nodes.copy()
        else:
            nodes_detect = np.array(list(set(mesh.nodes)-set(nodes_crack)))

        # Builds edge nodes
        nodes_edges=[]
        for nodes in [nodes_lower,nodes_right,nodes_upper]:
            nodes_edges.extend(nodes)

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        if materialType == "Elas_Isot":
            material = Materials.Elas_Isot(dim, E=210e9, v=0.3,
                                           planeStress=False, thickness=thickness)
            Gc = 2.7e3 # J/m2
        elif materialType == "Elas_Anisot":
            if dim == 2:
                c11 = 65
                c22 = 260
                c33 = 30
                c12 = 20
                C_voigt = np.array([[c11, c12, 0],
                                    [c12, c22, 0],
                                    [0, 0, c33]])*1e9
                C_mandel = Materials.KelvinMandel_Matrix(dim, C_voigt)

                theta_rad = theta * np.pi/180
                axis1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0])
                axis2 = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0])

                material = Materials.Elas_Anisot(dim, C=C_voigt, useVoigtNotation=True,
                                                 axis1=axis1, axis2=axis2,
                                                 planeStress=False, thickness=thickness)
                Gc = 1e3 # J/m2
            else:
                raise Exception("Not implemented in 3D")

        pfm = Materials.PhaseField(material, split, regu, Gc=Gc, l0=l0, solver=pfmSolver)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        if materialType == "Elas_Isot":
            # load < treshold
            uinc0 = 1e-7 if test else 1e-8
            N0 = 40 if test else 400
            dep0 = uinc0*N0

            # load >= treshold
            uinc1 = 1e-8 if test else 1e-9
            N1 = 400 if test else 4000
            dep1 = dep0 + uinc1*N1

            treshold = uinc0*N0
            
            listInc = [uinc0, uinc1]
            listThreshold = [dep0, dep1]
            optionTreshold = ["displacement"]*2
        else:
            # load < treshold
            uinc0 = 12e-8 if test else 6e-8
            # load >= treshold
            uinc1 = 4e-8 if test else 2e-8

            treshold = 0.6

            listInc = [uinc0, uinc1]
            listThreshold = [0, treshold]
            optionTreshold = ["damage"]*2

        def Loading(dep):
            """Boundary conditions"""

            simu.Bc_Init()            

            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")
            
            if dim == 2:
                simu.add_dirichlet(nodes_upper, [0,dep], ["x","y"])
            elif dim == 3:
                simu.add_dirichlet(nodes_upper, [0,dep,0], ["x","y","z"])
            simu.add_dirichlet(nodes_lower, [0],["y"])

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.PhaseFieldSimu(mesh, pfm, verbosity=False)
        simu.Results_Set_Bc_Summary(0.0,listInc, listThreshold, optionTreshold)

        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        tic = Tic()
        
        # INIT
        nDetect = 0
        displacement=[]
        force=[]

        dep = -uinc0
        iter = -1
        while True: # simu until break
            
            iter += 1
            if materialType == 'Elas_Isot':
                dep += uinc0 if dep < treshold else uinc1
            else:
                if np.max( simu.damage[nodes_detect]) < treshold:
                    dep += uinc0
                else:
                    dep += uinc1
            
            # apply new boundary conditions
            Loading(dep)

            # solve and save iter
            u, d, Kglob, converg = simu.Solve(tolConv, maxIter, convOption=1)
            simu.Save_Iter()

            # print iter solution
            simu.Results_Set_Iteration_Summary(iter, dep*1e6, "µm", 0, True)

            # If the solver has not converged, stop the simulation.
            if not converg: break            
            
            # resulting force on upper edge
            f = np.sum(Kglob[dofsY_upper, :] @ u)

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
        # ----------------------------------------------
        # Loading
        # ---------------------------------------------
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
            
    if saveParaview:
        Paraview_Interface.Make_Paraview(simu, folder, Nparaview)

    if makeMovie:
        Display.Movie_Simu(folder, "damage", simu, NiterFin=0)

    if plotMesh:
        Display.Plot_Mesh(simu.mesh)
            
    if plotEnergy:
        Display.Plot_Energy(simu, N=400, folder=folder)

    Tic.Resume()

    if solve:
        Tic.Plot_History(folder, False)

    if showResult:
        plt.show()

    Tic.Clear()
    plt.close('all')

if __name__ == '__main__':

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