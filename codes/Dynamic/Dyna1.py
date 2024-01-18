import os
import matplotlib.pyplot as plt

import Folder
import PostProcessing
import Display
from Geom import *
import Materials
from Interface_Gmsh import Interface_Gmsh, ElemType
import Simulations
from TicTac import Tic

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    dim = 2
    folder = Folder.New_File(f"Dynamics{dim}D", results=True)
    plotResult = True
    
    initSimu = True
    depInit = -7
    load = -800 # N

    makeParaview = False; NParaview = 500
    makeMovie = False; NMovie = 400
    plotIter = True; resultToPlot = "uy"

    # Dumping
    # coefM = 1e-2
    # coefK = 1e-2*2
    coefM = 1e-3
    coefK = 1e-3

    tic_Tot = Tic()

    # geom
    L = 120;  #mm
    h = 13
    b = 13

    # --------------------------------------------------------------------------------------------
    # Meshing
    # --------------------------------------------------------------------------------------------
    # meshSize = h/1
    # meshSize = L/2
    meshSize = h/5

    interfaceGmsh = Interface_Gmsh(False)
    if dim == 2:
        elemType = ElemType.QUAD4 # TRI3, TRI6, TRI10, TRI15, QUAD4, QUAD8
        domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), meshSize)
        Line0 = Line(Point(y=-h/2), Point(y=h/2))
        LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
        LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
        circle = Circle(Point(x=L/2, y=0), h*0.2, isHollow=False)

        mesh = interfaceGmsh.Mesh_2D(domain, elemType=elemType, isOrganised=True)
        area = mesh.area - L*h
    elif dim == 3:
        elemType = ElemType.HEXA8 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15
        domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), meshSize=meshSize)
        mesh = interfaceGmsh.Mesh_3D(domain, [], [0,0,b], elemType=elemType, layers=[3], isOrganised=True)

        volume = mesh.volume - L*b*h
        area = mesh.area - (L*h*4 + 2*b*h)

    Display.Plot_Mesh(mesh)

    nodes_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    nodes_L = mesh.Nodes_Conditions(lambda x,y,z: x == L)
    nodes_h = mesh.Nodes_Conditions(lambda x,y,z: y == h/2)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------

    material = Materials.Elas_Isot(dim, thickness=b)
    simu = Simulations.Simu_Displacement(mesh, material, useIterativeSolvers=initSimu)    
    simu.rho = 8100*1e-9

    if initSimu:
        simu.Bc_Init()
        simu.add_dirichlet(nodes_0, [0]*dim, simu.Get_directions(), description="Fixed")
        simu.add_dirichlet(nodes_L, [depInit], ["y"], description="dep")
        simu.Solve()
        simu.Save_Iter()

    factorDef = 1
    if plotIter:
        fig, ax, cb = Display.Plot_Result(simu, resultToPlot, nodeValues=True, plotMesh=True, deformFactor=factorDef)

    Tmax = 0.5
    N = 100
    dt = Tmax/N
    time = -dt

    simu.Solver_Set_Newton_Raphson_Algorithm(dt)
    simu.Set_Rayleigh_Damping_Coefs(coefM=coefM, coefK=coefK)

    while time <= Tmax:
        
        time += dt

        simu.Bc_Init()
        simu.add_dirichlet(nodes_0, [0]*dim, simu.Get_directions(), description="Fixed")
        if not initSimu:            
            simu.add_surfLoad(nodes_L, [load*time/Tmax/(h*b)], ['y'])
        simu.Solve()
        simu.Save_Iter()

        if plotIter:
            cb.remove()
            fig, ax, cb = Display.Plot_Result(simu, resultToPlot, nodeValues=True, plotMesh=True, ax=ax, deformFactor=factorDef)
            plt.pause(1e-12)


        print(f"{time:.3f} s", end='\r')

    tic_Tot.Tac("Time","total time", True)        

    # --------------------------------------------------------------------------------------------
    # Post processing
    # --------------------------------------------------------------------------------------------
    Display.Section("Post processing")

    Display.Plot_BoundaryConditions(simu)

    # folder=""

    if makeParaview:
        PostProcessing.Make_Paraview(folder, simu,Niter=NParaview)

    if makeMovie:
        PostProcessing.Make_Movie(folder, resultToPlot, simu, plotMesh=True, Niter=NMovie, deformation=True, nodeValues=True, factorDef=1)

    if plotResult:

        tic = Tic()
        print(simu)
        # Display.Plot_Result(simu, "amplitude")
        # Display.Plot_Mesh(simu, deformation=True, folder=folder)
        Display.Plot_Result(simu, "uy", deformFactor=factorDef, nodeValues=False)        
        Display.Plot_Result(simu, "Svm", plotMesh=False, nodeValues=False)
        # Display.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)
        
        tic.Tac("Display","Results", plotResult)

    Tic.Plot_History(details=False)
    plt.show()