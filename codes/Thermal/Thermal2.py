import matplotlib.pyplot as plt
import Display
import PostProcessing
import Folder
from Gmsh_Interface import Mesher, ElemType
from Geoms import Circle, Domain, Line, Point
import Materials
import Simulations
import numpy as np

if __name__ == '__main__':

    Display.Clear()
    
    folder = Folder.New_File(Folder.Join("Thermal",f"Revolve"), results=True)

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------

    plotIter = True; resultIter = "thermal"
    makeMovie = False; NMovie = 300

    R = 10
    e = 2
    h = 10

    a = 1
    domain = Domain(Point(R), Point(R+e, h), e / 4)
    axis = Line(Point(), Point(0,1,0))

    # Define simulation time parameters
    Tmax = 5  # Total simulation time
    N = 100  # Number of time steps
    dt = Tmax / N  # Time step size

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    # Generate the mesh based on the specified dimension
    angle = 360 * 3/4
    mesh = Mesher().Mesh_Revolve(domain, [], axis, angle, [angle*np.pi/180*R/domain.meshSize], elemType=ElemType.HEXA8, isOrganised=True)

    noeudsY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    noeudsYH = mesh.Nodes_Conditions(lambda x, y, z: y == h)

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    thermalModel = Materials.Thermal_Model(3, k=1, c=1)
    simu = Simulations.Simu_Thermal(mesh, thermalModel, False)

    # Set the density of the material
    simu.rho = 1

    def Iteration(steadyState: bool):
        """Function to perform one iteration of the simulation"""
        
        simu.Bc_Init()    

        # simu.add_dirichlet(noeudsY0, [40], [""])
        # simu.add_dirichlet(noeudsYH, [40], [""])
        simu.add_surfLoad(noeudsY0, [5], [""])
        simu.add_surfLoad(noeudsYH, [5], [""])
        
        thermal = simu.Solve()
        simu.Save_Iter()

        return thermal

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

    simu.set_u_n(simu.problemType, np.ones(mesh.Nn)*-10)

    # Check if it's a steady-state simulation or a time-dependent simulation
    if Tmax == 0:
        steadyState = True
        plotIter = False
    else:
        steadyState = False

    # If plotIter is True, create a figure for result visualization
    if plotIter:
        fig, ax, cb = Display.Plot_Result(simu, resultIter, nodeValues=True, plotMesh=True)

    print()
    t = 0  # init time
    # Main loop for time-dependent simulation
    while t < Tmax:
        # Perform one iteration of the simulation
        thermal = Iteration(False)

        # Increment time
        t += dt

        # If plotIter is True, update the result visualization
        if plotIter:
            cb.remove()
            fig, ax, cb = Display.Plot_Result(simu, resultIter, nodeValues=True, plotMesh=True, ax=ax)
            plt.pause(1e-12)

        # Print the current simulation time
        print(f"{np.round(t)} s", end='\r')

    # --------------------------------------------------------------------------------------------
    # PostProcessing
    # --------------------------------------------------------------------------------------------
    # Display the final thermal distribution
    Display.Plot_Result(simu, "thermal", plotMesh=True, nodeValues=True)

    # Save simulation results in Paraview format
    # PostProcessing.Make_Paraview(folder, simu)

    # Create a movie of the simulation if pltMovie is True
    if makeMovie:
        PostProcessing.Make_Movie(folder, resultIter, simu, NMovie, plotMesh=True)

    # Print the minimum temperature achieved in the simulation
    print(thermal.min())

    plt.show()