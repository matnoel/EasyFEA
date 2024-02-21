import matplotlib.pyplot as plt
import Display
import PostProcessing
import Folder
from Gmsh_Interface import Mesher, ElemType
from Geoms import Circle, Domain, Point
import Materials
import Simulations
import numpy as np

if __name__ == '__main__':

    Display.Clear()

    dim = 3 # Set the simulation dimension (2D or 3D)
    
    folder = Folder.New_File(Folder.Join("Thermal",f"{dim}D"), results=True)

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------

    plotIter = True; resultIter = "thermal"
    makeMovie = False; NMovie = 300

    a = 1
    domain = Domain(Point(), Point(a, a), a / 20)
    circle = Circle(Point(a / 2, a / 2), diam=a / 3, isHollow=True, meshSize=a / 50)

    # Define simulation time parameters
    Tmax = 0.5  # Total simulation time
    N = 50  # Number of time steps
    dt = Tmax / N  # Time step size

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    # Generate the mesh based on the specified dimension
    if dim == 2:
        mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI3)
    else:
        mesh = Mesher().Mesh_Extrude(domain, [circle], [0, 0, -a], [4], ElemType.PRISM6)

    noeudsX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    noeudsXa = mesh.Nodes_Conditions(lambda x, y, z: x == a)
    nodesCircle = mesh.Nodes_Cylinder(circle, [0, 0, -1])

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, thickness=1)
    simu = Simulations.Simu_Thermal(mesh, thermalModel, False)

    # Set the density of the material
    simu.rho = 1

    def Iteration(steadyState: bool):
        """Function to perform one iteration of the simulation"""
        # Initialize the boundary conditions for the current iteration
        simu.Bc_Init()

        # Apply Dirichlet boundary conditions to specific nodes
        simu.add_dirichlet(noeudsX0, [0], [""])
        simu.add_dirichlet(noeudsXa, [40], [""])

        # Uncomment and modify the following lines to apply additional boundary conditions
        # simu.add_dirichlet(nodesCircle, [10], [""])
        # simu.add_dirichlet(nodesCircle, [10], [""])
        # simu.add_surfLoad(nodesCircle, [1], [""])

        # Solve the thermal simulation for the current iteration
        thermal = simu.Solve()

        # Save the results of the current iteration
        simu.Save_Iter()

        return thermal

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

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
    Display.Plot_Result(simu, "thermal", plotMesh=True, nodeValues=True, folder=folder)

    # Calculate and display the volume of the mesh if it's a 3D simulation
    if dim == 3:
        print(f"Volume: {mesh.volume:.3}")

    # Save simulation results in Paraview format
    # PostProcessing.Make_Paraview(folder, simu)

    # Create a movie of the simulation if pltMovie is True
    if makeMovie:
        PostProcessing.Make_Movie(folder, resultIter, simu, NMovie, plotMesh=True)

    # Print the minimum temperature achieved in the simulation
    print(thermal.min())

    plt.show()