"""Transient thermal simulation."""

from EasyFEA import (Display, Folder, plt, np,
                     Mesher, ElemType, 
                     Materials, Simulations,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Circle, Domain, Point

if __name__ == '__main__':

    Display.Clear()

    dim = 3 # Set the simulation dimension (2D or 3D)
    
    folder = Folder.New_File(Folder.Join("Thermal",f"{dim}D"), results=True)

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    plotIter = True; resultIter = "thermal"
    makeMovie = False

    a = 1
    domain = Domain(Point(), Point(a, a), a / 20)
    circle = Circle(Point(a / 2, a / 2), diam=a / 3, isHollow=True, meshSize=a / 50)

    # Define simulation time parameters
    Tmax = 0.5  # Total simulation time
    N = 50  # Number of time steps
    dt = Tmax / N  # Time step size

    plotIter = False if N == 1 else plotIter

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Generate the mesh based on the specified dimension
    if dim == 2:
        mesh = Mesher().Mesh_2D(domain, [circle], ElemType.TRI3)
    else:
        mesh = Mesher().Mesh_Extrude(domain, [circle], [0, 0, -a], [4], ElemType.PRISM6)

    noeudsX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    noeudsXa = mesh.Nodes_Conditions(lambda x, y, z: x == a)
    nodesCircle = mesh.Nodes_Cylinder(circle, [0, 0, -1])

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    thermalModel = Materials.Thermal(dim=dim, k=1, c=1, thickness=1)
    simu = Simulations.ThermalSimu(mesh, thermalModel, False)

    # Set the density of the material
    simu.rho = 1

    # Apply Dirichlet boundary conditions to specific nodes
    simu.add_dirichlet(noeudsX0, [0], ["t"])
    simu.add_dirichlet(noeudsXa, [40], ["t"])

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

    # If plotIter is True, create a figure for result visualization
    if plotIter:
        ax = Display.Plot_Result(simu, resultIter, nodeValues=True, plotMesh=True)

    print()
    t = 0  # init time
    # Main loop for time-dependent simulation
    while t < Tmax:        
        
        simu.Solve()
        simu.Save_Iter()
        
        # Increment time
        t += dt

        # If plotIter is True, update the result visualization
        if plotIter:
            Display.Plot_Result(simu, resultIter, nodeValues=True, plotMesh=True, ax=ax)
            plt.pause(1e-12)

        # Print the current simulation time
        print(f"{t:.3f} s", end='\r')

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    # Display the final thermal distribution
    Display.Plot_Result(simu, "thermal", plotMesh=True, nodeValues=True, folder=folder)

    # Calculate and display the volume of the mesh if it's a 3D simulation
    if dim == 3:
        print(f"Volume: {mesh.volume:.3}")

    # Create a movie of the simulation if pltMovie is True
    if makeMovie:
        pvi.Movie_simu(simu, "thermal", folder, f"thermal{dim}D.mp4", show_edges=True)

    # Print the minimum temperature achieved in the simulation
    print(simu)

    plt.show()