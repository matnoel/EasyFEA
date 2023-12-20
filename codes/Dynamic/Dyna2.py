import matplotlib.pyplot as plt
import numpy as np
import Simulations
import Materials
from Geom import Domain, Point, Circle, Line
from Interface_Gmsh import Interface_Gmsh, ElemType
import Display
import PostProcessing
import Folder
import TicTac

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    folder = Folder.New_File("Waves", results=True) # Create a folder for the simulation results

    # Define geometric parameters
    a = 1
    meshSize = a / 100
    diam = a / 10
    r = diam / 2

    # Time parameters
    tMax = 1e-5 
    Nt = 150
    dt = tMax / Nt

    # Load parameters
    load = 1e-3
    f0 = 2
    a0 = 1
    t0 = dt * 4

    plotModel = False # Define whether to plot the model  
    plotIter = True # Define whether to plot the results at each iteration

    # Specify the result to plot (amplitudeSpeed in this case)
    resultat = "amplitudeSpeed"

    # Define whether to create a movie
    makeMovie = False

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------

    # Define the domain and create the mesh
    domain = Domain(Point(x=-a / 2, y=-a / 2), Point(x=a / 2, y=a / 2), meshSize)
    circle = Circle(Point(), diam, meshSize, isHollow=False)
    line = Line(Point(), Point(diam / 4))
    interfaceGmsh = Interface_Gmsh(False)
    mesh = interfaceGmsh.Mesh_2D(domain, [circle], ElemType.TRI3, cracks=[line])

    # Plot the model if specified
    if plotModel:
        Display.Plot_Model(mesh)
        plt.show()

    # Get nodes for boundary conditions and loading
    nodesBorders = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    # nodesLoad = mesh.Nodes_Line(line)
    nodesLoad = mesh.Nodes_Point(Point())

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------

    # Define material properties
    material = Materials.Elas_Isot(2, E=210000e6, v=0.3, planeStress=False, thickness=1)
    l = material.get_lambda()
    mu = material.get_mu()

    # Create the simulation object
    simu = Simulations.Simu_Displacement(mesh, material)
    simu.Set_Rayleigh_Damping_Coefs(1e-10, 1e-10)
    simu.Solver_Set_Newton_Raphson_Algorithm(betha=1 / 4, gamma=1 / 2, dt=dt)

    t = 0

    def Loading():
        # Set displacement boundary conditions
        simu.add_dirichlet(nodesBorders, [0, 0], ["x", "y"], description="[0,0]")

        # Set Neumann boundary conditions (loading) at t = t0
        if t == t0:
            # simu.add_neumann(nodesLoad, [load, -load], ["x", "y"], description="[load,load]")
            simu.add_neumann(nodesLoad, [load], ["x"])
            # simu.add_lineLoad(nodesLoad, [load/line.length], ['x'])

    # Plot the result at the initial iteration if specified
    if plotIter:
        fig, ax, cb = Display.Plot_Result(simu, resultat, nodeValues=True)

    # Create a timer object
    tic = TicTac.Tic()

    # Time loop
    while t <= tMax:
        # Apply loading conditions
        Loading()

        # Solve the simulation
        simu.Solve()

        # Save the iteration results
        simu.Save_Iter()

        # Print the progress
        print(f"{t / tMax * 100:.3f}  %", end="\r")

        # Update the plot at each iteration if specified
        if plotIter:
            cb.remove()
            fig, ax, cb = Display.Plot_Result(simu, resultat, nodeValues=True, ax=ax)
            plt.pause(1e-12)

        t += dt

    # --------------------------------------------------------------------------------------------
    # PostProcessing
    # --------------------------------------------------------------------------------------------
    # Make a movie if specified
    if makeMovie:
        PostProcessing.Make_Movie(folder, resultat, simu)

    plt.show()