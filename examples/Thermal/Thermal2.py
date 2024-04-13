"""Transient thermal simulation."""

from EasyFEA import (Display, Folder, plt, np,
                     Mesher, ElemType, 
                     Materials, Simulations,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Line, Domain, Point

if __name__ == '__main__':

    Display.Clear()
    
    folder = Folder.New_File(Folder.Join("Thermal",f"Revolve"), results=True)

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    plotIter = True; resultIter = "thermal"
    makeMovie = False

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

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Generate the mesh based on the specified dimension
    angle = 360 * 3/4
    mesh = Mesher().Mesh_Revolve(domain, [], axis, angle, [angle*np.pi/180*R/domain.meshSize], elemType=ElemType.HEXA8, isOrganised=True)

    noeudsY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    noeudsYH = mesh.Nodes_Conditions(lambda x, y, z: y == h)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    thermalModel = Materials.Thermal(3, k=1, c=1)
    simu = Simulations.ThermalSimu(mesh, thermalModel, False)

    # Set the density of the material
    simu.rho = 1

    simu.add_surfLoad(noeudsY0, [5], ["t"])
    simu.add_surfLoad(noeudsYH, [5], ["t"])

    # Set the parabolic algorithm for the solver
    simu.Solver_Set_Parabolic_Algorithm(alpha=0.5, dt=dt)

    simu._Set_u_n(simu.problemType, np.ones(mesh.Nn)*-10)

    # If plotIter is True, create a figure for result visualization
    if plotIter:
        ax = Display.Plot_Result(simu, resultIter, nodeValues=True, plotMesh=True)

    print()
    t = 0  # init time
    # Main loop for time-dependent simulation
    while t < Tmax:
        # Perform one iteration of the simulation
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

    # Create a movie of the simulation if pltMovie is True
    if makeMovie:
        pvi.Movie_simu(simu, "thermal", folder, f"thermal.mp4", show_edges=True)

    # Print the minimum temperature achieved in the simulation
    print(simu)

    plt.show()