# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Dyna2
=====

Wave propagation.
"""
# TODO: Compare results with analytical values.

from EasyFEA import Folder, Display, Models, Tic, plt, ElemType, Simulations, PyVista
from EasyFEA.Geoms import Domain, Circle, Line

if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    # outputs
    folder = Folder.Join(Folder.RESULTS_DIR, "Dynamics", "Dyna2")
    plotModel = False
    plotIter = False
    makeMovie = True
    result = "speed_norm"

    # Define geometric parameters
    a = 1
    meshSize = a / 50
    diam = a / 10
    r = diam / 2

    # Time parameters
    tMax = 1e-6
    Nt = 20
    dt = tMax / Nt

    # Load parameters
    load = 1e-3
    f0 = 2
    a0 = 1
    t0 = dt * 4

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Define the domain and create the mesh
    domain = Domain((-a / 2, -a / 2), (a / 2, a / 2), meshSize)
    circle = Circle((0, 0), diam, meshSize, isHollow=False)
    line = Line((0, 0), (diam / 4, 0))
    mesh = domain.Mesh_2D([circle], ElemType.TRI6, cracks=[line])

    # Plot the model if specified
    if plotModel:
        Display.Plot_Tags(mesh)
        plt.show()

    # Get nodes for boundary conditions and loading
    nodesBorders = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    nodesLoad = mesh.Nodes_Point((0, 0))

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    # Define material properties
    material = Models.ElasIsot(2, E=210000e6, v=0.3, planeStress=False, thickness=1)
    lmbda = material.get_lambda()
    mu = material.get_mu()

    # Create the simulation object
    simu = Simulations.ElasticSimu(mesh, material)
    simu.Set_Rayleigh_Damping_Coefs(1e-10, 1e-10)
    simu.Solver_Set_Hyperbolic_Algorithm(beta=1 / 4, gamma=1 / 2, dt=dt)

    t = 0

    def Loading():
        # Set displacement boundary conditions
        simu.add_dirichlet(nodesBorders, [0, 0], ["x", "y"], description="[0,0]")

        # Set Neumann boundary conditions (loading) at t = t0
        if t == t0:
            simu.add_neumann(nodesLoad, [load], ["x"])

    # Plot the result at the initial iteration if specified
    if plotIter:
        ax = Display.Plot_Result(simu, result, nodeValues=True, title=result)

    # Create a timer object
    tic = Tic()

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
            ax = Display.Plot_Result(simu, result, nodeValues=True, ax=ax, title=result)
            plt.pause(1e-12)

        t += dt

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    if makeMovie:
        PyVista.Movie_simu(
            simu,
            f"{result}",
            folder,
            f"{result}.gif",
        )

    plt.show()
