# Import necessary modules
import matplotlib.pyplot as plt
import Display
import PostProcessing
import Folder
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Circle, Domain, Line, Point
import Materials
import Simulations
import numpy as np

# Clear the existing display
Display.Clear()

# Set plotIter and affichageIter for result visualization
plotIter = True
resultIter = "thermal"

# Set pltMovie and NMovie for creating a movie of simulation results
makeMovie = False
NMovie = 300

# Set the simulation dimension (2D or 3D)
dim = 3

# Create a folder for storing simulation results
folder = Folder.New_File(filename=f"Thermal{dim}D", results=True)

# Define the domain
a = 1
if dim == 2:
    domain = Domain(Point(), Point(a, a), a / 20)
else:
    domain = Domain(Point(), Point(a, a), a / 20)

# Create a circular region inside the domain
circle = Circle(Point(a / 2, a / 2), diam=a / 3, isHollow=True, meshSize=a / 50)

# Create an interface to Gmsh for meshing
interfaceGmsh = Interface_Gmsh(False, False, True)

# Generate the mesh based on the specified dimension
if dim == 2:
    mesh = interfaceGmsh.Mesh_2D(domain, [circle], ElemType.QUAD4)
else:
    mesh = interfaceGmsh.Mesh_3D(domain, [circle], [0, 0, -a], 4, ElemType.PRISM6)

# Create a thermal material model
thermalModel = Materials.Thermal_Model(dim=dim, k=1, c=1, thickness=1)

# Initialize the thermal simulation with the mesh and material model
simu = Simulations.Simu_Thermal(mesh, thermalModel, False)

# Set the density of the material
simu.rho = 1

# Define boundary conditions
noeuds0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
noeudsL = mesh.Nodes_Conditions(lambda x, y, z: x == a)

if dim == 2:
    noeudsCircle = mesh.Nodes_Circle(circle)
else:
    noeudsCircle = mesh.Nodes_Cylinder(circle, [0, 0, a])

# Function to perform one iteration of the simulation
def Iteration(steadyState: bool):
    # Initialize the boundary conditions for the current iteration
    simu.Bc_Init()

    # Apply Dirichlet boundary conditions to specific nodes
    simu.add_dirichlet(noeuds0, [0], [""])
    simu.add_dirichlet(noeudsL, [40], [""])

    # Uncomment and modify the following lines to apply additional boundary conditions
    # simu.add_dirichlet(noeudsCircle, [10], [""])
    # simu.add_dirichlet(noeudsCircle, [10], [""])
    # simu.add_volumeLoad(noeudsCircle, [100], [""])

    # Solve the thermal simulation for the current iteration
    thermal = simu.Solve()

    # Save the results of the current iteration
    simu.Save_Iter()

    return thermal

# Define simulation time parameters
Tmax = 0.5  # Total simulation time
N = 50  # Number of time steps
dt = Tmax / N  # Time step size
t = 0  # Current time

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

# Uncomment the following lines to display specific mesh elements or nodes
# Display.Plot_NoeudsMaillage(mesh, noeuds=noeudsCircle)
# Display.Plot_ElementsMaillage(mesh, noeuds=noeudsCircle, dimElem=3)

# Display the final thermal distribution
Display.Plot_Result(simu, "thermal", plotMesh=True, nodeValues=True)

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

# Display the plots
plt.show()