from Geom import Domain, Point
from Mesh import Mesh
import Materials
from Interface_Gmsh import Interface_Gmsh, GroupElem
import Simulations
import Display as Display
from TicTac import Tic
import Folder
import PostProcessing
import numpy as np
import matplotlib.pyplot as plt

# Clear the display to start fresh
Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
dim = 2 # Define the dimension of the problem (2D or 3D)

# List of mesh sizes (number of elements) to investigate convergence
if dim == 2:
    listNbElement = np.arange(1, 10, 1)
else:
    listNbElement = np.arange(1, 4, 2)

# Create a folder to store the simulation results
folder = Folder.New_File(f"Convergence {dim}D", results=True)

# Define whether to plot the results
plotResult = True
makeParaview = False

# Define geometry parameters
L = 120  # mm
h = 13   # Height
b = 13   # Width
P = 800  # N

# Material properties
E = 210000  # MPa (Young's modulus)
v = 0.25    # Poisson's ratio

# Define the material behavior (elasticity with plane stress assumption)
material = Materials.Elas_Isot(dim, thickness=b, E=E, v=v, planeStress=True)

# Compute the theoretical deformation energy (reference value)
WdefRef = 2 * P**2 * L / E / h / b * (L**2 / h / b + (1 + v) * 3 / 5)

# Lists to store data for plotting
listTimes_e_nb = []  # List to store computation times for each element type and mesh size
listWdef_e_nb = []   # List to store deformation energies for each element type and mesh size
listDofs_e_nb = []    # List to store degrees of freedom for each element type and mesh size

# ----------------------------------------------
# Simulation
# ----------------------------------------------
# Timer
tic = Tic()

# Loop over each element type for both 2D and 3D simulations
elemTypes = GroupElem.get_Types2D() if dim == 2 else GroupElem.get_Types3D()
interfaceGmsh = Interface_Gmsh()

for e, elemType in enumerate(elemTypes):
    listTimes_nb = []
    listWdef_nb = []
    listDofs_nb = []

    # Loop over each mesh size (number of elements)
    for nbElem in listNbElement:
        meshSize = b / nbElem

        # Define the domain for the mesh
        domain = Domain(Point(), Point(x=L, y=h), meshSize=meshSize)

        # Generate the mesh using Gmsh
        if dim == 2:
            mesh = interfaceGmsh.Mesh_2D(domain, [], elemType, isOrganised=True)
        else:
            mesh = interfaceGmsh.Mesh_3D(domain, [], elemType=elemType, extrude=[0, 0, b], nLayers=4)        

        # Calculate the volume of the mesh for verification
        if mesh.dim == 3:
            volume = mesh.volume
        else:
            volume = mesh.area * material.thickness

        # Ensure that the volume matches the expected value (L * h * b)
        assert np.abs(volume - (L * h * b)) / volume <= 1e-10

        # Define nodes on the left boundary (x=0) and right boundary (x=L)
        nodes_x0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_xL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

        # Create or update the simulation object with the current mesh        
        if e == 0 and nbElem == listNbElement[0]:
            simu = Simulations.Simu_Displacement(mesh, material, verbosity=False, useNumba=False)
        else:
            simu.Bc_Init()
            simu.mesh = mesh

        # Set displacement boundary conditions
        if dim == 2:
            simu.add_dirichlet(nodes_x0, [0, 0], ["x", "y"])
        else:
            simu.add_dirichlet(nodes_x0, [0, 0, 0], ["x", "y", "z"])

        # Set surface load on the right boundary (y-direction)
        simu.add_surfLoad(nodes_xL, [-P / h / b], ["y"])

        # Solve the simulation
        simu.Solve()
        simu.Save_Iter()

        # Get the computed deformation energy
        Wdef = simu.Get_Result("Wdef")

        # Store the results for the current mesh size
        listTimes_nb.append(tic.Tac("Resolutions", "Temps total", False))
        listWdef_nb.append(Wdef)
        listDofs_nb.append(mesh.Nn * dim)

        if elemType != mesh.elemType:
            print("Error in mesh generation")

        print(f"Elem: {mesh.elemType}, nby: {nbElem:2}, Wdef = {np.round(Wdef, 3)}, "
              f"error = {np.abs(WdefRef - Wdef) / WdefRef:.2e}")

    # Store the results for the current element type
    listTimes_e_nb.append(listTimes_nb)
    listWdef_e_nb.append(listWdef_nb)
    listDofs_e_nb.append(listDofs_nb)

# ----------------------------------------------
# PostProcessing
# ----------------------------------------------
# Display the convergence of deformation energy
fig_Wdef, ax_Wdef = plt.subplots()
fig_Error, ax_Times_Error = plt.subplots()
fig_Times, ax_Times = plt.subplots()

WdefRefArray = np.ones_like(listDofs_e_nb[0]) * WdefRef
WdefRefArray5 = WdefRefArray * 0.95

print(f"\nWSA = {np.round(WdefRef, 4)} mJ")

for e, elemType in enumerate(elemTypes):
    # Convergence of deformation energy
    ax_Wdef.plot(listDofs_e_nb[e], listWdef_e_nb[e])

    # Error in deformation energy
    Wdef = np.array(listWdef_e_nb[e])
    erreur = (WdefRef - Wdef) / WdefRef * 100
    ax_Times_Error.loglog(listDofs_e_nb[e], erreur)

    # Computation time
    ax_Times.loglog(listDofs_e_nb[e], listTimes_e_nb[e])

# Deformation energy
ax_Wdef.grid()
ax_Wdef.set_xlim([-10, 8000])
ax_Wdef.set_xlabel('Degrees of Freedom (DDL)')
ax_Wdef.set_ylabel('Deformation Energy (Wdef) [mJ]')
ax_Wdef.legend(elemTypes)
ax_Wdef.fill_between(listDofs_nb, WdefRefArray, WdefRefArray5, alpha=0.5, color='red')

# Error in deformation energy
ax_Times_Error.grid()
ax_Times_Error.set_xlabel('Degrees of Freedom (DDL)')
ax_Times_Error.set_ylabel('Error in Deformation Energy [%]')
ax_Times_Error.legend(elemTypes)

# Computation time
ax_Times.grid()
ax_Times.set_xlabel('Degrees of Freedom (DDL)')
ax_Times.set_ylabel('Computation Time [s]')
ax_Times.legend(elemTypes)

# Plot the von Mises stress result using 20 color levels
Display.Plot_Result(simu, "Svm", nColors=20)

if makeParaview:
    # Generate Paraview files for visualization
    PostProcessing.Make_Paraview(folder, simu, details=True)

# Show the total computation time
print()
Tic.Resume()

# Display the computation time history
# Tic.Plot_History(folder)

# Show all plots
plt.show()