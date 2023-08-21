# Import required libraries and modules
import matplotlib.pyplot as plt
import numpy as np
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Domain, Line, Point, Section
import Display
import Materials
import Simulations
import Folder
import PostProcessing

Display.Clear()

# Create a new folder for storing results
folder = Folder.New_File("Beam", results=True)

# Initialize the Gmsh interface
interfaceGmsh = Interface_Gmsh(False, False, False)

# Define the problem type and beam dimensions
# problem = "Flexion"
# problem = "BiEnca"
problem = "Portique"

elemType = ElemType.SEG4

beamDim = 3

# Set problem-specific parameters
if problem in ["Flexion", "BiEnca", "Portique"]:
    L = 120
    nL = 1
    h = 13
    b = 13
    E = 210000
    v = 0.3
    charge = 800
elif problem == "Traction":
    L = 10
    nL = 1
    h = 0.1
    b = 0.1
    E = 200000e6
    ro = 7800
    v = 0.3
    g = 10
    q = ro * g * (h * b)
    charge = 5000

# Create a section object for the beam mesh
section = Section(interfaceGmsh.Mesh_2D(Domain(Point(x=-b / 2, y=-h / 2), Point(x=b / 2, y=h / 2))))

# Depending on the problem type, create appropriate beam segments
if problem in ["Traction"]:
    point1 = Point()
    point2 = Point(x=L / 2)
    point3 = Point(x=L)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    poutre1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, v)
    poutre2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, v)
    liste_Poutre = [poutre1, poutre2]

elif problem in ["Flexion", "BiEnca"]:
    point1 = Point()
    point2 = Point(x=L / 2)
    point3 = Point(x=L)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    line = Line(point1, point3)
    poutre1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, v)
    poutre2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, v)
    liste_Poutre = [poutre1, poutre2]

elif problem == "Portique":
    point1 = Point()
    point2 = Point(y=L)
    point3 = Point(y=L, x=L / 2)
    line1 = Line(point1, point2, L / nL)
    line2 = Line(point2, point3, L / nL)
    poutre1 = Materials.Beam_Elas_Isot(beamDim, line1, section, E, v)
    poutre2 = Materials.Beam_Elas_Isot(beamDim, line2, section, E, v)
    liste_Poutre = [poutre1, poutre2]

# Generate the beam mesh
mesh = interfaceGmsh.Mesh_Beams(beamList=liste_Poutre, elemType=elemType)

# Plot the initial model and boundary conditions
Display.Plot_Model(mesh)

# Initialize the beam structure with the defined beam segments
beamStructure = Materials.Beam_Structure(liste_Poutre)

# Create the beam simulation
simu = Simulations.Simu_Beam(mesh, beamStructure, verbosity=True)

# Set Dirichlet boundary conditions based on the beam's dimension and problem type
if beamStructure.dim == 1:
    simu.add_dirichlet(mesh.Nodes_Point(point1), [0], ["x"])
    if problem == "BiEnca":
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0], ["x"])
elif beamStructure.dim == 2:
    simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0, 0], ["x", "y", "rz"])
    if problem == "BiEnca":
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0, 0], ["x", "y", "rz"])
elif beamStructure.dim == 3:
    simu.add_dirichlet(mesh.Nodes_Point(point1), [0, 0, 0, 0, 0, 0], ["x", "y", "z", "rx", "ry", "rz"])
    if problem == "BiEnca":
        simu.add_dirichlet(mesh.Nodes_Point(point3), [0, 0, 0, 0, 0, 0], ["x", "y", "z", "rx", "ry", "rz"])

# Add connection constraints for multi-beam systems
if beamStructure.nBeam > 1:
    simu.add_connection_fixed(mesh.Nodes_Point(point2))

# Set Neumann boundary conditions based on the problem type
if problem in ["Flexion"]:
    simu.add_neumann(mesh.Nodes_Point(point3), [-charge], ["y"])
elif problem == "Portique":
    simu.add_neumann(mesh.Nodes_Point(point3), [-charge, charge], ["y", "z"])
elif problem == "BiEnca":
    simu.add_neumann(mesh.Nodes_Point(point2), [-charge], ["y"])
elif problem == "Traction":
    noeudsLine = mesh.Nodes_Line(Line(point1, point3))
    simu.add_lineLoad(noeudsLine, [q], ["x"])
    simu.add_neumann(mesh.Nodes_Point(point3), [charge], ["x"])

# Display the boundary conditions on the model
Display.Plot_BoundaryConditions(simu)

# Solve the beam problem and get displacement results
beamDisplacement = simu.Solve()
simu.Save_Iter()

# Calculate stresses and forces
stress = simu.Get_Result("Stress")
forces = stress / section.area

# Function for printing result information
affichage = lambda name, result: print(f"{name} = [{result.min():2.2}; {result.max():2.2}]") if isinstance(result, np.ndarray) else ""

# Plot the boundary conditions and displacement results
Display.Plot_BoundaryConditions(simu)
Display.Plot_Result(simu, "ux", plotMesh=False, deformation=False)
if beamStructure.dim > 1:
    Display.Plot_Result(simu, "uy", plotMesh=False, deformation=False)
    Display.Plot_Mesh(simu, deformation=True, factorDef=10)

# Display section for results
Display.Section("Resultats")

# Print displacement results at nodes
print()
u = simu.Get_Result("ux", nodeValues=True);
affichage("ux", u)
if beamStructure.dim > 1:
    v = simu.Get_Result("uy", nodeValues=True);
    affichage("uy", v)
    rz = simu.Get_Result("rz", nodeValues=True);
    affichage("rz", rz)

# Analytical solution for Flexion problem
if problem == "Flexion":
    listX = np.linspace(0, L, 100)
    v_x = charge / (E * section.Iz) * (listX ** 3 / 6 - (L * listX ** 2) / 2)
    flecheanalytique = charge * L ** 3 / (3 * E * section.Iz)
    erreur = np.abs(flecheanalytique + v.min()) / flecheanalytique
    rapport = np.abs(flecheanalytique / v.min())

    # Plot the analytical and finite element solutions for vertical displacement (v)
    fig, ax = plt.subplots()
    ax.plot(listX, v_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:, 0], v, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$v(x)$")
    ax.legend()

    rz_x = charge / E / section.Iz * (listX ** 2 / 2 - L * listX)
    rotalytique = -charge * L ** 2 / (2 * E * section.Iz)

    # Plot the analytical and finite element solutions for rotation (rz)
    fig, ax = plt.subplots()
    ax.plot(listX, rz_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:, 0], rz, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$r_z(x)$")
    ax.legend()
elif problem == "Traction":
    listX = np.linspace(0, L, 100)
    u_x = (charge * listX / (E * (section.area))) + (ro * g * listX / 2 / E * (2 * L - listX))
    rapport = u_x[-1] / u.max()

    # Plot the analytical and finite element solutions for displacement (u)
    fig, ax = plt.subplots()
    ax.plot(listX, u_x, label='Analytique', c='blue')
    ax.scatter(mesh.coordo[:, 0], u, label='EF', c='red', marker='x', zorder=2)
    ax.set_title(fr"$u(x)$")
    ax.legend()

# # Post-process and save results in Paraview format
# PostProcessing.Make_Paraview(folder, simu)

# Display all plots
plt.show()
