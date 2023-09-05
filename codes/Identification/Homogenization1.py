import Display
from Interface_Gmsh import Interface_Gmsh
from Geom import *
import Materials
import Simulations
from BoundaryCondition import LagrangeCondition

plt = Display.plt

# Example from : Computational Homogenization of Heterogeneous Materials with Finite Elements
# http://link.springer.com/10.1007/978-3-030-18383-7
# SECTION 4.7

Display.Clear()

# use Periodic boundary conditions ?
usePER = True 

# --------------------------------------
# Mesh
# --------------------------------------
p0 = Point(-1/2, -1/2)
p1 = Point(1/2, -1/2)
p2 = Point(1/2, 1/2)
p3 = Point(-1/2, 1/2)
pts = [p0, p1, p2, p3]

meshSize = 1/30

contour = PointsList(pts, meshSize)

f = 0.4

r = 1 * np.sqrt(f/np.pi)

inclusion = Circle(Point(), 2*r, meshSize, isHollow=False)

gmshInterface = Interface_Gmsh()

mesh = gmshInterface.Mesh_2D(contour, [inclusion], "TRI6")

coordo = mesh.coordoGlob

Display.Plot_Mesh(mesh)
Display.Plot_Model(mesh)

nodes_left = mesh.Nodes_Conditions(lambda x,y,z: x==-1/2)
# sort by y and exclude first and last nodes
nodes_left = nodes_left[np.argsort(coordo[nodes_left,1])][1:-1]

nodes_right = mesh.Nodes_Conditions(lambda x,y,z: x==1/2)
# sort by y and exclude first and last nodes
nodes_right = nodes_right[np.argsort(coordo[nodes_right,1])][1:-1]

nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==1/2)
# sort by x and exclude first and last nodes
nodes_upper = nodes_upper[np.argsort(coordo[nodes_upper,0])][1:-1]

nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==-1/2)
# sort by x and exclude first and last nodes
nodes_lower = nodes_lower[np.argsort(coordo[nodes_lower,0])][1:-1]

nodes_b0 = np.concatenate((nodes_lower, nodes_left))
nodes_b1 = np.concatenate((nodes_upper, nodes_right))

assert nodes_b0.size == nodes_b1.size, 'Edges must contain the same number of nodes.'

if usePER:
    nodes_border = mesh.Nodes_Tags(["P0", "P1", "P2", "P3"])
else:
    nodes_border = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

# --------------------------------------
# Material and Simulation
# --------------------------------------
elements_inclusion = mesh.Elements_Tags(["S1"])
elements_matrix = mesh.Elements_Tags(["S0"])

E = np.zeros_like(mesh.groupElem.elements, dtype=float)
v = np.zeros_like(mesh.groupElem.elements, dtype=float)

E[elements_matrix] = 1 # MPa
v[elements_matrix] = 0.45

if elements_inclusion.size > 0:
    E[elements_inclusion] = 50
    v[elements_inclusion] = 0.3

material = Materials.Elas_Isot(2, E, v, planeStress=False)

simu = Simulations.Simu_Displacement(mesh, material, useNumba=True)

Display.Plot_Result(simu, E, nodeValues=False, title="E [MPa]")
Display.Plot_Result(simu, v, nodeValues=False, title="v")

# --------------------------------------
# Homogenization
# --------------------------------------
r2 = np.sqrt(2)
E11 = np.array([[1, 0],[0, 0]])
E22 = np.array([[0, 0],[0, 1]])
E12 = np.array([[0, 1/r2],[1/r2, 0]])

def Calc_ukl(Ekl: np.ndarray, pltSol=False):

    simu.Bc_Init()

    func_ux = lambda x, y, z: Ekl.dot([x, y])[0]
    func_uy = lambda x, y, z: Ekl.dot([x, y])[1]
    simu.add_dirichlet(nodes_border, [func_ux, func_uy], ["x","y"])

    if usePER:        
        
        # requires the u field to have zero mean
        useMean0 = False

        for n0, n1 in zip(nodes_b0, nodes_b1):
                
            nodes = np.array([n0, n1])

            # plt.gca().scatter(coordo[nodes, 0],coordo[nodes, 1], marker='+', c='red')

            for direction in ["x", "y"]:
                dofs = simu.Bc_dofs_nodes(nodes, [direction])
                
                values = Ekl @ [coordo[n0,0]-coordo[n1,0], coordo[n0,1]-coordo[n1,1]]
                value = values[0] if direction == "x" else values[1]

                condition = LagrangeCondition("displacement", nodes, dofs, [direction], [value], [1, -1])
                simu._Bc_Add_Lagrange(condition)

        if useMean0:            

            nodes = mesh.nodes
            vect = np.ones(mesh.Nn) * 1/mesh.Nn 

            # sum u_i / Nn = 0
            dofs = simu.Bc_dofs_nodes(nodes, ["x"])
            condition = LagrangeCondition("displacement", nodes, dofs, ["x"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)

            # sum v_i / Nn = 0
            dofs = simu.Bc_dofs_nodes(nodes, ["y"])
            condition = LagrangeCondition("displacement", nodes, dofs, ["y"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)            

    # Display.Plot_BoundaryConditions(simu)

    ukl = simu.Solve()

    # print(np.mean(simu.Get_Resultat("ux")))
    # print(np.mean(simu.Get_Resultat("uy")))

    simu.Save_Iter()

    if pltSol:
        # Display.Plot_Result(simu, "ux", deformation=False)
        # Display.Plot_Result(simu, "uy", deformation=False)

        Display.Plot_Result(simu, "Sxx", factorDef=0.3, deformation=True, nodeValues=True)
        Display.Plot_Result(simu, "Syy", factorDef=0.3, deformation=True, nodeValues=True)
        Display.Plot_Result(simu, "Sxy", factorDef=0.3, deformation=True, nodeValues=True)

    return ukl

u11 = Calc_ukl(E11, False)
u22 = Calc_ukl(E22, False)
u12 = Calc_ukl(E12, True)

u11_e = mesh.Locates_sol_e(u11)
u22_e = mesh.Locates_sol_e(u22)
u12_e = mesh.Locates_sol_e(u12)

# --------------------------------------
# Effective elasticity tensor (C_hom)
# --------------------------------------
U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

matrixType = "mass"
jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
weight_pg = mesh.Get_weight_pg(matrixType)
B_e_pg = mesh.Get_B_e_pg(matrixType)

C_Mat = Materials.Reshape_variable(material.C, mesh.Ne, weight_pg.size)

# Be careful here you have to use all the air even if there are holes remarks ZAKARIA confirmed by saad
# area = 1
area = mesh.area
# if you use the mesh area, multiply C_hom by the porosity (1-f)
C_hom = np.einsum('ep,p,epij,epjk,ekl->il', jacobian_e_pg, weight_pg, C_Mat, B_e_pg, U_e, optimize='optimal') * 1 / area

if inclusion.isHollow and area != 1:
    C_hom *= (1-f)

# Display.Plot_BoundaryConditions(simu)

print(f"f = {f}")
print(f"c1111 = {C_hom[0,0]}")
print(f"c1122 = {C_hom[0,1]}")
print(f"c1212 = {C_hom[2,2]/2}")

plt.show()