import Folder
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import normalize_vect
import Display
import Materials
import Simulations
from BoundaryCondition import BoundaryCondition, LagrangeCondition

Display.Clear()
np = Display.np

# use Periodic boundary conditions ?
usePER = False

# --------------------------------------
# Mesh
# --------------------------------------

# meshFile = 'D6_TRI3.msh'
meshFile = 'D6_TRI6.msh'

meshFile = Folder.Join([Folder.Get_Path(), 'codes', '_parts_and_meshes', meshFile])

mesh = Interface_Gmsh().Mesh_Import_mesh(meshFile)
Display.Plot_Mesh(mesh)
coordo = mesh.coordo

nodes_matrix = mesh.Nodes_Tags(['S0'])
elements_matrix = mesh.Elements_Nodes(nodes_matrix)

nodes_inclusion = mesh.Nodes_Tags(['S1'])
elements_inclusion = mesh.Elements_Nodes(nodes_inclusion)

nodes_corners = mesh.nodes[:6]

if usePER:
    nodes_border = nodes_corners.copy()
else:
    nodes_border = mesh.Nodes_Tags([f'L{i}' for i in range(6)])

paired_tags = [('L0','L3'),
               ('L1','L4'),
               ('L2','L5')]

list_nodes1 = []
list_nodes2 = []

# ax = Display.Plot_Mesh(mesh, alpha=0)

list1, list2 =  zip(*paired_tags)
for tag1, tag2 in zip(list1, list2):

    nodes1 = mesh.Nodes_Tags([tag1])
    nodes1 = nodes1[np.argsort(coordo[nodes1, 1])][1:-1] # sort by y and exclude first and last nodes

    nodes2 = mesh.Nodes_Tags([tag2])
    nodes2 = nodes2[np.argsort(coordo[nodes2, 1])][1:-1] # sort by y and exclude first and last nodes

    assert nodes1.size == nodes2.size, 'Edges must contain the same number of nodes.'

    list_nodes1.extend(nodes1)
    list_nodes2.extend(nodes2)

    # plot the paired nodes
    # [Display.Plot_Nodes(mesh, [n1, n2], showId=True, ax=ax) for n1 ,n2 in zip(nodes1, nodes2)]

# --------------------------------------
# Simulation
# --------------------------------------

Display.Plot_Mesh(mesh)

E = np.ones(mesh.Ne) * 70 * 1e9
E[elements_inclusion] = 200 * 1e9

v = np.ones(mesh.Ne) * 0.45
v[elements_inclusion] = 0.3

Display.Plot_Result(mesh, E*1e-9, nodeValues=False, title='E [GPa]')
Display.Plot_Result(mesh, v, nodeValues=False, title='v')

material = Materials.Elas_Isot(2, E, v, planeStress=False)

simu = Simulations.Simu_Displacement(mesh, material, useIterativeSolvers=False)

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
        useMean0 = True        

        for n1, n2 in zip(list_nodes1, list_nodes2):
                
            nodes = np.array([n1, n2])

            # plt.gca().scatter(coordo[nodes, 0],coordo[nodes, 1], marker='+', c='red')

            for direction in ["x", "y"]:
                ddls = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes, [direction])
                
                values = Ekl @ [coordo[n1,0]-coordo[n2,0], coordo[n1,1]-coordo[n2,1]]
                value = values[0] if direction == "x" else values[1]

                condition = LagrangeCondition("displacement", nodes, ddls, [direction], [value], [1, -1])
                simu._Bc_Add_Lagrange(condition)

        if useMean0:            

            nodes = mesh.nodes
            vect = np.ones(mesh.Nn) * 1/mesh.Nn 

            # sum u_i / Nn = 0
            ddls = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes, ["x"])        
            condition = LagrangeCondition("displacement", nodes, ddls, ["x"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)

            # sum v_i / Nn = 0
            ddls = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes, ["y"])        
            condition = LagrangeCondition("displacement", nodes, ddls, ["y"], [0], [vect])
            simu._Bc_Add_Lagrange(condition)

    ukl = simu.Solve()

    simu.Save_Iter()

    if pltSol:
        Display.Plot_Result(simu, "ux", deformation=False)
        Display.Plot_Result(simu, "uy", deformation=False)

        Display.Plot_Result(simu, "Sxx", factorDef=0.3, deformation=True, nodeValues=True, coef=1e-9)
        Display.Plot_Result(simu, "Syy", factorDef=0.3, deformation=True, nodeValues=True, coef=1e-9)
        Display.Plot_Result(simu, "Sxy", factorDef=0.3, deformation=True, nodeValues=True, coef=1e-9)

        # Display.Plot_Result(simu, "Exx", factorDef=0.3, deformation=True, nodeValues=True)
        # Display.Plot_Result(simu, "Eyy", factorDef=0.3, deformation=True, nodeValues=True)
        # Display.Plot_Result(simu, "Exy", factorDef=0.3, deformation=True, nodeValues=True)

    return ukl

u11 = Calc_ukl(E11, False)
u22 = Calc_ukl(E22, False)
u12 = Calc_ukl(E12, True)

u11_e = mesh.Locates_sol_e(u11)
u22_e = mesh.Locates_sol_e(u22)
u12_e = mesh.Locates_sol_e(u12)

# --------------------------------------
# Effective elasticity tensor
# --------------------------------------

U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

matrixType = "mass"
jacobian_e_pg = mesh.Get_jacobian_e_pg(matrixType)
weight_pg = mesh.Get_weight_pg(matrixType)
B_e_pg = mesh.Get_B_e_pg(matrixType)

C_Mat = Materials.Reshape_variable(material.C, mesh.Ne, weight_pg.size)

C_hom = np.einsum('ep,p,epij,epjk,ekl->il', jacobian_e_pg, weight_pg, C_Mat, B_e_pg, U_e, optimize='optimal') * 1 / mesh.area

print(f"c1111 = {C_hom[0,0]}")
print(f"c1122 = {C_hom[0,1]}")
print(f"c1212 = {C_hom[2,2]/2}")

Display.plt.show()