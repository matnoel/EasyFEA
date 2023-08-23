import Folder
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import normalize_vect, Point, PointsList, Line, Circle
import Display
import Materials
import Simulations
from BoundaryCondition import BoundaryCondition, LagrangeCondition

Display.Clear()
np = Display.np

# use Periodic boundary conditions ?
usePER = True

elemType = ElemType.TRI6

geom = 'D666' # hexagon
# geom = 'D2' # rectangle
# geom = 'D6'

hollowInclusion = True

# --------------------------------------
# Mesh
# --------------------------------------

N = 10

if geom == 'D666':

    a = 1
    R = 2*a/np.sqrt(3)
    r = R/np.sqrt(2)/2
    phi = np.pi/6

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Creates the contour geometrie
    p0 = Point(0, R)
    p1 = Point(-cos_phi*R, sin_phi*R)
    p2 = Point(-cos_phi*R, -sin_phi*R)
    p3 = Point(0, -R)
    p4 = Point(cos_phi*R, -sin_phi*R)
    p5 = Point(cos_phi*R, sin_phi*R)
    # edge length and area
    s = Line(p0,p1).length
    area = 3*np.sqrt(3)/2*s**2

    contour = PointsList([p0,p1,p2,p3,p4,p5], s/N)
    corners = contour.points

    # Creates the inclusion
    p6 = Point(0, (R-r))
    p7 = Point(-cos_phi*(R-r), sin_phi*(R-r))
    p8 = Point(-cos_phi*(R-r), -sin_phi*(R-r))
    p9 = Point(0, -(R-r))
    p10 = Point(cos_phi*(R-r), -sin_phi*(R-r))
    p11 = Point(cos_phi*(R-r), sin_phi*(R-r))
    inclusions = [PointsList([p6,p7,p8,p9,p10,p11], s/N, hollowInclusion)]    

elif geom == 'D2':

    a = 1 # width
    b = 1.4 # height
    e = 1/10 # thickness
    area = a*b
    meshSize = e/N*2
    
    # Creates the contour geometry
    p0 = Point(-a/2, b/2)
    p1 = Point(-a/2, -b/2)
    p2 = Point(a/2, -b/2)
    p3 = Point(a/2, b/2)
    contour = PointsList([p0,p1,p2,p3], meshSize)
    corners = contour.points

    # Creates the inclusion geometry
    p4 = p0 + [e, -e]
    p5 = p1 + [e, e]
    p6 = p2 + [-e, e]
    p7 = p3 + [-e, -e]
    inclusions = [PointsList([p4,p5,p6,p7], meshSize, hollowInclusion)]

elif geom  == 'D6':

    a=1 # height
    b=2 # width
    c = np.sqrt(a**2+b**2)
    
    e = b/10 # thickness
    l1 = b/2

    area = a*b
    
    theta = np.arctan(a/b)
    alpha = (np.pi/2 - theta)/2; cos_alpha = np.cos(alpha); sin_alpha = np.sin(alpha)    
    phi = np.pi/3; cos_phi = np.cos(phi); sin_phi = np.sin(phi)
    
    l2 = (b - l1*sin_alpha)/2
    hx = e/cos_phi/4
    hy = e/sin_phi/4

    # symmetry functions
    def Sym_x(point: Point) -> Point:
        return Point(-point.x, point.y)
    def Sym_y(point: Point) -> Point:
        return Point(point.x, -point.y)
    
    # points in the non-rotated base
    p0 = Point(l1/2 + l2*cos_phi + e/2*cos_alpha, l2*sin_phi - e/2*sin_alpha)
    p1 = p0 + [-e*cos_alpha, e*sin_alpha]
    p2 = Point(l1/2-hy,hx)
    p3 = Sym_x(p2)
    p4 = Sym_x(p1)
    p5 = Sym_x(p0)
    p6 = Point(-l1/2- np.sqrt(hx**2+hy**2))
    p7 = Sym_y(p5)
    p8 = Sym_y(p4)
    p9 = Sym_y(p3)
    p10 = Sym_y(p2)
    p11 = Sym_y(p1)
    p12 = Sym_y(p0)
    p13 = Sym_x(p6)

    # do some tests to check if the geometry has been created correctly
    t1 = Line(p2, p10).length
    t2 = Line(p2, p13).length
    t3 = Line(p10, p13).length
    assert np.abs(e-(t1+t2+t3)/3)/e <= 1e-12 # check that t1 = t2 = t3 = e    
    t4 = Line(p0, p1).length
    assert np.abs(t4-e)/e <= 1e-12 # check that t4 = e

    alpha = -alpha
    rot = np.array([[np.cos(alpha),-np.sin(alpha), 0],
                      [np.sin(alpha), np.cos(alpha), 0],
                      [0,0,1]])
    
    rotate_points = []
    ax = Display.plt.subplots()[1]
    for p, point in enumerate([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]):

        assert isinstance(point, Point)

        newCoord = rot @ point.coordo

        ax.scatter(*newCoord[:2], c='black')
        ax.text(*newCoord[:2], f'p{p}', c='black')

        rotate_points.append(Point(*newCoord))
    
    corners = [rotate_points[p] for p in [0,1,4,5,7,8,11,12]]
    
    hollowInclusion = True

    contour = PointsList(rotate_points, e/N*2)

    inclusions = []

else:
    raise Exception('Unknown geom')

mesh = Interface_Gmsh().Mesh_2D(contour, inclusions, elemType)

Display.Plot_Mesh(mesh)
Display.Plot_Model(mesh)
coordo = mesh.coordo

nodes_matrix = mesh.Nodes_Tags(['S0'])
elements_matrix = mesh.Elements_Nodes(nodes_matrix)

if not hollowInclusion:
    nodes_inclusion = mesh.Nodes_Tags(['S1'])
    elements_inclusion = mesh.Elements_Nodes(nodes_inclusion)

nCorners = len(corners)
nEdges = nCorners//2

if usePER:
    nodes_border = np.unique([mesh.Nodes_Point(point) for point in corners])

    group_nodes1 = []; list_nodes1 = []
    group_nodes2 = []; list_nodes2 = []

    for c, corner in enumerate(corners):
        
        if c+1 == nCorners:
            next_corner = corners[0]
        else:
            next_corner = corners[c+1]

        line = next_corner.coordo - corner.coordo
        lineLength = np.linalg.norm(line)
        vect = normalize_vect(line) # normalized vector between the edge corners
        vect_i = coordo - corner.coordo # vector coordinates from the first corner of the edge
        scalarProduct = np.einsum('ni,i', vect_i, vect, optimize="optimal")
        crossProduct = np.cross(vect_i, vect)
        norm = np.linalg.norm(crossProduct, axis=1)

        eps=1e-12
        nodes = np.where((norm<eps) & (scalarProduct>=-eps) & (scalarProduct<=lineLength+eps))[0]

        nodes = nodes[np.argsort(scalarProduct[nodes])][1:-1]

        if c+1 > nEdges:
            nodes = nodes[::-1]
            group_nodes2.append(nodes)
            list_nodes2.extend(nodes)
        else:
            group_nodes1.append(nodes)
            list_nodes1.extend(nodes)

    group_nodes1 = [group_nodes1[p] for p in range(nEdges) if group_nodes1[p].size > 0]
    group_nodes2 = [group_nodes2[p] for p in range(nEdges) if group_nodes2[p].size > 0]

    ax = Display.Plot_Mesh(mesh, alpha=0, title='Periodic boundary conditions')
    from matplotlib.collections import LineCollection
    for n, (nodes1, nodes2) in enumerate(zip(group_nodes1, group_nodes2)):

        paired_nodes = np.concatenate((nodes1.reshape(-1,1), nodes2.reshape(-1,1)), axis=1)
        lines = coordo[paired_nodes, :2]

        pc = ax.scatter(lines[:,:,0], lines[:,:,1], label=f'edges{n}')
        ax.add_collection(LineCollection(lines, edgecolor=pc.get_edgecolor()))    
    ax.legend()
else:
    nodes_border = mesh.Nodes_Tags([f'L{i}' for i in range(6)])

# --------------------------------------
# Simulation
# --------------------------------------

E = np.ones(mesh.Ne) * 70 * 1e9
v = np.ones(mesh.Ne) * 0.45

if not hollowInclusion:
    E[elements_inclusion] = 200 * 1e9
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

C_hom = np.einsum('ep,p,epij,epjk,ekl->il', jacobian_e_pg, weight_pg, C_Mat, B_e_pg, U_e, optimize='optimal') * 1 / area

print(f"c1111 = {C_hom[0,0]}")
print(f"c1122 = {C_hom[0,1]}")
print(f"c1212 = {C_hom[2,2]/2}")

Display.plt.show()