"""Conduct full-field homogenization."""

from EasyFEA import (Display, Folder, plt, np,
                     Geoms, Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.fem import LagrangeCondition

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    # use Periodic boundary conditions ?
    usePER = True

    L = 120 # mm
    h = 13
    b = 13

    nL = 40 # number of inclusions following x
    nH = 4 # number of inclusions following y
    isHollow = True # hollow inclusions

    # c = 13/2
    cL = L/(2*nL)
    cH = h/(2*nH)

    E = 210000
    v = 0.3

    load = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    elemType = ElemType.TRI3
    meshSize = h/20

    pt1 = Geoms.Point()
    pt2 = Geoms.Point(L,0)
    pt3 = Geoms.Point(L,h)
    pt4 = Geoms.Point(0,h)

    domain = Geoms.Domain(pt1, pt2, meshSize)

    inclusions = []
    for i in range(nL):
        x = cL + cL*(2*i)
        for j in range(nH):
            y = cH + cH*(2*j)

            ptd1 = Geoms.Point(x-cL/2, y-cH/2)
            ptd2 = Geoms.Point(x+cL/2, y+cH/2)

            inclusion = Geoms.Domain(ptd1, ptd2, meshSize, isHollow)

            inclusions.append(inclusion)

    mesher = Mesher(False)

    inclusion = Geoms.Domain(ptd1, ptd2, meshSize)
    area_inclusion = mesher.Mesh_2D(inclusion).area

    points = Geoms.Points([pt1, pt2, pt3, pt4], meshSize)

    # mesh with inclusions
    mesh_inclusions = mesher.Mesh_2D(points, inclusions, elemType)

    # mesh without inclusions
    mesh = mesher.Mesh_2D(points, [], elemType)

    ptI1 = Geoms.Point(-cL,-cH)
    ptI2 = Geoms.Point(cL,-cH)
    ptI3 = Geoms.Point(cL, cH)
    ptI4 = Geoms.Point(-cL, cH)

    pointsI = Geoms.Points([ptI1, ptI2, ptI3, ptI4], meshSize/4)

    mesh_VER = mesher.Mesh_2D(pointsI, [Geoms.Domain(Geoms.Point(-cL/2,-cH/2), Geoms.Point(cL/2, cH/2),
                                                        meshSize/4, isHollow)], elemType)
    area_VER = mesh_VER.area

    Display.Plot_Mesh(mesh_inclusions, title='non hom')
    Display.Plot_Mesh(mesh_VER, title='VER')
    Display.Plot_Mesh(mesh, title='hom')

    # ----------------------------------------------
    # Material
    # ----------------------------------------------

    # elastic behavior of the beam
    material_inclsuion = Materials.Elas_Isot(2, E=E, v=v, planeStress=True, thickness=b)
    CMandel = material_inclsuion.C

    material = Materials.Elas_Anisot(2, CMandel, False)
    testC = np.linalg.norm(material_inclsuion.C-material.C)/np.linalg.norm(material_inclsuion.C)
    assert testC < 1e-12, "the matrices are different"

    # ----------------------------------------------
    # Homogenization
    # ----------------------------------------------
    simu_inclusions = Simulations.ElasticSimu(mesh_inclusions, material_inclsuion)
    simu_VER = Simulations.ElasticSimu(mesh_VER, material_inclsuion)
    simu = Simulations.ElasticSimu(mesh, material)

    r2 = np.sqrt(2)
    E11 = np.array([[1, 0],[0, 0]])
    E22 = np.array([[0, 0],[0, 1]])
    E12 = np.array([[0, 1/r2],[1/r2, 0]])

    if usePER:
        nodes_border = mesh_VER.Nodes_Tags(["P0","P1","P2","P3"])
        paired_nodes = mesh_VER.Get_Paired_Nodes(nodes_border, True)
    else:
        nodes_border = mesh_VER.Nodes_Tags(["L0", "L1", "L2", "L3"])

    def Calc_ukl(Ekl: np.ndarray):

        simu_VER.Bc_Init()

        func_ux = lambda x, y, z: Ekl.dot([x, y])[0]
        func_uy = lambda x, y, z: Ekl.dot([x, y])[1]
        simu_VER.add_dirichlet(nodes_border, [func_ux, func_uy], ["x","y"])

        if usePER:

            coordo = mesh_VER.coord

            for n0, n1 in paired_nodes:
                    
                nodes = np.array([n0, n1])

                for direction in ["x", "y"]:
                    dofs = simu_VER.Bc_dofs_nodes(nodes, [direction])
                    
                    values = Ekl @ [coordo[n0,0]-coordo[n1,0], coordo[n0,1]-coordo[n1,1]]
                    value = values[0] if direction == "x" else values[1]

                    condition = LagrangeCondition("elastic", nodes, dofs, [direction], [value], [1, -1])
                    simu_VER._Bc_Add_Lagrange(condition)

        ukl = simu_VER.Solve()

        simu_VER.Save_Iter()
        # Display.Plot_Result(simu_VER, "Exx")
        # Display.Plot_Result(simu_VER, "Eyy")
        # Display.Plot_Result(simu_VER, "Exy")

        return ukl

    u11 = Calc_ukl(E11)
    u22 = Calc_ukl(E22)
    u12 = Calc_ukl(E12)

    u11_e = mesh_VER.Locates_sol_e(u11)
    u22_e = mesh_VER.Locates_sol_e(u22)
    u12_e = mesh_VER.Locates_sol_e(u12)

    U_e = np.zeros((u11_e.shape[0],u11_e.shape[1], 3))

    U_e[:,:,0] = u11_e; U_e[:,:,1] = u22_e; U_e[:,:,2] = u12_e

    matrixType = "rigi"
    jacobien_e_pg = mesh_VER.Get_jacobian_e_pg(matrixType)
    poids_pg = mesh_VER.Get_weight_pg(matrixType)
    B_e_pg = mesh_VER.Get_B_e_pg(matrixType)

    C_hom = np.einsum('ep,p,ij,epjk,ekl->il', jacobien_e_pg, poids_pg, CMandel, B_e_pg, U_e, optimize='optimal') * 1/mesh_VER.area

    if isHollow:
        coef = (1 - area_inclusion/area_VER)
        C_hom *= coef

    # print(np.linalg.eigvals(C_hom))

    # ----------------------------------------------
    # Comparison
    # ----------------------------------------------
    def Simulation(simu: Simulations._Simu, title=""):

        simu.Bc_Init()

        simu.add_dirichlet(simu.mesh.Nodes_Tags(['L3']), [0,0], ['x', 'y'])
        simu.add_surfLoad(simu.mesh.Nodes_Tags(['L1']), [-load/(b*h)], ['y'])

        simu.Solve()

        # Display.Plot_BoundaryConditions(simu)
        Display.Plot_Result(simu, "uy", title=f"{title} uy")
        # Display.Plot_Result(simu, "Eyy")

        print(f"{title}: dy = {np.max(simu.Result('uy')[simu.mesh.Nodes_Point(Geoms.Point(L,0))]):.3f}")

    Simulation(simu_inclusions, "inclusions")
    Simulation(simu, "non hom")

    testSym = np.linalg.norm(C_hom.T - C_hom)/np.linalg.norm(C_hom)

    if testSym >= 1e-12 and testSym <= 1e-3:
        C_hom = 1/2 * (C_hom.T + C_hom)

    material.Set_C(C_hom, False)
    Simulation(simu, "hom")

    # ax = Display.Plot_Result(simu, "uy")
    # Display.Plot_Result(simuInclusions, "uy", ax=ax)

    plt.show()