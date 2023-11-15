from Interface_Gmsh import gmsh, Interface_Gmsh, Point, Circle, Domain, ElemType, PointsList, Line, CircleArc
import Display
import Folder
import numpy as np

folder = Folder.Get_Path(__file__)

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Geom
    # ----------------------------------------------
    H = 90
    L = 45
    D = 10
    e = 20    

    N = 5
    mS = (np.pi/4 * D/2) / N

    # PI for Points
    # pi for gmsh points
    PC = Point(0, H/2, 0)
    circle = Circle(PC, D, mS)
    
    P1 = Point(-L/2,0)
    P2 = Point(L/2,0)
    P3 = Point(L/2,H)
    P4 = Point(-L/2,H)
    contour = PointsList([P1, P2, P3, P4])

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    interface = Interface_Gmsh(False, True)
    dim, elemType = 2, ElemType.QUAD4

    factory = interface._init_gmsh_factory('occ')

    # mesh = interface.Mesh_2D(contour, [circle], elemType)

    # gmsh points for the domain
    p1 = factory.addPoint(*P1.coordo, meshSize=mS)
    p2 = factory.addPoint(*P2.coordo, meshSize=mS)
    p3 = factory.addPoint(*P3.coordo, meshSize=mS)
    p4 = factory.addPoint(*P4.coordo, meshSize=mS)
    
    # gmsh points for the circle
    dx = np.cos(np.pi/4) * D/2
    dy = np.sin(np.pi/4) * D/2
    pc = factory.addPoint(*PC.coordo, meshSize=mS)
    pc1 = factory.addPoint(*(PC+[D/2,0]).coordo, meshSize=mS)
    pc2 = factory.addPoint(*(PC+[0,D/2]).coordo, meshSize=mS)
    pc3 = factory.addPoint(*(PC+[-D/2,0]).coordo, meshSize=mS)
    pc4 = factory.addPoint(*(PC+[0,-D/2]).coordo, meshSize=mS)
    pc12 = factory.addPoint(*(PC+[dx,dy]).coordo, meshSize=mS)
    pc23 = factory.addPoint(*(PC+[-dx,dy]).coordo, meshSize=mS)
    pc34 = factory.addPoint(*(PC+[-dx,-dy]).coordo, meshSize=mS)
    pc41 = factory.addPoint(*(PC+[dx,-dy]).coordo, meshSize=mS)

    p23 = factory.addPoint(*(P2+P3).coordo/2, meshSize=mS)
    p34 = factory.addPoint(*(P3+P4).coordo/2, meshSize=mS)
    p41 = factory.addPoint(*(P4+P1).coordo/2, meshSize=mS)
    p12 = factory.addPoint(*(P1+P2).coordo/2, meshSize=mS)

    points_1 = [pc1, pc12, pc2, pc23, pc3, pc34, pc4, pc41]
    points_2 = [p23, p3, p34, p4, p41, p1, p12, p2]
    points_3 = [p3, p34, p4, p41, p1, p12, p2, p23]
    points_4 = [pc12, pc2, pc23, pc3, pc34, pc4, pc41, pc1]

    fuse = True
    addedLines = []
    for p in range(len(points_1)):

        if p == 0:
            line1 = factory.addLine(points_1[p], points_2[p])
            firsLine = line1
        elif fuse:
            line1 = line3
        else:
            line1 = factory.addLine(points_1[p], points_2[p])
        line2 = factory.addLine(points_2[p], points_3[p])    
        if p+1 == len(points_1) and fuse:
            line3 = firsLine
        else:
            line3 = factory.addLine(points_3[p], points_4[p])
        line4 = factory.addCircleArc(points_4[p], pc, points_1[p])

        lines = [line1, line2, line3, line4]
        addedLines.extend(lines)

        loop = factory.addCurveLoop(lines)
        surf = factory.addPlaneSurface([loop])

        factory.synchronize()

        [gmsh.model.mesh.setTransfiniteCurve(line, N) for line in lines]

    factory.remove([(0, pc)])

    # factory.fragment(factory.getEntities(2), [(1, l) for l in addedLines])

    factory.synchronize()

    interface._Set_PhysicalGroups()
    
    interface._Meshing(dim, elemType, isOrganised=True)

    mesh = interface._Construct_Mesh()

    if len(mesh.orphanNodes) > 0:
        ax = Display.Plot_Nodes(mesh, mesh.orphanNodes)
        ax.set_title("Orphan nodes")

    print(mesh)

    if dim == 3:
        print(f'volume = {mesh.volume:.3f}')


    import Simulations
    import Materials

    mat = Materials.Elas_Isot(mesh.dim)
    simu = Simulations.Simu_Displacement(mesh, mat)

    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==0), [0]*mesh.dim, simu.Get_directions())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==H), [1], ['y'])    
    simu.Solve()

    Display.Plot_Result(simu, 'uy', True, 4, plotMesh=True)
        
    Display.Plot_Model(mesh)

    Display.Plot_Mesh(mesh)

    Display.plt.show()