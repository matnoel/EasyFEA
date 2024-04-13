"""Meshing of a perforated plate with a structured mesh."""

from EasyFEA import (Display, Folder, np,
                     Mesher, ElemType, 
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Circle, Points, Line, CircleArc, Contour

folder = Folder.Get_Path(__file__)

if __name__ == '__main__':

    dim = 2

    if dim == 2:
        elemType =  ElemType.QUAD4
    else:
        elemType =  ElemType.HEXA8

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
    PC = Point(L/2, H/2, 0)
    circle = Circle(PC, D, mS)

    P1 = Point()
    P2 = Point(L,0)
    P3 = Point(L,H)
    P4 = Point(0,H)
    contour1 = Points([(P3+P2)/2,P3,(P3+P4)/2,
                           P4,(P4+P1)/2,P1,
                           (P1+P2)/2,P2,(P3+P2)/2], mS)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    mesher = Mesher(False, True, True)
    factory = mesher._factory
    

    contours1: list[Contour] = []
    
    for c in range(4):

        pc = circle.center
        pc1 = circle.contour.geoms[c].pt1
        pc2 = circle.contour.geoms[c].pt2
        pc3 = circle.contour.geoms[c].pt3
        
        p1,p2,p3 = contour1.points[c*2:c*2+3]

        cont1 = Contour([Line(pc1, p1), 
                            Line(p1,p2),
                            Line(p2,pc3),
                            CircleArc(pc3,pc1,pc)])
        loop1, lines1, points1 = mesher._Loop_From_Geom(cont1)

        cont2 = Contour([Line(pc3, p2),
                            Line(p2,p3),
                            Line(p3,pc2),
                            CircleArc(pc2,pc3,pc)])
        loop2, lines2, points2 = mesher._Loop_From_Geom(cont2)

        surf1 = factory.addSurfaceFilling(loop1)
        surf2 = factory.addSurfaceFilling(loop2)

        mesher._OrganiseSurfaces([surf1, surf2], elemType, True, [N]*4)

        contours1.extend([cont1, cont2])
    
    cont1.Plot_Geoms(contours1)

    if dim == 3:
        
        for cont1 in contours1:
            cont2 = cont1.Copy()
            cont2.Translate(dz=e)
            # cont2.rotate(np.pi/8, PC.coordo)
            mesher._Link_Contours(cont1, cont2, elemType, 3, [N]*4)

    mesher._Set_PhysicalGroups()
    
    mesher._Meshing(dim, elemType)

    mesh = mesher._Construct_Mesh()

    if len(mesh.orphanNodes) > 0:
        ax = Display.Plot_Nodes(mesh, mesh.orphanNodes)
        ax.set_title("Orphan nodes detected")
        Display.plt.show()

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    mat = Materials.Elas_Isot(mesh.dim)
    simu = Simulations.ElasticSimu(mesh, mat)

    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==0), [0]*mesh.dim, simu.Get_dofs())
    simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==H), [4], ['y'])    
    simu.Solve()
        
    Display.Plot_Tags(mesh, alpha=0.1, showId=False)
    Display.Plot_Mesh(simu, 1)
    Display.Plot_Result(simu, 'uy', 1, plotMesh=True)

    print(simu)

    Display.plt.show()