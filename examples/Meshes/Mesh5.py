"""Mesh of a cracked part."""

from EasyFEA import Display, Mesher, ElemType, Materials, Simulations
from EasyFEA.Geoms import Point, Line, Points, Domain, Contour

if __name__ == '__main__':

    Display.Clear()

    L = 1
    openCrack = True

    contour = Domain(Point(), Point(L, L))

    def DoMesh(dim, elemType, cracks=[]):
        if dim == 2:
            mesh = Mesher().Mesh_2D(contour, [], elemType, cracks)
        elif dim == 3:
            # WARNING :
            # only works with TETRA4 and TETRA10
            # only works with nLayers = []            
            mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, L], [], elemType, cracks)

        material = Materials.Elas_Isot(dim)
        simu = Simulations.ElasticSimu(mesh, material)

        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==0), [0]*dim, simu.Get_dofs())
        simu.add_dirichlet(mesh.Nodes_Conditions(lambda x,y,z: y==L), [L*0.05], ['y'])
        simu.Solve()
        Display.Plot_Result(simu, 'uy', 1, plotMesh=True)

        return mesh

    # ----------------------------------------------
    # 2D CRACK
    # ----------------------------------------------

    crack1 = Line(Point(L/4,L/2), Point(3*L/4,L/2), isOpen=openCrack)
    crack2 = Line(Point(0,L/3, isOpen=openCrack), Point(L/2,L/3), isOpen=openCrack)
    crack3 = Line(Point(0,2*L/3, isOpen=openCrack), Point(L/2,2*L/3), isOpen=openCrack)
    crack4 = Line(Point(0,4*L/5), Point(L,4*L/5), isOpen=False)
    crack5 = Points([Point(L/2,L/5),
                        Point(2*L/3,L/5),
                        Point(L,L/10, isOpen=True)], isOpen=True)

    cracks = [crack1, crack2, crack3, crack4, crack5]

    meshes2D = [DoMesh(2, elemType, cracks) for elemType in ElemType.Get_2D()]
    Display.Plot_Tags(meshes2D[0], alpha=0.1, showId=True)

    # ----------------------------------------------
    # 3D CRACK
    # ----------------------------------------------

    line1 = Line(Point(L/4, L/2), Point(3*L/4, L/2), isOpen=openCrack)
    line2 = Line(line1.pt2, line1.pt2+[0,0.25,L])
    line3 = Line(line2.pt2, line1.pt1+[0,0.25,L], isOpen=openCrack)
    line4 = Line(line3.pt2, line1.pt1)
    crack1 = Points([Point(L/2,L/5,L),
                        Point(2*L/3,L/5,L),
                        Point(L,L/2,L, isOpen=True)], isOpen=True)

    cracks = [Contour([line1, line2, line3, line4], isOpen=openCrack), crack1]

    meshes3D = [DoMesh(3, elemType, cracks) for elemType in [ElemType.TETRA4, ElemType.TETRA10]]
    Display.Plot_Tags(meshes3D[0], alpha=0.1, showId=True)

    Display.plt.show()