"""Plate with a hole subjected to uniform tensile loading."""

from EasyFEA import (Display,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points, Domain, Circle

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------

    dim = 3
    isSymmetric = True

    a = 10
    l = 50
    h = 20
    meshSize = h/10
    thickness = 1

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    if isSymmetric:
        p0 = Point(0, 0, r=-a)
        p1 = Point(l, 0)
        p2 = Point(l, h)
        p3 = Point(0, h)
        contour = Points([p0, p1, p2, p3], meshSize)
        inclusions = []
    else:
        p0 = Point(-l, -h)
        p1 = Point(l, h)
        contour = Domain(p0, p1, meshSize)
        inclusions = [Circle(Point(), 2*a, meshSize, isHollow=True)]

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, inclusions, elemType=ElemType.TRI3)
    else:
        mesh = Mesher().Mesh_Extrude(contour, inclusions, [0,0,thickness], [4], ElemType.PRISM6)

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=True, thickness=thickness)
    simu = Simulations.ElasticSimu(mesh, material)

    if isSymmetric:
        nodes_x0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
        nodes_xl = mesh.Nodes_Conditions(lambda x,y,z: x == l)
        simu.add_dirichlet(nodes_x0, [0], ['x'])
        simu.add_dirichlet(nodes_y0, [0], ['y'])
        simu.add_surfLoad(nodes_xl, [800/20], ['x'])
    else:
        nodes_pl = mesh.Nodes_Conditions(lambda x,y,z: x == l)
        nodes_ml = mesh.Nodes_Conditions(lambda x,y,z: x == -l)
        nodes_y0 = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
        simu.add_dirichlet(nodes_y0, [0], ['y'])
        simu.add_surfLoad(nodes_pl, [800/20], ['x'])
        simu.add_surfLoad(nodes_ml, [-800/20], ['x'])

    simu.Solve()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    Display.Plot_Mesh(simu)
    Display.Plot_BoundaryConditions(simu)

    Display.Plot_Result(simu, 'ux', ncolors=10, nodeValues=True)
    Display.Plot_Result(simu, 'uy', ncolors=10, nodeValues=True)
    Display.Plot_Result(simu, 'Svm', ncolors=10, nodeValues=True)

    print(simu)

    Display.plt.show()