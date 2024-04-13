"""Bending bracket component."""

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points, Circle, Domain

if __name__ == '__main__':

    Display.Clear()

    # Define dimension and mesh size parameters
    dim = 3
    N = 20 if dim == 2 else 10

    # Define material properties
    E = 210000  # MPa (Young's modulus)
    v = 0.3     # Poisson's ratio
    coef = 1

    L = 120 # mm
    h = L*0.3
    load = 800

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    # Define points and crack geometry for the mesh
    pt1 = Point(isOpen=True, r=-10)
    pt2 = Point(x=L)
    pt3 = Point(x=L, y=h)
    pt4 = Point(x=h, y=h, r=10)
    pt5 = Point(x=h, y=L)
    pt6 = Point(y=L)
    pt7 = Point(x=h, y=h)
    contour = Points([pt1, pt2, pt3, pt4, pt5, pt6], h/N)

    inclusions = [Circle(Point(x=h/2, y=h*(i + 1)), h/4, meshSize=h/N, isHollow=True) for i in range(3)]
    inclusions.extend([Domain(Point(x=h, y=h/2 - h*0.1), Point(x=h*2.1, y=h/2 + h*0.1), isHollow=False, meshSize=h/N)])

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, inclusions, ElemType.TRI3)
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(contour, inclusions, [0, 0, -h], [4], elemType=ElemType.TETRA4)

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.Elas_Isot(dim, E, v, planeStress=True, thickness=h)
    simu = Simulations.ElasticSimu(mesh, material)

    if dim == 2:
        simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
        simu.add_lineLoad(nodesXL, [-800/h], ["y"])
    else:
        simu.add_dirichlet(nodesX0, [0, 0, 0], ["x", "y", "z"])
        simu.add_surfLoad(nodesXL, [-800/(h*h)], ["y"])

    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Plot_Tags(mesh)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, h/2/np.abs(sol).max())
    Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1/coef, ncolors=20)

    Tic.Plot_History(details=False)

    plt.show()