"""Hydraulic dam subjected to water pressure and its own weight."""

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Points

if __name__ == '__main__':

    Display.Clear()

    # Define dimension and mesh size parameters
    dim = 2
    N = 50 if dim == 2 else 10

    coef = 1e6
    E = 15000*coef  # Pa (Young's modulus)
    v = 0.25          # Poisson's ratio

    g = 9.81   # m/s^2 (acceleration due to gravity)
    ro = 2400  # kg/m^3 (density)
    w = 1000   # kg/m^3 (density)

    h = 180  # m (thickness)
    thickness = 2*h

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    pt1 = Point()
    pt2 = Point(x=h)
    pt3 = Point(y=h)
    contour = Points([pt1, pt2, pt3], h/N)

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.TRI6)
        print(f"err area = {np.abs(mesh.area - h**2/2):.3e}")
    elif dim == 3:
        mesh = Mesher().Mesh_Extrude(contour, [], [0, 0, -thickness], [3], ElemType.PRISM15)
        print(f"error volume = {np.abs(mesh.volume - h**2/2 * thickness):.3e}")

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesY0 = mesh.Nodes_Conditions(lambda x, y, z: y == 0)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.Elas_Isot(dim, E, v, planeStress=False, thickness=thickness)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodesY0, [0]*dim, simu.Get_dofs())
    simu.add_surfLoad(nodesX0, [lambda x, y, z: w*g*(h - y)], ["x"], description="[w*g*(h-y)]")
    simu.add_volumeLoad(mesh.nodes, [-ro*g], ["y"], description="[-ro*g]")

    sol = simu.Solve()
    simu.Save_Iter()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Plot_Tags(mesh)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, h/10/np.abs(sol.max()))
    Display.Plot_Result(simu, "Svm", nodeValues=True, coef=1/coef, ncolors=20)

    Tic.Plot_History(details=False)

    plt.show()