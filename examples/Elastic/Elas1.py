"""A cantilever beam undergoing bending deformation."""

from EasyFEA import (Display, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain

if __name__ == '__main__':

    Display.Clear()

    # Define dimension and mesh size parameters
    dim = 2
    N = 20 if dim == 2 else 10

    # Define material properties
    E = 210000  # MPa (Young's modulus)
    v = 0.3     # Poisson's ratio
    coef = 1

    L = 120 # mm
    h = 13
    I = h**4/12 # mm4
    load = 800 # N

    W_an = 2*load**2*L/E/h**2 * (L**2/h**2 + (1+v)*3/5) # mJ
    uy_an = load*L**3/(3*E*I)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    meshSize = h/N

    domain = Domain(Point(), Point(L,h), meshSize)

    if dim == 2:
        mesh = Mesher().Mesh_2D(domain, [], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = Mesher().Mesh_Extrude(domain, [], [0,0,-h], [4], ElemType.HEXA8, isOrganised=True)

    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    material = Materials.Elas_Isot(dim, E, v, planeStress=True, thickness=h)
    simu = Simulations.ElasticSimu(mesh, material)

    simu.add_dirichlet(nodesX0, [0]*dim, simu.Get_dofs())
    simu.add_surfLoad(nodesXL, [-load/h**2], ["y"])

    sol = simu.Solve()
    simu.Save_Iter()
    
    uy_num = - simu.Result('uy').min()
    W_num = simu._Calc_Psi_Elas()

    # ----------------------------------------------
    # Results
    # ----------------------------------------------
    print(simu)

    Display.Section('Result')

    print(f"err W : {np.abs(W_an-W_num)/W_an*100:.2f} %")

    print(f"err uy : {np.abs(uy_an-uy_num)/uy_an*100:.2f} %")


    Display.Plot_Tags(mesh)
    Display.Plot_BoundaryConditions(simu)
    Display.Plot_Mesh(simu, h/2/np.abs(sol).max())
    Display.Plot_Result(simu, "uy", nodeValues=True, coef=1/coef, ncolors=20)

    Tic.Plot_History(details=False)

    plt.show()