"""Mesh optimization with the ZZ1 criterion for a bending part."""

from EasyFEA import (Display, Folder, Tic, plt, np,
                     Mesher, ElemType, Mesh,
                     Materials, Simulations,
                     Paraview_Interface,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Points
from EasyFEA.fem import Mesh, Calc_projector

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2

    # Options for plotting the results
    plotProj = True
    makeMovie = False
    makeParaview = False   
    
    treshold = 1/100 if dim == 2 else 0.04 # Target error for the optimization process
    iterMax = 20 # Maximum number of iterations
    coef = 1/10 # Scaling coefficient for the optimization process
    
    # Selecting the element type for the mesh
    if dim == 2:
        elemType = ElemType.TRI3 # TRI3, TRI6, TRI10, QUAD4, QUAD8
    else:
        elemType = ElemType.TETRA4 # TETRA4, TETRA10, HEXA8, HEXA20, PRISM6, PRISM15

    # Creating a folder to store the results
    folder = Folder.New_File(Folder.Join('Meshes', f'Optim{dim}D'), results=True)
    if not Folder.Exists(folder):
        Folder.os.makedirs(folder)

    # ----------------------------------------------
    # Meshing
    # ----------------------------------------------

    L = 120
    h = L * 2/3
    b = h
    r = h/(2+1e-2)
    e = (L - 2*r)/2

    meshSize = h/10

    pt1 = Point()
    pt2 = Point(e,0)
    pt3 = Point(e,r,r=r)
    pt4 = Point(L-e,r,r=r)
    pt5 = Point(L-e,0)
    pt6 = Point(L,0)

    pt7 = Point(L,h)
    pt8 = Point(L-e,h)
    pt9 = Point(L-e,h-r,r=r)
    pt10 = Point(e,h-r,r=r)
    pt11 = Point(e,h)
    pt12 = Point(0,h)

    points = Points([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9, pt10, pt11, pt12], meshSize)    
    inclusions = []

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used to generate the mesh."""
        if dim == 2:
            return Mesher().Mesh_2D(points, inclusions, elemType, [], [refineGeom])
        else:
            return Mesher().Mesh_Extrude(points, inclusions, [0,0,b], [], elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()

    # ----------------------------------------------
    # Material and Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.ElasticSimu(mesh, material)
    simu.rho = 8100*1e-9

    P = 800  # N
    lineLoad = P / h  # N/mm
    surfLoad = P / h / b  # N/mm2

    def DoSimu(refineGeom: str):

        simu.mesh = DoMesh(refineGeom)
        
        # get the nodes
        nodes_Fixed = simu.mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        nodes_Load = simu.mesh.Nodes_Conditions(lambda x,y,z: x == L)
        
        # do the simulation
        simu.Bc_Init()
        simu.add_dirichlet(nodes_Fixed, [0]*dim, simu.Get_dofs(), description="Fixed")
        simu.add_surfLoad(nodes_Load, [-surfLoad], ["y"])

        simu.Solve()

        simu.Save_Iter()

        return simu

    simu = Simulations.Mesh_Optim_ZZ1(DoSimu, folder, treshold, iterMax, 1/10)
    
    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    Display.Plot_Result(simu, "ZZ1_e", nodeValues=False, title="ZZ1", ncolors=11)

    if plotProj:
        
        simu.Set_Iter(0)
        mesh0 = simu.mesh
        u0 = np.reshape(simu.displacement, (mesh0.Nn, -1))

        simu.Set_Iter(1)
        mesh1 = simu.mesh        

        proj = Calc_projector(mesh0, mesh1)
        uProj = np.zeros((mesh1.Nn, dim), dtype=float)
        for d in range(dim):
            uProj[:,d] = proj @ u0[:,d]
        
        ax = Display.Plot_Result(mesh0, np.linalg.norm(u0, axis=1), plotMesh=True, title='u0')
        ax.plot(*mesh1.coord[:,:dim].T, ls='', marker='+', c='k', label='new nodes')
        ax.legend()
        Display.Plot_Result(mesh1, np.linalg.norm(uProj, axis=1), plotMesh=True, title='uProj')


    if makeParaview:
        Paraview_Interface.Make_Paraview(simu, folder, nodesField=["ZZ1_e"])
    
    if makeMovie:
        def func(plotter, n):

            simu.Set_Iter(n)

            pvi.Plot(simu, 'ZZ1_e', show_edges=True, edge_color='grey', plotter=plotter, clim=(0, 1), verticalColobar=False)
            # pvi.Plot_BoundaryConditions(simu, plotter=plotter)

            zz1 = simu._Calc_ZZ1()[0]

            plotter.add_title(f'ZZ1 = {zz1*100:.2f} %')

        pvi.Movie_func(func, len(simu.results), folder, f'lmt.gif')

    Tic.Plot_History(details=True)
    plt.show()