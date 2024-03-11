"""Mesh optimization with ZZ1 criterion."""

import Folder
import Display
from Geoms import *
import Materials
from Mesh import Mesh, Calc_projector
from Gmsh_Interface import Mesher, ElemType
import Simulations
from TicTac import Tic
import PyVista_Interface as pvi
import Paraview_Interface

import matplotlib.pyplot as plt

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------------------------
    # Meshing
    # --------------------------------------------------------------------------------------------

    L = 80
    h1 = L/4
    e1 = h1*.1
    h2 = (h1-e1)*.95
    e2 = (h1-e1-h2)/2
    r = h2/4
    l = L/2
    b = h1

    meshSize = r/3

    F = 1e-3 # 5g
    surfLoad = F/(h1-e1)/b # 1g

    pt1 = Point()
    pt2 = Point(L-h1)
    pt3 = pt2 + [e1,-e1]
    pt4 = Point(L,-e1)
    pt5 = pt4 + [0,h1]
    pt6 = Point(h1, h1-e1)
    pt7 = pt6 + [-e1,e1]
    pt8 = pt1 + [0,h1]

    points = Points([pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8], meshSize)

    p1 = Point(L/2-l/2,e2, r=r)
    p2 = Point(L/2-l/2+2*r,e2, r=r)
    p3 = Point(L/2-l/2+2*r,e2+r)
    p4 = p3.copy(); p4.symmetry((L/2, (h1-e1)/2), (1,0))
    p5 = p2.copy(); p5.symmetry((L/2, (h1-e1)/2), (1,0))
    p6 = p1.copy(); p6.symmetry((L/2, (h1-e1)/2), (1,0))
    p7 = p6.copy(); p7.symmetry((L/2, (h1-e1)/2), (0,1))
    p8 = p5.copy(); p8.symmetry((L/2, (h1-e1)/2), (0,1))
    p9 = p4.copy(); p9.symmetry((L/2, (h1-e1)/2), (0,1))
    p10 = p3.copy(); p10.symmetry((L/2, (h1-e1)/2), (0,1))
    p11 = p2.copy(); p11.symmetry((L/2, (h1-e1)/2), (0,1))
    p12 = p1.copy(); p12.symmetry((L/2, (h1-e1)/2), (0,1))

    inclusion = Points([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], meshSize, True)
    inclusions = [inclusion]

    def DoMesh(refineGeom=None) -> Mesh:
        """Function used to generate the mesh."""
        if dim == 2:
            return Mesher().Mesh_2D(points, inclusions, elemType, [], [refineGeom])
        else:
            return Mesher().Mesh_Extrude(points, inclusions, [0,0,b], [], elemType, [], [refineGeom])

    # Construct the initial mesh
    mesh = DoMesh()

    # --------------------------------------------------------------------------------------------
    # Material and Simulation
    # --------------------------------------------------------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, thickness=b)
    simu = Simulations.Displacement(mesh, material)
    simu.rho = 8100*1e-9

    P = 800  # N
    lineLoad = P / h1  # N/mm
    surfLoad = P / h1 / b  # N/mm2

    def DoSimu(refineGeom: str):

        simu.mesh = DoMesh(refineGeom)
        
        # get the nodes
        nodes_Fixed = simu.mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        nodes_Load = simu.mesh.Nodes_Conditions(lambda x,y,z: x == L)
        
        # do the simulation
        simu.Bc_Init()
        simu.add_dirichlet(nodes_Fixed, [0]*dim, simu.Get_directions(), description="Fixed")
        simu.add_surfLoad(nodes_Load, [-surfLoad], ["y"])

        simu.Solve()

        simu.Save_Iter()

        return simu

    simu = Simulations.MeshOptim_ZZ1(DoSimu, folder, treshold, iterMax, 1/10)
    
    # --------------------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------------------

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
        ax.plot(*mesh1.coordo[:,:dim].T, ls='', marker='+', c='k', label='new nodes')
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

        pvi.Movie_func(func, len(simu.results), folder, f'letterWeigher.gif')

    Tic.Plot_History(details=True)
    plt.show()