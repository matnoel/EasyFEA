"""Performing a contact problem simulation with the assumption of frictionless contact.
The master mesh is presumed to be non-deformable.
WARNING: The assumption of small displacements is highly questionable for this simulation.
"""

from EasyFEA import (Display, Folder, Tic, plt, np,
                     Mesher, ElemType,
                     Materials, Simulations,
                     PyVista_Interface as pvi)
from EasyFEA.Geoms import Point, Domain, Points

folder = Folder.New_File('Contact', results=True)

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    dim = 2
    pltIter = True; result = 'uy'
    makeMovie = False

    N = 20

    e = 10
    L = 3*e
    t = 1
    h = 10
    r = 3

    thickness = e/2

    mS = t/5 if dim == 2 else t/2

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    p1 = Point(-L/2-e)
    p2 = Point(-L/2, r=r)
    p3 = Point(-e/2, h-t, r=r)
    p4 = Point(e/2, h-t, r=r)
    p5 = Point(L/2, r=r)
    p6 = Point(L/2+e)

    line1 = Points([p1,p2,p3,p4,p5,p6])
    line2 = line1.Copy(); line2.Translate(dy=t)

    points = line1.points
    points.extend(line2.points[::-1])

    contour = Points(points, mS)

    enca = Domain(p1 - [e, 5*t], p6 + [e])

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, elemType=ElemType.TRI3)
        master_mesh = Mesher().Mesh_2D(enca, elemType=ElemType.QUAD4, isOrganised=True)
    else:
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,thickness], [thickness//mS])
        master_mesh = Mesher().Mesh_Extrude(enca, [], [0,0,4*thickness], [thickness//mS], elemType=ElemType.HEXA8, isOrganised=True)

        mesh.translate(dz=-thickness/2)
        master_mesh.translate(dz=-4*thickness/2)

    nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)

    # Display.Plot_Tags(mesh, True, alpha=0.1)
    if dim == 2:
        nodes_slave = mesh.Nodes_Tags([f'L{i}' for i in range(9)])
    else:
        nodes_slave = mesh.Nodes_Tags([f'S{i+1}' for i in range(9)])
    
    # nodes_slave = mesh.Get_list_groupElem(1)[0].nodes
        
    nodes_master = master_mesh.Nodes_Conditions(lambda x,y,z: y==0)

    # ax = Display.Plot_Mesh(mesh)
    # Display.Plot_Mesh(master_mesh, alpha=0.1, ax=ax)

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------

    material = Materials.Elas_Isot(dim, planeStress=True, thickness=thickness)

    simu = Simulations.ElasticSimu(mesh, material)

    displacements = np.linspace(0, t*2, N)    

    if pltIter:
        ax = Display.Plot_Result(simu, result, 1, plotMesh=True)

    for d, du in enumerate(displacements):

        simu.Bc_Init()
        simu.add_dirichlet(nodes_upper, [0,-du], ['x','y'])        
        simu.Solve()
        nodes, newU = simu.Get_contact(master_mesh, nodes_slave, nodes_master)
        if len(nodes) > 0:
            # simu.add_dirichlet(nodes, [newU[:,0],newU[:,1]], ['x','y'])
            simu.add_dirichlet(nodes, [newU[:,1]], ['y'])
            simu.Solve()            

        simu.Save_Iter()

        if pltIter:
            ax = Display.Plot_Result(simu, result, 1, plotMesh=False, ax=ax)
            Display.Plot_Mesh(master_mesh, ax=ax, title='', alpha=0)
            ax.set_title(result)

            plt.pause(1e-12)

    print(simu)

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------        

    ax = Display.Plot_Mesh(mesh)
    Display.Plot_Mesh(master_mesh, alpha=0.1, ax=ax)
    Display.Plot_BoundaryConditions(simu, ax=ax)
    ax.set_title('')

    if makeMovie:

        def DoAnim(plotter, n):
            simu.Set_Iter(n)
            pvi.Plot(simu, "Svm", 1, style='surface', color='k', plotter=plotter,
                 show_grid=True, verticalColobar=False)
            pvi.Plot(master_mesh, plotter=plotter, show_edges=True, opacity=.8)

        pvi.Movie_func(DoAnim, N, folder=folder, filename='Contact3.mp4')

    plt.show()