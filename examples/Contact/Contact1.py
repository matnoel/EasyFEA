"""Performing a 'Hertz contact problem' with the assumption of frictionless contact.
The master mesh is considered non-deformable.
TODO: Compare results with analytical values.
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

    R = 10
    height = R
    meshSize = R/20
    thickness = R/3

    N = 30

    inc = 1e-0/N
    cx, cy = 0, -1

    # ----------------------------------------------
    # Meshes
    # ----------------------------------------------

    # slave mesh
    contour_slave = Domain(Point(-R/2,0), Point(R/2,height), meshSize)
    if dim == 2:
        mesh_slave = Mesher().Mesh_2D(contour_slave, [], ElemType.QUAD4, isOrganised=True)
    else:
        mesh_slave = Mesher().Mesh_Extrude(contour_slave, [], [0,0,-thickness], [4], ElemType.HEXA8, isOrganised=True)

    # nodes_slave = mesh_slave.Get_list_groupElem(dim-1)[0].nodes
    nodes_slave = mesh_slave.Nodes_Conditions(lambda x,y,z: y==height)
    nodes_y0 = mesh_slave.Nodes_Conditions(lambda x,y,z: y==0)

    # master mesh
    r = R/2
    p0 = Point(-R/2, height, r=r)
    p1 = Point(R/2, height, r=r)
    p2 = Point(R/2, height+R)
    p3 = Point(-R/2, height+R)
    contour_master = Points([p0,p1,p2,p3])

    yMax = height+np.abs(r)
    if dim == 2:
        mesh_master = Mesher().Mesh_2D(contour_master, [], ElemType.TRI3)
    else:    
        mesh_master = Mesher().Mesh_Extrude(contour_master, [], [0,0,-thickness-2], [4], ElemType.TETRA4)
        groupMaster = mesh_master.Get_list_groupElem(dim-1)[0]
        if len(mesh_master.Get_list_groupElem(dim-1)) > 1:
            Display.myPrintError(f"The {groupMaster.elemType.name} element group is used. In 3D, TETRA AND HEXA elements are recommended.")
    mesh_master.translate(dz=-(mesh_master.center[2]-mesh_slave.center[2]))

    # Display.Plot_Tags(mesh_master, alpha=0.1, showId=True)

    # get master nodes
    # nodes_master = mesh_master.Get_list_groupElem(dim-1)[0].nodes
    if dim == 2:
        nodes_master = mesh_master.Nodes_Tags(['L0','L1'])
    else:
        nodes_master = mesh_master.Nodes_Tags(['S1','S2'])

    # # plot meshes
    # ax = Display.Plot_Mesh(mesh_master, alpha=0)
    # Display.Plot_Mesh(mesh_slave, ax=ax, alpha=0)
    # # add nodes interface
    # ax.scatter(*mesh_slave.coordo[nodes_slave,:dim].T, label='slave nodes')
    # ax.scatter(*mesh_master.coordo[nodes_master,:dim].T, label='master nodes')
    # ax.legend()
    # ax.set_title('Contact nodes')

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=True, thickness=thickness)
    simu = Simulations.ElasticSimu(mesh_slave, material)

    list_mesh_master = [mesh_master]

    if pltIter:
        ax = Display.Plot_Result(simu, result, deformFactor=1)

    for i in range(N):

        mesh_master = mesh_master.copy()
        mesh_master.translate(cx*inc, cy*inc)

        list_mesh_master.append(mesh_master)

        convergence=False

        coordo_old = simu.Results_displacement_matrix() + simu.mesh.coord

        while not convergence:

            # apply new boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_dofs())

            nodes, newU = simu.Get_contact(mesh_master, nodes_slave, nodes_master)

            if nodes.size > 0:
                simu.add_dirichlet(nodes, [newU[:,0], newU[:,1]], ['x','y'])

            simu.Solve()

            # check if there is no new nodes in the master mesh
            oldSize = nodes.size
            nodes, __ = simu.Get_contact(mesh_master, nodes_slave, nodes_master)
            convergence = oldSize == nodes.size

        simu.Save_Iter()

        print(f"Eps max = {simu.Result('Strain').max()*100:3.2f} %")
        
        if pltIter:
            Display.Plot_Result(simu, result, plotMesh=True, deformFactor=1, ax=ax)
            Display.Plot_Mesh(mesh_master, alpha=0, ax=ax)
            ax.set_title(result)
            if dim == 3:
                Display._Axis_equal_3D(ax, np.concatenate((mesh_master.coord, mesh_slave.coord), 0))
        
            # # Plot arrows
            # if nodes.size >0:
            #     # get the nodes coordinates on the interface
            #     coordinates = groupMaster.Get_GaussCoordinates_e_p('mass').reshape(-1,3)
            #     ax.scatter(*coordinates[:,:dim].T)

            #     coordo_new = simu.Results_displacement_matrix() + simu.mesh.coordo
            #     ax.scatter(*coordo_old[nodes,:dim].T)
            #     incU = coordo_new - coordo_oldq
            #     [ax.arrow(*coordo_old[node, :dim], *incU[node,:dim],length_includes_head=True) for node in nodes]

            plt.pause(1e-12)

    print(simu)

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    Display.Plot_Result(simu, 'Eyy', nodeValues=True)
    Display.Plot_Result(simu, 'ux')
    Display.Plot_Result(simu, 'uy')

    Tic.Plot_History(details=True)

    if makeMovie:

        def DoAnim(plotter, n):
            simu.Set_Iter(n)
            pvi.Plot(simu, "Svm", 1, style='surface', color='k', plotter=plotter, n_colors=10, show_grid=True)
            pvi.Plot(list_mesh_master[n], plotter=plotter, show_edges=True, opacity=0.2)

        pvi.Movie_func(DoAnim, N, folder=folder, filename='Contact1.gif')
    
    # TODO bending 3 pts

    plt.show()