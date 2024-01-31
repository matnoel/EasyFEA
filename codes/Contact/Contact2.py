# Frictionless contact assumption
# WARNING : the assumption of small displacements is more than questionable for this simulation

import Display
from Interface_Gmsh import Interface_Gmsh, ElemType, Mesh
from Geom import Point, Domain, Circle, PointsList, Geom
import Materials
import Simulations

plt = Display.plt
np = Display.np

if __name__ == '__main__':

    Display.Clear()

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    dim = 2

    R = 10
    height = R
    meshSize = R/20
    thickness = R/3

    N = 30   

    inc = 2*R/N
    cx, cy = 1, 0

    # --------------------------------------------------------------------------------------------
    # Meshes
    # --------------------------------------------------------------------------------------------

    # slave mesh
    contour_slave = Domain(Point(-R/2,0), Point(R/2,height), meshSize)
    if dim == 2:
        mesh_slave = Interface_Gmsh().Mesh_2D(contour_slave, [], ElemType.QUAD4, isOrganised=True)
    else:
        mesh_slave = Interface_Gmsh().Mesh_3D(contour_slave, [], [0,0,-thickness], [4], ElemType.PRISM6, isOrganised=True)

    nodes_slave = mesh_slave.Get_list_groupElem(dim-1)[0].nodes
    nodes_y0 = mesh_slave.Nodes_Conditions(lambda x,y,z: y==0)

    # master mesh    
    r = R/2
    p0 = Point(-R/2, height, r=r)
    p1 = Point(R/2, height, r=r)
    p2 = Point(R/2, height+R)
    p3 = Point(-R/2, height+R)
    contour_master = PointsList([p0,p1,p2,p3])
    if dim == 2:
        contour_master.translate(-R, -2)
    else:
        contour_master.translate(-R, -2, 1)
    yMax = height+np.abs(r)
    if dim == 2:
        mesh_master = Interface_Gmsh().Mesh_2D(contour_master, [], ElemType.TRI3)
    else:    
        mesh_master = Interface_Gmsh().Mesh_3D(contour_master, [], [0,0,-thickness-2], [4], ElemType.PRISM6)

    # get master nodes
    nodes_master = mesh_master.Get_list_groupElem(dim-1)[0].nodes

    # plot meshes
    ax = Display.Plot_Mesh(mesh_master, alpha=0)
    Display.Plot_Mesh(mesh_slave, ax=ax, alpha=0)
    # add nodes interface
    ax.scatter(*mesh_slave.coordo[nodes_slave,:dim].T, label='slave nodes')
    ax.scatter(*mesh_master.coordo[nodes_master,:dim].T, label='master nodes')
    ax.legend()
    ax.set_title('Contact nodes')

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=True, thickness=thickness)
    simu = Simulations.Simu_Displacement(mesh_slave, material)

    list_mesh_master = [mesh_master]

    fig, ax, cb = Display.Plot_Result(simu, 'uy', deformFactor=1)

    for i in range(N):

        mesh_master = mesh_master.copy()
        mesh_master.translate(cx*inc, cy*inc)

        list_mesh_master.append(mesh_master)

        groupMaster = mesh_master.Get_list_groupElem(dim-1)[0]
        if dim == 3 and i == 0 and len(mesh_master.Get_list_groupElem(dim-1)) > 1:
            print(Display.Error(f"The {groupMaster.elemType.name} element group is used. In 3D, TETRA AND HEXA elements are recommended."))

        convergence=False

        coordo_old = simu.Results_displacement_matrix() + simu.mesh.coordo

        while not convergence:

            # apply new boundary conditions
            simu.Bc_Init()
            simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_directions())

            nodes, newU = simu.Get_contact(mesh_master, nodes_slave)

            if nodes.size > 0:        
                simu.add_dirichlet(nodes, [newU[:,0], newU[:,1]], ['x','y'])

            simu.Solve()

            # check if there is no new nodes in the master mesh
            oldSize = nodes.size
            nodes, _ = simu.Get_contact(mesh_master, nodes_slave)

            convergence = oldSize == nodes.size

        simu.Save_Iter()

        print(f"Eps max = {simu.Result('Strain').max()*100:3.2f} %")
        
        ax.clear()
        cb.remove()
        _,ax,cb = Display.Plot_Result(simu, 'uy', plotMesh=True, deformFactor=1, ax=ax)
        Display.Plot_Mesh(mesh_master, alpha=0, ax=ax)
        ax.set_title('uy')
        if dim == 3:
            Display._Axis_equal_3D(ax, np.concatenate((mesh_master.coordo, mesh_slave.coordo), 0))
        
        # # Plot arrows
        # if nodes.size >0:
        #     # get the nodes coordinates on the interface
        #     coordinates = groupMaster.Get_GaussCoordinates_e_p('mass').reshape(-1,3)
        #     ax.scatter(*coordinates[:,:dim].T)

        #     coordo_new = simu.Results_displacement_matrix() + simu.mesh.coordo
        #     ax.scatter(*coordo_old[nodes,:dim].T)
        #     incU = coordo_new - coordo_old
        #     [ax.arrow(*coordo_old[node, :dim], *incU[node,:dim],length_includes_head=True) for node in nodes]

        plt.pause(1e-12)
        
        pass

    # --------------------------------------------------------------------------------------------
    # PostProcessing
    # --------------------------------------------------------------------------------------------
    Display.Plot_Result(simu, 'Eyy', nodeValues=True)
    Display.Plot_Result(simu, 'ux')
    Display.Plot_Result(simu, 'uy')

    Simulations.Tic.Plot_History(details=True)

    print(simu)

    plt.show()