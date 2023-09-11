# WARNING : the assumption of small displacements is more than questionable for this simulation

import Display
from Interface_Gmsh import Interface_Gmsh, ElemType, Mesh
from Geom import Point, Domain, Circle, PointsList, Geom
from Mesh import Get_new_mesh
import Materials
import Simulations

Display.Clear()
plt = Display.plt
np = Display.np

# ----------------------------------------------
# Configuration
# ----------------------------------------------
dim = 2

R = 10
height = R
meshSize = R/20
thickness = R/3

N = 30

displacements = np.ones(N) * 1e-0/N
cx, cy = 0, -1
dec = [0, 0]

# displacements = np.ones(N) * 2*R/N
# cx, cy = 1, 0
# dec = [R, 2]

# dep = [cx, cy] * ud

# ----------------------------------------------
# Meshes
# ----------------------------------------------

# slave mesh
contour_slave = Domain(Point(-R/2,0), Point(R/2,height), meshSize)
if dim == 2:
    mesh_slave = Interface_Gmsh().Mesh_2D(contour_slave, [], ElemType.TRI3, isOrganised=True)
else:
    mesh_slave = Interface_Gmsh().Mesh_3D(contour_slave, [], [0,0,-thickness], 4, ElemType.PRISM6)

nodes_slave = mesh_slave.Get_list_groupElem(dim-1)[0].nodes
nodes_y0 = mesh_slave.Nodes_Conditions(lambda x,y,z: y==0)

# master mesh
if dim == 3: dec.append(-1)    
r = R/2
p0 = Point(-R/2, height, r=r) - dec 
p1 = Point(R/2, height, r=r) - dec 
p2 = Point(R/2, height+R) - dec 
p3 = Point(-R/2, height+R) - dec 
contour_master = PointsList([p0,p1,p2,p3], meshSize*2)
yMax = height+np.abs(r)
if dim == 2:
    mesh_master = Interface_Gmsh().Mesh_2D(contour_master, [], ElemType.TRI3)
else:    
    mesh_master = Interface_Gmsh().Mesh_3D(contour_master, [], [0,0,-thickness-2], 4, ElemType.PRISM6)

# get nodes and elements
groupElem = mesh_master.Get_list_groupElem(dim-1)[0]
nodes_master = groupElem.Get_Nodes_Conditions(lambda x,y,z: y <= yMax)
elements_master = groupElem.Get_Elements_Nodes(nodes_master)

# plot meshes
ax = Display.Plot_Mesh(mesh_master)
Display.Plot_Mesh(mesh_slave, ax=ax)
# add nodes interface
Display.Plot_Nodes(mesh_master, nodes_master, ax=ax)
Display.Plot_Nodes(mesh_slave, nodes_slave, ax=ax)
ax.set_title('Contact nodes')

# ----------------------------------------------
# Simulation
# ----------------------------------------------
material = Materials.Elas_Isot(dim, E=210000, v=0.3, planeStress=True, thickness=thickness)
simu = Simulations.Simu_Displacement(mesh_slave, material)

nodesInterface = []
def nodesInMaster():
    """Function to get """

    # update nodes coordinates
    newCoordo = simu.Results_displacement_matrix() + simu.mesh.coordo
    # check nodes in master mesh
    idx = mesh_master.groupElem.Get_Mapping(newCoordo[nodes_slave])[0]
    idx = np.unique(idx)

    if idx.size > 0:
        nodes = nodes_slave[idx]
        # add new nodes
        [nodesInterface.append(n) for n in nodes if n not in nodesInterface]
    else:
        nodes = idx.copy()

    return nodes, newCoordo

list_mesh_master = [mesh_master]

fig, ax, cb = Display.Plot_Result(simu, 'uy', deformation=True, factorDef=1)

for i, ud in enumerate(displacements):

    # create the new mesh
    displacementMatrix = np.zeros((mesh_master.Nn, 3))    
    displacementMatrix[:,0] = cx * ud
    displacementMatrix[:,1] = cy * ud
    mesh_master = Get_new_mesh(mesh_master, displacementMatrix)
    list_mesh_master.append(mesh_master)

    convergence=False
    nodesInterface = []

    while not convergence:

        # apply new boundary conditions
        simu.Bc_Init()
        simu.add_dirichlet(nodes_y0, [0]*dim, simu.Get_directions())

        # detect if there is slave nodes in the interface
        nodes, newCoordo = nodesInMaster()
        
        if nodes.size > 0:
            # slave nodes have been detected in the master mesh

            # get the elemGroup on the interface
            groupElem = mesh_master.Get_list_groupElem(dim-1)[0]        
            gaussCoordo_e_p = groupElem.Get_GaussCoordinates_e_p('mass', elements_master)
            
            # empty new displacement
            newU = []
            # for each nodes in master mesh we will detects the shortest displacement vector to the interface
            for node in nodes:
                # vectors between the interface coordinates and the detected node
                vi_e_pg  = gaussCoordo_e_p - newCoordo[node]
                # distance between the interface coordinates and the detected node
                d_e_pg = np.linalg.norm(vi_e_pg, axis=2)
                e, p = np.where(d_e_pg == d_e_pg.min())
                # retrieves the nearest coordinate
                closeCoordo = np.reshape(gaussCoordo_e_p[e[0],p[0]], -1)
                elem = elements_master[e[0]]
                # normal vector
                if dim == 2:
                    normal_vect = - groupElem.sysCoord_e[elem,:,1]
                else:                    
                    normal_vect = groupElem.sysCoord_e[elem,:,2]                    
                # distance to project the node to the element
                d = np.abs((newCoordo[node] - closeCoordo) @ normal_vect)
                # vector to the interface
                u = d * normal_vect
                newU.append(u)

            # Apply the displacement to meet the interface 
            oldU = simu.Results_displacement_matrix()[nodes]
            newU = np.array(newU) + oldU
            simu.add_dirichlet(nodes, [newU[:,0], newU[:,1]], ['x','y'])

        simu.Solve()

        # check if there is no new nodes in the master mesh
        old = len(nodesInterface)
        nodes, _ = nodesInMaster()

        convergence = old == len(nodesInterface)

        if not convergence:
            pass

    simu.Save_Iter()

    print(f"Eps max = {simu.Get_Result('Strain').max()*100:3.2f} %")
    
    # ax.clear()
    cb.remove()
    _,ax,cb = Display.Plot_Result(simu, 'uy', plotMesh=True, deformation=True, factorDef=1, ax=ax)
    Display.Plot_Mesh(mesh_master, alpha=0, ax=ax)
    ax.set_title('uy')
    if dim == 3:
        Display._ScaleChange(ax, np.concatenate((mesh_master.coordo, mesh_slave.coordo), 0))

    # nodes, newCoordo = nodesInMaster()
    # if nodes.size >0:
    #     # get the nodes coordinates on the interface
    #     coordinates = gaussCoordo_e_p.reshape(-1,3)
    #     if dim == 2:            
    #         ax.scatter(coordinates[:,0], coordinates[:,1])
    #         [ax.arrow(*mesh_slave.coordo[node, :2], newU[n,0], newU[n,1],length_includes_head=True) for n, node in enumerate(nodes)]
    #         ax.scatter(mesh_slave.coordo[nodes, 0], mesh_slave.coordo[nodes, 1])
    #     else:
    #         ax.scatter(coordinates[:,0], coordinates[:,1], coordinates[:,2])
    #         ax.scatter(mesh_slave.coordo[nodes, 0], mesh_slave.coordo[nodes, 1], mesh_slave.coordo[nodes, 2])

    # ax.set_xlim(xmin=-R/2, xmax=R/2)
    # ax.set_ylim(ymin=height-ud-height/10, ymax=height-ud+height/10)

    plt.pause(1e-12)
    
    pass

# ----------------------------------------------
# PostProcessing
# ----------------------------------------------
Display.Plot_Result(simu, 'Eyy', nodeValues=True)

Display.Plot_Result(simu, 'ux')
Display.Plot_Result(simu, 'uy')

# import Folder
# import PostProcessing
# folder = Folder.New_File('Contact', results=True)
# PostProcessing.Make_Paraview(folder, simu)
# # TODO how to plot the two meshes ?

print(simu)

plt.show()