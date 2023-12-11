import Display
from Interface_Gmsh import Interface_Gmsh, ElemType, Mesh
from Mesh import Calc_projector
from Geom import Point, Domain, Line, Circle
import Materials
import Simulations
import Folder

import numpy as np

if __name__ == '__main__':

    Display.Clear()

    folder = Folder.New_File("PFM_Adapt", results=True)

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------
    dim = 2
    test = True
    optimMesh = False

    # geom
    L = 15e-3
    h = 30e-3
    thickness = 1 
    diam = 6e-3

    # material
    material = Materials.Elas_Isot(dim, 12e9, 0.3, False, thickness)

    # phase field
    gc = 1.4
    l0 = 0.12e-3
    split = "Miehe"
    regu = "AT2"
    tolConv = 1e-0
    maxIter = 200
    pfm = Materials.PhaseField_Model(material, split, regu, gc, l0)

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------
    if test:
        if optimMesh:
            clD = l0*4
            clC = l0
        else:
            clD = 0.25e-3 # l0*2
            clC = 0.12e-3 # l0
            clD = l0*3
            clC = l0
    else:        
        if optimMesh:
            clD = l0*4
            clC = l0/2
        else:
            clD = l0/2
            clC = l0/2

    contour = Domain(Point(), Point(L,h), clD)
    circle = Circle(Point(L/2, h/2), diam, clC)

    if optimMesh:
        refineZone = diam*1.5/2
        if split in ["Bourdin", "Amor"]:
            refineDomain = Domain(Point(0, h/2-refineZone), Point(L, h/2+refineZone), clC)
        else:
            refineDomain = Domain(Point(L/2-refineZone, 0), Point(L/2+refineZone, h), clC)
    else:
        refineDomain = None

    refineMesh = None

    def DoMesh(refineGeoms: list) -> Mesh:
        return Interface_Gmsh().Mesh_2D(contour, [circle], ElemType.TRI3, [], refineGeoms)

    mesh = DoMesh([refineDomain, refineMesh])

    # --------------------------------------------------------------------------------------------
    # Simulation
    # --------------------------------------------------------------------------------------------
    # loading
    treshold = 0.6
    u_max = 25e-6 # 25e-6, 35e-6
    uinc0 = 8e-8; uinc1 = 2e-8
    listInc = [uinc0, uinc1]
    listTresh = [0, treshold]
    listOption = ["damage"]*len(listTresh)

    simu = Simulations.Simu_PhaseField(mesh, pfm)
    simu.Results_Set_Bc_Summary(u_max, listInc, listTresh, listOption)        

    def Loading(ud: float):
        """Boundary conditions"""

        mesh = simu.mesh
        
        # Get Nodes
        nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)            
        nodes_x0y0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
        nodes_y0z0 = mesh.Nodes_Conditions(lambda x,y,z: (y==0) & (z==0))

        simu.Bc_Init()
        simu.add_dirichlet(nodes_lower, [0], ["y"])
        simu.add_dirichlet(nodes_x0y0, [0], ["x"])
        simu.add_dirichlet(nodes_upper, [-ud], ["y"])
        if dim == 3:
            simu.add_dirichlet(nodes_y0z0, [0], ["z"])

    ud = 0
    iter = 0
    nDetect = 0
    displacement = []
    load = []
    damageNodes = []
    while ud <= u_max:

        iter += 1
        ud += uinc0 if simu.damage.max() < treshold else uinc1

        Loading(ud)

        u, d, Kglob, convergence = simu.Solve(tolConv, maxIter)
        simu.Save_Iter()

        # stop if the simulation does not converge
        if not convergence: break   

        nodes_edges = mesh.Nodes_Tags(["L0","L1","L2","L3"])
        nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)
        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ["y"])

        f = np.sum(Kglob[dofsY_upper, :] @ u)

        simu.Results_Set_Iteration_Summary(iter, ud*1e6, "µm", ud/u_max, True)

        # Detection if the edges has been touched
        if np.any(d[nodes_edges] >= 0.98):
            nDetect += 1
            if nDetect == 10:                    
                break

        displacement.append(ud)
        load.append(f)

        # --------------------------------------------------------------------------------------------
        # update the mesh
        # --------------------------------------------------------------------------------------------

        filter = d >= 1e-2

        if np.any(filter):
            
            # x = simu.mesh.coordo[:,0]
            # d = -(x * (x-L))
            # d = d/d.max()
            # d = d[]
            # Display.Plot_Result(simu, d)

            meshSize_n = (clC-clD) * d + clD

            # Display.Plot_Result(simu, meshSize_n)

            refineMesh = Interface_Gmsh().Create_posFile(simu.mesh.coordo[filter], meshSize_n[filter], folder)

            newMesh = DoMesh([refineDomain, refineMesh])

            if newMesh.Nn > simu.mesh.Nn:

                oldMesh = simu.mesh
                oldU = simu.displacement
                oldD = simu.damage

                nodesDetected = simu.mesh.nodes[filter]

                ax = Display.Plot_Mesh(newMesh, alpha=0, edgecolor='red')
                Display.Plot_Mesh(simu.mesh, alpha=0, ax=ax)
                Display.Plot_Nodes(simu.mesh, nodesDetected, ax=ax, c='blue', marker='+')            

                newU = np.zeros(newMesh.Nn*dim)
                newD = np.zeros(newMesh.Nn)

                proj = Calc_projector(oldMesh, newMesh)
                Simulations.Tic.Plot_History()

                simu.mesh = newMesh
                newDofs = simu.Bc_dofs_nodes(newMesh.nodes, ["x"], "displacement")
                oldDofs = simu.Bc_dofs_nodes(oldMesh.nodes, ["x"], "displacement")
                for d in range(dim):                
                    newU[newDofs+d] = proj @ oldU[oldDofs+d]
                newD = proj @ oldD

                simu.set_u_n("displacement", newU)
                simu.set_u_n("damage", newD)

                Display.Plot_Result(oldMesh, oldU.reshape(-1,dim)[:,0])
                Display.Plot_Result(newMesh, newU.reshape(-1,dim)[:,0])


                Loading(ud)
                u, d, Kglob, convergence = simu.Solve(tolConv, maxIter)
                simu.Save_Iter()

                Display.Plot_Result(simu, "ux")

                pass


























    # if updateMesh:

    #     meshSize_n = (clC-clD) * d + clD

    #     # Display.Plot_Result(simu, meshSize_n)

    #     refineDomain = Interface_Gmsh().Create_posFile(simu.mesh.coordo, meshSize_n, folder)

    #     newMesh = DoMesh(refineDomain)

    #     if newMesh.Nn > simu.mesh.Nn:
            
    #         oldNodes = simu.mesh.Nodes_Conditions(lambda x,y,z: (x<L/2)&(y==L/2))
    #         oldNodes = np.unique(oldNodes)
    #         # oldNodes = oldNodes[np.argsort(simu.mesh.coordo[oldNodes,0])]

    #         newNodes = newMesh.Nodes_Conditions(lambda x,y,z: (x<L/2)&(y==L/2))
    #         newNodes = np.unique(newNodes)
    #         # newNodes = newNodes[np.argsort(newMesh.coordo[newNodes,0])]

    #         assert len(oldNodes) == len(newNodes)
            
    #         # axOld = Display.Plot_Mesh(simu.mesh, alpha=0)
    #         # axNew = Display.Plot_Mesh(newMesh, alpha=0)
    #         # for n in range(len(newNodes)):
    #         #     if n > len(newNodes)//2:
    #         #         c="black"
    #         #     else:
    #         #         c="red"
    #         #     Display.Plot_Nodes(simu.mesh, [oldNodes[n]], True, ax=axOld, c=c)
    #         #     # Display.Plot_Nodes(newMesh, [newNodes[n]], True, ax=axNew, c=c)
    #         #     pass
            
    #         # for n in range(len(newNodes)):
    #         #     plt.close("all")
    #         #     Display.Plot_Nodes(simu.mesh, [oldNodes[n]], True)
    #         #     Display.Plot_Nodes(newMesh, [newNodes[n]], True)
    #         #     pass                    

    #         proj = Calc_projector(simu.mesh, newMesh)
    #         proj = proj.tolil()
    #         proj[newNodes, :] = 0                    
    #         proj[newNodes, oldNodes] = 1

    #         newU = np.zeros((newMesh.Nn, 2))

    #         for i in range(dim):
    #             newU[:,i] = proj @ u.reshape(-1,2)[:,i]

    #         newD = proj @ d

    #         plt.close("all")
    #         # Display.Plot_Result(simu.mesh, d, plotMesh=True)
    #         # Display.Plot_Result(newMesh, newD, plotMesh=True)                    

    #         Display.Plot_Result(simu.mesh, u.reshape(-1,2)[:,0])
    #         Display.Plot_Result(newMesh, newU[:,0])

    #         plt.pause(1e-12)
    #         # Tic.Plot_History()

    #         simu.mesh = newMesh
    #         mesh = newMesh
    #         simu.set_u_n("displacement", newU.reshape(-1))
    #         simu.set_u_n("damage", newD.reshape(-1))