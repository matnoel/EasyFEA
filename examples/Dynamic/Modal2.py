"""Modal analysis of a structure. Currently, only the structure is available due to the slow performance of `scipy.sparse.linalg.eigs`, and the cause is not yet determined.
"""
# TODO: Use https://gitlab.com/slepc/slepc

from EasyFEA import (Display, Folder, np,
                     Mesher, ElemType, Mesh,
                     Materials, Simulations)

from scipy.sparse import linalg, eye
from scipy.linalg import eig

def Construct_struct(L: float,e: float,t: float, meshSize: float = 0.0, openGmsh=False, verbosity=False) -> Mesh:

    mesher = Mesher()

    h = L-e-t

    factory = mesher._factory

    # create the pilars
    pilar1 = [(3, factory.addBox(0,0,0,e,e,h))]
    pilar2 = factory.copy(pilar1); factory.translate(pilar2, L-e, 0, 0)
    pilar3 = factory.copy(pilar1); factory.translate(pilar3, L-e, L-e, 0)
    pilar4 = factory.copy(pilar1); factory.translate(pilar4, 0, L-e, 0)
    pilars = factory.getEntities(3)

    # creates the plate
    plate = [(3, factory.addBox(0,0,h,L,L,t))]

    # creates the table (pilars + plate)
    table, __ = factory.fragment(plate, pilars)

    # creates the cuve (Empty Box)
    box = [(3, factory.addBox(0,0,L-e,L,L,L))]    
    inc = [(3, factory.addBox(e,e,L,L-2*e,L-2*e,L-2*e))]
    cuve, __ = factory.cut(box, inc)

    # creates the structure (table + cuve)
    struct, __ = factory.fragment(table, cuve)

    if meshSize > 0:
        mesher.Set_meshSize(meshSize)
    
    mesher._Set_PhysicalGroups()

    mesher._Meshing(3, 'TETRA4')
    
    mesh = mesher._Construct_Mesh()

    return mesh    

if __name__ == '__main__':

    Display.Clear()

    folder = Folder.Get_Path(__file__)

    isFixed = True

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
        
    L = 21 # m
    e = 1
    t = 0.5    

    mesh = Construct_struct(L,e,t,0,False,False)

    nodes_pilars = mesh.Nodes_Tags(['V0', 'V1', 'V2', 'V3'])
    elems_pilars = mesh.Elements_Tags(['V0', 'V1', 'V2', 'V3'])

    nodes_plate = mesh.Nodes_Tags(['V4'])
    nodes_cuve = mesh.Nodes_Tags(['V5'])
    nodesZ0 = mesh.Nodes_Conditions(lambda x,y,z: z==0)
    nodesSupZ0 = mesh.Nodes_Conditions(lambda x,y,z: z>0)

    ax = Display.Plot_Nodes(mesh, nodes_pilars, c='red')
    Display.Plot_Nodes(mesh, nodes_plate, c='blue', ax=ax)
    Display.Plot_Nodes(mesh, nodes_cuve, c='green', ax=ax)

    Display.Plot_Mesh(mesh, alpha=0.5)

    # ----------------------------------------------
    # Material
    # ----------------------------------------------

    E_pilars = 2000 * 1e9 # GPa
    E_cuve = 20 * 1e9
    E_plate = E_cuve

    E = np.ones(mesh.Ne) * E_cuve
    E[elems_pilars] = E_pilars

    material = Materials.Elas_Isot(3, E, 0.3)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    simu = Simulations.ElasticSimu(mesh, material)
    simu.rho = 7860 # kg/m3

    simu.add_dirichlet(nodesZ0, [0]*3, simu.Get_dofs())
    known, unknown = simu.Bc_dofs_known_unknow(simu.problemType)

    simu.Solver_Set_Newton_Raphson_Algorithm(0.1)

    Display.Plot_BoundaryConditions(simu)

    # K, C, M, F = simu.Get_K_C_M_F()    

    # if isFixed:
    #     K_t = K[unknown, :].tocsc()[:, unknown].tocsr()
    #     M_t = M[unknown, :].tocsc()[:, unknown].tocsr()
    # else:
    #     K_t = K + K.min() * eye(K.shape[0]) * 1e-12
    #     M_t = M


    # eigenValues, eigenVectors = linalg.eigs(K_t, mesh.Nn, M_t)
    # cov = np.cov(M_t.toarray())
    
    # # eigenValues, eigenVectors = linalg.eigs(K_t, mesh.Nn)
    # # cov = np.cov(K_t.toarray())

    # # eigenValues = np.real(eigenValues)


    # # cov = np.cov(M_t.toarray())
    # tt = np.trace(cov)
    # # print(np.max(eigenValues))
    # # print(np.min(eigenValues))

    # err = [1 - np.sum(eigenValues[:i])/np.trace(cov) for i in range(mesh.Nn)]

    # # err = lambda m:  1 - np.sum(eigenValues[:m])/np.trace(cov)

    # ax = Display.init_Axes()

    # ax.plot(range(mesh.Nn), err)

    # # TODO this is very slow !!!!
        
    # linalg.eigsh

    # eigenValues, eigenVectors = eig(K_t.toarray(), M_t.toarray())
    
    # M_t = (M_t.T+M_t)/2

    # eigenValues, eigenVectors = linalg.eigsh(K_t, 3, M_t, which="SM")
    # # eigenValues, eigenVectors = linalg.eigs(K_t, mesh.Nn, M_t)
    # # eigenValues, eigenVectors = eig(K_t.toarray(), M_t.toarray())
    
    # pass

    # # eigenValues = np.array(eigenValues, dtype=float)
    # # eigenVectors = np.array(eigenVectors, dtype=float)

    # freq_t = np.sqrt(eigenValues.real)/2/np.pi

    # # ----------------------------------------------
    # # Plot modes
    # # ----------------------------------------------
    # for n, eigenValue in enumerate(eigenValues[:3]):    

    #     if isFixed:
    #         mode = np.zeros((mesh.Nn, 3))
    #         mode[nodesSupZ0,:] = np.reshape(eigenVectors[:,n], (-1, 3))
    #     else:
    #         mode = np.reshape(eigenVectors[:,n], (-1, 3))

    #     simu.set_u_n(simu.problemType, mode.ravel())
    #     simu.Save_Iter()        

    #     sol = np.linalg.norm(mode, axis=1)
    #     deformFactor = L/5/np.abs(sol).max() 
    #     Display.Plot_Mesh(simu, deformFactor, title=f'mode {n+1}')
    #     # Display.Plot_Result(simu, sol, deformFactor, title=f"mode {n}", plotMesh=True)
    #     pass

    # axModes = Display.init_Axes()
    # axModes.plot(np.arange(eigenValues.size), freq_t, ls='', marker='.')
    # axModes.set_xlabel('modes')
    # axModes.set_ylabel('freq [Hz]')

    Display.plt.show()