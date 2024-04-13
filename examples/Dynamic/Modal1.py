"""Modal analysis of a wall structure."""

from EasyFEA import (Display, Folder, np,
                     Mesher, ElemType,
                     Materials, Simulations)
from EasyFEA.Geoms import Point, Domain

from scipy.sparse import linalg, eye

folder = Folder.Get_Path(__file__)

if __name__ == '__main__':

    Display.Clear()
    
    dim = 3
    isFixed = True

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------
    contour = Domain(Point(), Point(1,1))
    thickness = 1/10

    if dim == 2:
        mesh = Mesher().Mesh_2D(contour, [], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = Mesher().Mesh_Extrude(contour, [], [0,0,-thickness], [2], ElemType.HEXA8, isOrganised=True)
    nodesY0 = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    nodesSupY0 = mesh.Nodes_Conditions(lambda x,y,z: y>0)

    Display.Plot_Mesh(mesh)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------
    material = Materials.Elas_Isot(dim, planeStress=True, thickness=thickness)

    simu = Simulations.ElasticSimu(mesh, material)

    simu.Solver_Set_Newton_Raphson_Algorithm(0.1)

    K, C, M, F = simu.Get_K_C_M_F()
    
    if isFixed:
        simu.add_dirichlet(nodesY0, [0]*dim, simu.Get_dofs())
        known, unknown = simu.Bc_dofs_known_unknow(simu.problemType)
        K_t = K[unknown, :].tocsc()[:, unknown].tocsr()
        M_t = M[unknown, :].tocsc()[:, unknown].tocsr()

    else:        
        K_t = K + K.min() * eye(K.shape[0]) * 1e-12
        M_t = M

    eigenValues, eigenVectors = linalg.eigs(K_t, 10, M_t, which="SM")

    eigenValues = np.array(eigenValues.real, dtype=float)
    eigenVectors = np.array(eigenVectors.real, dtype=float)

    freq_t = np.sqrt(eigenValues.real)/2/np.pi

    # ----------------------------------------------
    # Plot modes
    # ----------------------------------------------
    for n in range(eigenValues.size):

        if isFixed:
            mode = np.zeros((mesh.Nn, dim))
            mode[nodesSupY0,:] = np.reshape(eigenVectors[:,n], (-1, dim))
        else:
            mode = np.reshape(eigenVectors[:,n], (-1, dim))

        simu._Set_u_n(simu.problemType, mode.ravel())
        simu.Save_Iter()        

        sol = np.linalg.norm(mode, axis=1)
        deformFactor = 1/5/np.abs(sol).max() 
        Display.Plot_Mesh(simu, deformFactor, title=f'mode {n+1}')
        # Display.Plot_Result(simu, sol, deformFactor, title=f"mode {n}", plotMesh=True)
        pass

    axModes = Display.init_Axes()
    axModes.plot(np.arange(eigenValues.size), freq_t, ls='', marker='.')
    axModes.set_xlabel('modes')
    axModes.set_ylabel('freq [Hz]')

    Display.plt.show()