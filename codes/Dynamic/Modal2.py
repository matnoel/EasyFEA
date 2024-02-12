from Interface_Gmsh import Mesher, gmsh, Mesh
import Display
import Folder
import Simulations
import Materials
import PostProcessing

import sys
import os
import numpy as np
from scipy.sparse import linalg, eye

if __name__ == '__main__':

    Display.Clear()

    folder = Folder.Get_Path(__file__)

    folderSave = Folder.New_File("ModalAnalysis",results=True)
    if not os.path.exists(folderSave): os.makedirs(folderSave)

    isFixed = True

    def Construct_struct(L: float,e: float,t: float, meshSize: float = 0.0, openGmsh=False, verbosity=False) -> Mesh:

        Display.Clear()

        # ----------------------------------------------
        # GMSH
        # ----------------------------------------------
        gmsh.initialize()

        if not verbosity:
            gmsh.option.setNumber('General.Verbosity', 0)

        h = L-e-t

        factory = gmsh.model.occ

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

        factory.synchronize()

        if meshSize > 0:
            gmsh.model.mesh.setSize(factory.getEntities(0), meshSize)

        factory.synchronize()

        gmsh.model.mesh.generate(3)

        msh = Folder.Join(folderSave, 'mesh.msh')
        gmsh.write(msh)

        if openGmsh: gmsh.fltk.run()

        gmsh.finalize()

        mesh = Mesher().Mesh_Import_mesh(msh, True)

        # Folder.os.remove(msh)

        return mesh

    if __name__ == '__main__':

        Display.Clear()

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
        # Simu
        # ----------------------------------------------

        simu = Simulations.Simu_Displacement(mesh, material)
        simu.rho = 7860 # kg/m3

        simu.add_dirichlet(nodesZ0, [0]*3, simu.Get_directions())
        known, unknown = simu.Bc_dofs_known_unknow(simu.problemType)

        simu.Solver_Set_Newton_Raphson_Algorithm(0.1)

        K, C, M, F = simu.Get_K_C_M_F()    

        if isFixed:
            K_t = K[unknown, :].tocsc()[:, unknown].tocsr()
            M_t = M[unknown, :].tocsc()[:, unknown].tocsr()
        else:
            K_t = K + K.min() * eye(K.shape[0]) * 1e-12
            M_t = M

        # # eigenValues, eigenVectors = linalg.eigsh(K_t, mesh.Nn, M_t)
        # eigenValues, eigenVectors = linalg.eigs(K_t, 10, M_t, which="SM")
        # # eigenValues, eigenVectors = linalg.eigsh(K_t, 10, M_t, which="SM")

        # eigenValues = np.array(eigenValues, dtype=float)
        # eigenVectors = np.array(eigenVectors, dtype=float)

        # freq_t = np.sqrt(eigenValues.real)/2/np.pi

        # for n, eigenValue in enumerate(eigenValues[:]):

        #     if isFixed:
        #         mode = np.zeros((mesh.Nn, 3))
        #         mode[nodesSupZ0,:] = np.reshape(eigenVectors[:,n], (-1, 3))
        #     else:
        #         mode = np.reshape(eigenVectors[:,n], (-1, 3))

        #     simu.set_u_n(simu.problemType, mode.reshape(-1))
        #     simu.Save_Iter()        

        #     sol = np.linalg.norm(mode, axis=1)
        #     deformFactor = L/5/np.abs(sol).max() 
        #     Display.Plot_Mesh(simu, deformFactor, title=f'mode {n+1}')
        #     # Display.Plot_Result(simu, sol, deformFactor, title=f"mode {n}", plotMesh=True)
        #     pass

        # axModes = Display.plt.subplots()[1]
        # axModes.plot(np.arange(eigenValues.size), freq_t, ls='', marker='.')
        # axModes.set_xlabel('modes')
        # axModes.set_ylabel('freq [Hz]')
        

        Display.plt.show()