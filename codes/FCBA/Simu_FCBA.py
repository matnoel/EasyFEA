"""Code used to perform phase field simulations with FCBA samples"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

import Folder
import Display
from Gmsh_Interface import Mesher, ElemType
from Geoms import Point, Domain, Circle
import Materials
import Simulations
from BoundaryCondition import BoundaryCondition
import PostProcessing
import Functions

Display.Clear()

folder_file = Folder.Get_Path(__file__)

if __name__  == '__main__':

    # --------------------------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------------------------

    dim = 3
    idxEssai = 10
    model = 1 # 0 -> essai Matthieu et 1 -> essai Laura

    test = True
    solve = True
    optimMesh = True
    useContact = True

    # posProcessing
    pltLoad = True
    pltIter = True
    pltContact = True
    makeParaview = False
    makeMovie = False

    # phase field
    split = "He" # he, Zhang, AnisotStress
    regu = "AT1"
    tolConv = 1e-0 # 1e-0, 1e-1, 1e-2
    convOption = 2
    # (0, bourdin)
    # (1, crack energy)
    # (2, crack + strain energy)

    folderName = 'FCBA'
    if model == 1:
        pltLoad = False
        folderName += ' Laura'

    simuName = "Simu"
    if dim == 3:
        simuName += ' 3D'

    folder_essai = Folder.New_File(Folder.Join(folderName, simuName, f"Essai{idxEssai}"), results=True)

    if useContact:
        folder_essai = Folder.Join(folder_essai, 'Contact')

    # --------------------------------------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------------------------------------    
    if model == 0:    
        # geom
        H = 90
        L = 45
        D = 10
        thickness = 20
        nL = 100
    else:
        # geom
        H = 120
        L = 90
        D = 10
        h = 35
        D2 = 7
        h2 = 55
        thickness = 20
        nL = L//1.5
    
    l0 = L/nL

    folder_save = Folder.PhaseField_Folder(folder_essai, "", split, regu, "", tolConv, "", test, optimMesh, nL=nL)

    pathSimu = Folder.Join(folder_save, "simulation.pickle")
    if not Folder.Exists(pathSimu) and not solve:
        print(folder_save)
        print("la simulation n'existe pas")
        solve = True

    # --------------------------------------------------------------------------------------------
    # Loading
    # --------------------------------------------------------------------------------------------
    treshold = 0.2
    # inc0 = 8e-3
    # inc1 = 2e-3

    inc0 = 8e-3/2
    inc1 = 2e-3/2

    if not solve:
        simu: Simulations.Simu_PhaseField = Simulations.Load_Simu(folder_save)

    # --------------------------------------------------------------------------------------------
    # Mesh
    # --------------------------------------------------------------------------------------------
    if model == 0:
        mesh = Functions.DoMesh(dim,L,H,D,thickness,l0,test, optimMesh)
    else:
        from codes.FCBA.Simu_Elas import DoMesh_FCBA
        mesh = DoMesh_FCBA(dim,L,H,D,h,D2,h2,thickness,l0,test,optimMesh)

    nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
    nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==H)
    nodes0 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
    nodes_edges = mesh.Nodes_Conditions(lambda x,y,z: (x==0) | (y==0) | (x==L) | (y==H))

    if useContact:
        # width = 5
        # master = Domain(Point(-width, h), Point(l+width, h+width), (l+2*width)/20)
        # master_mesh = Interface_Gmsh().Mesh_2D(master, [], ElemType.QUAD4, isOrganised=True)
        # slaveNodes = mesh.Nodes_Conditions(lambda x,y,z: (y==0) | (y==h))

        if model == 0:
            master = Circle(Point(L/2, H/2), D)
        elif model == 1:
            master = Circle(Point(L/2, H-h), D)

        if dim == 2:
            master_mesh = Mesher().Mesh_2D(master, [], ElemType.TRI3)
            slaveNodes = mesh.Nodes_Circle(master)
        elif dim == 3:
            master_mesh = Mesher().Mesh_Extrude(master, [], [0,0,thickness], [], ElemType.TETRA4)
            slaveNodes = mesh.Nodes_Cylinder(master, [0,0,thickness])
        
        # axMesh = Display.Plot_Mesh(mesh)
        # Display.Plot_Mesh(master_mesh, ax=axMesh)
        # Display.Plot_Nodes(mesh, slaveNodes, ax=axMesh)
        # Display._Axis_equal_3D(axMesh, mesh.coordo)

    # --------------------------------------------------------------------------------------------
    # Import Loading
    # --------------------------------------------------------------------------------------------
    if model == 0:
        forces, displacements, f_crit = Functions.Get_loads_informations(idxEssai, True)

        f_max = np.max(forces)
        # f_crit = 10
        idx_crit = np.where(forces >= f_crit)[0][0]
        dep_crit = displacements[idx_crit]

        # calculation of experimental stiffness
        k_exp, __ = Functions.Calc_a_b(forces, displacements, 15)

        # fStop = f_crit*1.2
        fStop = f_max

    # --------------------------------------------------------------------------------------------
    # Material
    # --------------------------------------------------------------------------------------------
    material = Functions.Get_material(idxEssai, thickness, dim)

    # Gc = 0.07 # mJ/mm2
    Gc = 7.6061e-02

    fibreVector = material.axis_t[:dim]
    M = np.einsum("i,j->ij", fibreVector, fibreVector)
    Betha = material.El/material.Et
    Betha = 0
    A = np.eye(dim) + Betha * (np.eye(dim) - M)

    pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0, A=A)

    if solve:
        simu = Simulations.Simu_PhaseField(mesh, pfm)    
    else:
        pfm = simu.phaseFieldModel
        material = pfm.material
    
    if useContact:
        dofsY_upper = simu.Bc_dofs_nodes(slaveNodes, ['y'])
    else:
        dofsY_upper = simu.Bc_dofs_nodes(nodes_upper, ['y'])


    damageMax = []
    list_fr = []
    list_dep = []

    if solve:

        if pltLoad:

            # Numerical slope calculation
            simuElas = Simulations.Simu_Displacement(mesh, material)
            simuElas.add_dirichlet(nodes_lower, [0]*dim, simu.Get_directions())
            simuElas.add_surfLoad(nodes_upper, [-f_crit*1000/L/thickness], ["y"])
            u_num = simuElas.Solve()
            
            fr_num = - np.sum(simuElas.Get_K_C_M_F()[0][dofsY_upper] @ u_num)/1000
            k_mat, __ = Functions.Calc_a_b(np.linspace(0, fr_num, 50), np.linspace(0, -np.mean(u_num[dofsY_upper]), 50), f_crit)

            k_montage = 1/(1/k_exp - 1/k_mat)

            # --------------------------------------------------------------------------------------------
            # Plot
            # --------------------------------------------------------------------------------------------
            axLoad = plt.subplots()[1]
            # axLoad.plot(deplacements, forces, label="exp")
            axLoad.set_xlabel("x [mm]")
            axLoad.set_ylabel("f [kN]")

            deplMat = np.linspace(0, -np.mean(u_num[dofsY_upper]), 20)
            forcesMat = np.linspace(forces[0], fr_num, 20)
            # axLoad.scatter(deplMat, forcesMat, label="mat", marker=".", c="black", zorder=10)

            # coef_a = k_mat/k_exp    
            # axLoad.plot(deplacements/coef_a, forces, label="(1)")    
            
            displacements = displacements-forces/k_montage
            axLoad.scatter(displacements[idx_crit], forces[idx_crit], marker='.', c='red', zorder=10)
            axLoad.text(displacements[idx_crit], forces[idx_crit],'$max(\phi)=1$',size=14,va='top')
            axLoad.plot(displacements, forces, label="redim", c='blue')


            argMax = np.argmax(forces)
            axLoad.axhline(np.max(forces),c='gray',ls='--')
            # axLoad.scatter(displacements[argMax], forces[argMax], marker='.', c='blue', zorder=10)
            # axLoad.text(displacements[argMax], forces[argMax],'(2)')

            # axLoad.legend()
            axLoad.grid()
            Display.Save_fig(folder_save, "load")

        # --------------------------------------------------------------------------------------------
        # Simulation
        # --------------------------------------------------------------------------------------------
        dep = -inc0
        fr = 0
        i = -1

        def Condition() -> bool:
            if model == 0:
                return fr <= fStop
            elif model == 1:
                # return simu.damage.max() <= 1
                return i <= 100

        while Condition():

            i += 1
            dep += inc0 if simu.damage.max() <= treshold else inc1

            simu.Bc_Init()        

            if useContact:
                simu.add_dirichlet(nodes0, [0], ["x"])
                simu.add_dirichlet(nodes_lower, [0], ["y"])

                # update master mesh coordinates
                dy = -inc0 if simu.damage.max() <= treshold else -inc1
                master_mesh.translate(dy=dy)
                
                nodes_cU, newU = simu.Get_contact(master_mesh, slaveNodes)
                if nodes_cU.size > 0:
                    # simu.add_dirichlet(nodes_cU, [newU[:,0], newU[:,1]], ['x','y'])
                    simu.add_dirichlet(nodes_cU, [newU[:,1]], ['y'])
                
            else:
                
                simu.add_dirichlet(nodes_lower, [0], ["y"])
                simu.add_dirichlet(nodes0, [0], ["x"])
                simu.add_dirichlet(nodes_upper, [-dep], ["y"])        

            # solve and save iter
            u, d, Kglob, convergence = simu.Solve(tolConv, convOption=2)
            simu.Save_Iter()

            damageMax.append(np.max(d))

            f = Kglob @ u

            f_Upper = f[dofsY_upper]

            fr = - np.sum(f_Upper)/1000

            if model == 0:
                simu.Results_Set_Iteration_Summary(i, fr, "kN", fr/fStop, True)
            elif model == 1:
                simu.Results_Set_Iteration_Summary(i, fr, "kN", d.max(), True)

            list_fr.append(fr)
            list_dep.append(dep)

            if pltLoad:

                depp = -np.mean(simu.displacement[dofsY_upper]) if useContact else dep

                plt.figure(axLoad.figure)
                axLoad.scatter(depp, fr, c='black', marker='.')        
                # axLoad.scatter(depp, fr, marker='.')        
                if np.max(D) >= 1:
                    axLoad.scatter(depp, fr, c='red', marker='.')
                plt.pause(1e-12)

            if pltIter:
                if i == 0:
                    _, axIter, cbIter = Display.Plot_Result(simu, "damage")
                else:
                    cbIter.remove()
                    factorDef = 1 if pltContact else 0
                    _, axIter, cbIter = Display.Plot_Result(simu, "damage", ax=axIter, deformFactor=factorDef)

                # title = axIter.get_title()
                # if pltContact and useContact:
                #     # Display.Plot_Mesh(master_mesh, alpha=0, ax=axIter)
                #     if nodes_cU.size > 0:
                #         Display.Plot_Nodes(mesh, nodes_cU, ax=axIter)
                #         Display._ScaleChange(axIter, mesh.coordo)
                # title = axIter.set_title(title)

                plt.figure(axIter.figure)        
                
            plt.pause(1e-12)    

            if not convergence or True in (d[nodes_edges] >= 0.98):
                print("\nPas de convergence")
                break

        # --------------------------------------------------------------------------------------------
        # Save
        # --------------------------------------------------------------------------------------------
        damageMax = np.array(damageMax)
        list_fr = np.array(list_fr)
        list_dep = np.array(list_dep)

        PostProcessing.Save_Load_Displacement(list_fr, list_dep, folder_save)

        fDamageSimu = list_fr[np.where(damageMax >= 0.95)[0][0]]
        
        if pltLoad:
            plt.figure(axLoad.figure)
            Display.Save_fig(folder_save, "forcedep")

        Display.Plot_Iter_Summary(simu, folder_save)

        simu.Save(folder_save)

    # --------------------------------------------------------------------------------------------
    # Results
    # --------------------------------------------------------------------------------------------
    Display.Plot_Result(simu, 'damage', folder=folder_save, colorbarIsClose=True)

    if makeParaview:
        PostProcessing.Make_Paraview(folder_save, simu)

    if makeMovie:
        PostProcessing.Make_Movie(folder_save, "damage", simu)

    plt.show()