import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

import Folder
import Display
import Simulations
import PostProcessing
import Materials
from Mesh import Mesh

import Functions

# Display.Clear()

folder = Folder.Join([Folder.New_File("Essais FCBA",results=True), "Identification"])

# ----------------------------------------------
# Configuration
# ----------------------------------------------
test = False
doSimulation = True
detectL0 = False
optimMesh = True

H = 90
L = 45
thickness = 20
D = 10
nL = 100 # 80, 50

l0_init = L/nL
Gc_init = 0.06 # mJ/mm2
GcMax = 2
lb = [0] if not detectL0 else [0, 0]
ub = [GcMax] if not detectL0 else [GcMax, L/20]
x0 = [Gc_init] if not detectL0 else [Gc_init, l0_init]

# solver = 0 # least_squares
solver = 1 # minimize

# ftol = 1e-12
# ftol = 1e-5
# ftol = 1e-3
ftol = 1e-2
# ftol = 1e-1

# split = "AnisotStress"
# split = "Zhang"
split = "He"
regu = "AT1"

# convOption = 0 # bourdin
# convOption = 1 # energie crack
convOption = 2 # energie tot

# tolConv = 1e-0
tolConv = 1e-2
# tolConv = 1e-3

# ----------------------------------------------
# Simu
# ----------------------------------------------
evals = []
simus: list[Simulations.Simu_PhaseField] = []

def DoSimu(x: np.ndarray, mesh: Mesh, idxEssai: int) -> float:

    material = Functions.Get_material(idxEssai, thickness)
    f_crit = Functions.Get_loads_informations(idxEssai)[-1]

    Gc = x[0]

    if detectL0:
        l0 = x[1]            
        print(f"\nGc = {x[0]:.5e}, l0 = {x[1]:.5e}")    
        mesh = Functions.DoMesh(L, H, D, l0, test, optimMesh)
    else:
        l0 = l0_init
        print(f"\nGc = {x[0]:.5e}")
        
    # mesh
    yn = mesh.coordo[:, 1]
    nodes_Lower = mesh.Nodes_Tags(["L0"])
    nodes_Upper = mesh.Nodes_Tags(["L2"])
    nodes0 = mesh.Nodes_Tags(["P0"])    
    
    pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0)    
    simu = Simulations.Simu_PhaseField(mesh, pfm)
    simus.clear(); simus.append(simu)

    dofsY_Upper = simu.Bc_dofs_nodes(nodes_Upper, ["y"])
    inc0 = 5e-3 # inc0 = 8e-3 # increment platewith hole
    inc1 = 1e-3 # inc1 = 2e-3
    treshold = 0.2
    dep = -inc0
    i = -1
    fr = 0    
    while simu.damage.max() <= 1:

        i += 1
        
        dep += inc0 if simu.damage.max() <= treshold else inc1

        # chargement
        simu.Bc_Init()
        simu.add_dirichlet(nodes_Lower, [0], ["y"])
        simu.add_dirichlet(nodes0, [0], ["x"])
        simu.add_dirichlet(nodes_Upper, [-dep], ["y"])

        # resolution
        u, d, Kglob, convergence = simu.Solve(tolConv, convOption=convOption)
        simu.Save_Iter()

        # force résultante
        f = Kglob[dofsY_Upper,:] @ u
        fr = - np.sum(f)/1000
        
        simu.Results_Set_Iteration_Summary(i, fr, "kN", 0, True)

    if solver == 0: # least_squares    
        J = (fr - f_crit)/f_crit
    elif solver == 1: # minimize
        J = (fr - f_crit)**2/f_crit**2
    
    print(f'\nfr = {fr}')
    print(f"J = {J:.5e}")        
    evals.append(J)

    return J

if __name__ == '__main__':

    for idxEssai in np.arange(0,18):

        # folder to save in
        add = "0" if idxEssai < 10 else ""
        essai = f"Essai{add}{idxEssai}"
        folder_essai = Folder.Join([folder, essai])
        if test:
            folder_essai = Folder.Join([folder_essai, "Test"])
        
        simu_name = f"{split} {regu} tolConv{tolConv} optimMesh{optimMesh} ftol{ftol}"    
        if not detectL0:
            simu_name += f" nL{nL}"

        folder_save = Folder.Join([folder_essai, simu_name])

        print()
        print(folder_save.replace(folder, ''))

        # ----------------------------------------------
        # Datas
        # ----------------------------------------------
        forces, deplacements, f_crit = Functions.Get_loads_informations(idxEssai)
        print(f"fcrit = {f_crit}")

        mesh_init = Functions.DoMesh(L, H, D, l0_init, test, optimMesh)

        # ----------------------------------------------
        # Solve
        # ----------------------------------------------        
        if doSimulation:
            
            evals.clear() # clear evals to store new J
            if solver == 0:
                res = least_squares(DoSimu, x0, bounds=(lb, ub), verbose=0, ftol=ftol, xtol=None, gtol=None, args=(mesh_init,idxEssai))
            elif solver == 1:
                bounds = [(l, u) for l, u in zip(lb, ub)]
                res = minimize(DoSimu, x0, bounds=bounds, tol=ftol, args=(mesh_init,idxEssai))

            Gc = res.x[0]
            if detectL0:
                l0 = res.x[1]
                print(f"Gc = {Gc:.10e}, l0 = {l0:.4e}")
            else:
                l0 = l0_init
                print(f"Gc = {Gc:.10e}")    

            print(res)

            # ----------------------------------------------
            # Save
            # ----------------------------------------------
            
            # save iterations
            ax_J = plt.subplots()[1]
            ax_J.set_xlabel("$N$"); ax_J.set_ylabel("$J$")
            ax_J.grid()
            ax_J.scatter(np.arange(len(evals)), evals, c='black', zorder=4)
            Display.Save_fig(folder_save, "iterations")           
            
            # get last simulation
            simu = simus[-1]
            simu.Save(folder_save)
            Display.Plot_Iter_Summary(simu, folder_save)

            simu.Update_Iter(-1)

            dofsY = simu.Bc_dofs_nodes(simu.mesh.Nodes_Conditions(lambda x,y,z: y==H), ["y"])
            fr = -np.sum(simu.Get_K_C_M_F()[0][dofsY,:] @ simu.displacement)/1000

            pathData = Folder.Join([folder, "identification.xlsx"])

            data = [
                {
                    "Essai": essai,
                    "split": simu.phaseFieldModel.split,
                    "regu": simu.phaseFieldModel.regularization,
                    "tolConv": tolConv,
                    "test": test,
                    "optimMesh": optimMesh,
                    "solveur": solver,
                    "ftol": ftol,
                    "detectL0": detectL0,
                    "f_crit": f_crit,
                    "fr": fr,
                    "err": np.abs(fr-f_crit)/f_crit,
                    "Gc": Gc,
                    "l0": l0
                }
            ]

            if Folder.Exists(pathData):
                df = pd.read_excel(pathData)
                newDf = pd.DataFrame(data)
                df = pd.concat([df,newDf])
            else:
                df = pd.DataFrame(data)

            df.to_excel(pathData, index=False)

        else:
            # charge la simulation
            simu: Simulations.Simu_PhaseField = Simulations.Load_Simu(folder_save)
        
        # ----------------------------------------------
        # PostProcessing
        # ----------------------------------------------
        deplacementsIdentif = []
        forcesIdentif = []
        dofsY = simu.Bc_dofs_nodes(simu.mesh.Nodes_Conditions(lambda x,y,z: y==H), ["y"])
        for iter in range(len(simu.results)):

            simu.Update_Iter(iter)

            displacement = simu.displacement
            deplacementsIdentif.append(-np.mean(displacement[dofsY]))
            forcesIdentif.append(-np.sum(simu.Get_K_C_M_F()[0][dofsY,:] @ displacement)/1000)

        deplacementsIdentif = np.asarray(deplacementsIdentif)
        forcesIdentif = np.asarray(forcesIdentif)
        PostProcessing.Save_Load_Displacement(forcesIdentif, deplacementsIdentif, folder_save)        

        k_exp, __ = Functions.Calc_a_b(forces, deplacements, 15)
        k_mat, __ = Functions.Calc_a_b(forcesIdentif, deplacementsIdentif, 15)
        k_montage = 1/(1/k_exp - 1/k_mat)
        deplacements = deplacements-forces/k_montage

        # plot loads displacements
        axLoad = plt.subplots()[1]
        axLoad.set_xlabel("x [mm]"); axLoad.set_ylabel("f [kN]"); axLoad.grid()
        axLoad.plot(deplacements, forces, label="exp")
        idx_crit = np.where(forces >= f_crit)[0][0]
        axLoad.scatter(deplacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=3)
        axLoad.plot(deplacementsIdentif, forcesIdentif, label="identif")
        axLoad.legend()
        Display.Save_fig(folder_save, "load")

        Display.Plot_Result(simu, "damage", folder=folder_save, colorbarIsClose=True)

        plt.close('all')