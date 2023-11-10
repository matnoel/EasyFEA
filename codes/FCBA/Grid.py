import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import multiprocessing

import Display
import Materials
import Simulations
import PostProcessing
import Folder
import Functions

# Display.Clear()

folder = Folder.Join([Folder.New_File("Essais FCBA",results=True), "Grille"])

useParallel = True
nProcs = 6 # number of processes in parallel

# ----------------------------------------------
# Configuration
# ----------------------------------------------
test = False
doSimulation = True
optimMesh = True

H = 90
L = 45
thickness = 20
D = 10

nL = 100 # 80, 50
l0_init = L/nL
Gc_init = 0.06 # mJ/mm2

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

N = 25 # 10, 4
Gc_array = np.linspace(0.01, 0.3, N)
l0_array = np.linspace(L/100, L/10, N)

# ----------------------------------------------
# Mesh
# ----------------------------------------------
# here we use the same mesh for all simulations
mesh = Functions.DoMesh(L, H, D, l0_array.min(), test, optimMesh)
yn = mesh.coordo[:, 1]
nodes_Lower = mesh.Nodes_Tags(["L0"])
nodes_Upper = mesh.Nodes_Tags(["L2"])
nodes0 = mesh.Nodes_Tags(["P0"])  

# ----------------------------------------------
# Simu
# ----------------------------------------------
def DoSimu(idxEssai: int, g: int, l: int) -> tuple[int, int, int, float]:

    # datas to do the simulation
    f_crit = Functions.Get_loads_informations(idxEssai)[-1]
    material = Functions.Get_material(idxEssai, thickness)
    Gc = Gc_array[g]
    l0 = l0_array[l]

    print(f"\nGc = {Gc:.3e}, l0 = {l0:.3e}")      
    
    # construct phase field simulation
    pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0)    
    simu = Simulations.Simu_PhaseField(mesh, pfm)
    # simu.solver = "cg" 

    # boundary conditions
    dofsY_Upper = simu.Bc_dofs_nodes(nodes_Upper, ["y"])
    inc0 = 5e-3 # inc0 = 8e-3 # platewith hole increment
    inc1 = 1e-3 # inc1 = 2e-3
    treshold = 0.2
    
    dep = -inc0
    i = -1
    fr = 0    
    while simu.damage.max() <= 1:

        i += 1
        
        dep += inc0 if simu.damage.max() <= treshold else inc1

        # loading
        simu.Bc_Init()
        simu.add_dirichlet(nodes_Lower, [0], ["y"])
        simu.add_dirichlet(nodes0, [0], ["x"])
        simu.add_dirichlet(nodes_Upper, [-dep], ["y"])

        # solve
        u, d, Kglob, convergence = simu.Solve(tolConv, convOption=convOption)
        simu.Save_Iter()

        # force résultante
        f = Kglob[dofsY_Upper,:] @ u
        fr = - np.sum(f)/1000
        
        simu.Results_Set_Iteration_Summary(i, fr, "kN", 0, True)

    J: float = (fr - f_crit)/f_crit

    return g, l, J

if __name__ == "__main__":

    nEssais = 17    
    # idxEssais = np.arange(nEssais+1)
    idxEssais = np.arange(10,11)

    for idxEssai in idxEssais:

        # ----------------------------------------------
        # Folder
        # ----------------------------------------------
        add = "0" if idxEssai < 10 else ""
        essai = f"Essai{add}{idxEssai}"
        folder_essai = Folder.Join([folder, essai])
        if test:
            folder_essai = Folder.Join([folder_essai, "Test"])
        simu_name = f"{split} {regu} tolConv{tolConv} optimMesh{optimMesh}"        
        folder_save = Folder.Join([folder_essai, simu_name])        
        print("\n"+folder_save.replace(folder, ''))
        print()

        path = Folder.New_File("data.pickle", folder_save)

        if doSimulation:

            L0, GC = np.meshgrid(l0_array, Gc_array)
            
            results = np.zeros((N, N), dtype=float)

            if useParallel:
                items = [(idxEssai,g,l) for g in range(N) for l in range(N)]
                with multiprocessing.Pool(nProcs) as pool:
                    for res in pool.starmap(DoSimu, items):
                        g, l, J = tuple(res)
                        results[g, l] = J
            else:
                for g in range(N):
                    for l in range(N):
                        J = DoSimu(idxEssai, g, l)[-1]
                        results[g,l] = J

            with open(path, 'wb') as file:
                data = {
                    'GC': GC,
                    'L0': L0,
                    'results': results
                }
                pickle.dump(data, file)
        
        else:
            # récupère les données
            with open(path, 'rb') as file:
                data = pickle.load(file)
                GC = data['GC']
                L0 = data['L0']
                results = data['results']

        results = results**2

        # ----------------------------------------------
        # Post-Processing
        # ----------------------------------------------                
        lMax = L0.max()
        cols = np.where(L0[0] <= lMax)[0]
        GC = GC[:,cols]
        L0 = L0[:,cols]
        results = results[:,cols]

        levels = np.linspace(results.min(), results.max(), 255)
        ticks = np.linspace(results.min(), results.max(), 11)
        argmin = np.where(results == results.min())

        axeX = "$G_c \ [mJ \ mm^{-2}]$"
        axeY = "$\ell \ [mm]$"

        fig = plt.figure()
        ax1 = fig.add_subplot(projection="3d")
        cc = ax1.plot_surface(GC, L0, results, cmap='jet')
        fig.colorbar(cc, ticks=ticks)
        ax1.set_xlabel(axeX, fontsize=14)
        ax1.set_ylabel(axeY, fontsize=14)
        ax1.set_title("$J$", fontsize=14)
        ax1.scatter(GC[argmin], L0[argmin], results[argmin], c='red', marker='.', zorder=10)    

        Display.Save_fig(folder_save, "J surface")

        ax2 = plt.subplots()[1]        
        cc = ax2.contourf(GC, L0, results,levels,  cmap='jet')
        ax2.set_xlabel(axeX, fontsize=14)
        ax2.set_ylabel(axeY, fontsize=14)
        ax2.set_title("$J$", fontsize=14)
        ax2.scatter(GC[argmin], L0[argmin], 200, c='red', marker='.', zorder=10,edgecolors='white')
        ax2.figure.colorbar(cc, ticks=ticks)

        Display.Save_fig(folder_save, "J contourf")

        # Display.Save_fig(folder_Save, "J_grid")
        # plt.show()

        pass

        plt.close('all')