import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd

import Folder
import Display
from Interface_Gmsh import Interface_Gmsh, Mesh
from Geom import Point, Domain, Circle
import Materials
import Simulations
import PostProcessing
import pickle
import Folder

# Display.Clear()

folder_file = Folder.Get_Path(__file__)

# ----------------------------------------------
# Configuration
# ----------------------------------------------
test = False
doSimulation = True
optimMesh = True

H = 90
L = 45
ep = 20
D = 10

nL = 100 # 80, 50
l0_init = L/nL
Gc_init = 0.06 # mJ/mm2

inc0 = 5e-3 # inc0 = 8e-3 # incrément platewith hole
inc1 = 1e-3 # inc1 = 2e-3
treshold = 0.2

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

N = 10 # 10, 4
Gc_array = np.linspace(0.01, 0.2, N)
l0_array = np.linspace(L/100, L/10, N)

# ----------------------------------------------
# Mesh
# ----------------------------------------------
def DoMesh(l0: float):

    meshSize = l0 if test else l0/2

    if optimMesh:
        epRefine = D
        refineGeom = Domain(Point(L/2-epRefine), Point(L/2+epRefine, H), meshSize)
        meshSize *= 3
    else:
        refineGeom = None

    domain = Domain(Point(), Point(L, H), meshSize)
    circle = Circle(Point(L/2, H/2), D, meshSize)

    mesh = Interface_Gmsh().Mesh_2D(domain, [circle], "TRI3", refineGeoms=[refineGeom])

    return mesh

mesh = DoMesh(l0_array.min()) # ici on prend le meme maillage pour toutes les simulations

# ----------------------------------------------
# Datas
# ----------------------------------------------

# récupère les courbes forces déplacements
# pathDataFrame = Folder.Join([folder_file, "data_dfEssais.pickle"])
pathDataFrame = Folder.Join([folder_file, "data_dfEssaisRedim.pickle"])
with open(pathDataFrame, "rb") as file:
    dfLoad = pd.DataFrame(pickle.load(file))
# print(dfLoad)

pathDataLoadMax = Folder.Join([folder_file, "data_df_loadMax.pickle"])
with open(pathDataLoadMax, "rb") as file:
    dfLoadMax = pd.DataFrame(pickle.load(file))
# print(dfLoadMax)

# récupère les proritétés identifiées
pathParams = Folder.Join([folder_file, "params_Essais.xlsx"])
dfParams = pd.read_excel(pathParams)
# print(dfParams)

folder = Folder.Join([Folder.New_File("Essais FCBA",results=True), "Grille"])

# ----------------------------------------------
# Simulations
# ----------------------------------------------

dCible = 1

def DoSimu(e: int, g: int, l: int) -> tuple[int, int, int, float]:

    Gc = Gc_array[g]
    l0 = l0_array[l]

    # ----------------------------------------------
    # Material
    # ----------------------------------------------
    forces = dfLoad["forces"][e]
    deplacements = dfLoad["deplacements"][e]

    f_max = np.max(forces)
    f_crit = dfLoadMax["Load [kN]"][e]

    print(f"\nEssai{e}, fcrit = {f_crit:.2f}, Gc = {Gc:.5e}, l0 = {l0:.5e}")
    
    idx_crit = np.where(forces >= f_crit)[0][0]
    dep_crit = deplacements[idx_crit]
    
    El = dfParams["El"][e]
    Et = dfParams["Et"][e]
    Gl = dfParams["Gl"][e]
    # vl = dfParams["vl"][idxEssai]
    vl = 0.02
    vt = 0.44

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])
    axis_t = np.cross(np.array([0,0,1]), axis_l)

    material = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axis_l, axis_t, True, ep)
    
    # mesh
    yn = mesh.coordo[:, 1]
    nodes_Lower = mesh.Nodes_Tags(["L0"])
    nodes_Upper = mesh.Nodes_Tags(["L2"])
    nodes0 = mesh.Nodes_Tags(["P0"])
    
    # construit le modèle d'endommagement
    pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0)
    
    simu = Simulations.Simu_PhaseField(mesh, pfm)
    # simu.solver = "cg"
    simu.solver = "umfpack"

    dofsY_Upper = simu.Bc_dofs_nodes(nodes_Upper, ["y"])

    dep = -inc0

    i = -1
    fr = 0
    
    while simu.damage.max() <= dCible:

        i += 1
        
        dep += inc0 if simu.damage.max() <= treshold else inc1

        # loading
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

    J: float = (fr - f_crit)/f_crit

    return e, g, l, J

if __name__ == "__main__":

    nEssais = 17    
    essais = np.arange(nEssais+1)

    if doSimulation:

        results_e = np.zeros((nEssais+1, N, N), dtype=float)
        items = [(e,g,l) for e in essais for g in range(N) for l in range(N)]        

        DoSimu(0,0,0)

        with multiprocessing.Pool() as pool:
            for res in pool.starmap(DoSimu, items):
                e, g, l, J = tuple(res)
                results_e[e, g, l] = J

        print(results_e)

    # ----------------------------------------------
    # PostProcessing
    # ----------------------------------------------
    for e in essais:

        # ----------------------------------------------
        # Folder
        # ----------------------------------------------
        add = "0" if e < 10 else ""
        essai = f"Essai{add}{e}"
        folder_essai = Folder.Join([folder, essai])
        if test:
            folder_essai = Folder.Join([folder_essai, "Test"])
        simu_name = f"{split} {regu} tolConv{tolConv} optimMesh{optimMesh}"        
        folder_save = Folder.Join([folder_essai, simu_name])        
        print()
        print(folder_save.replace(folder, ''))

        path = Folder.New_File("data.pickle", folder_save)
        
        # ----------------------------------------------
        # Save / Load
        # ----------------------------------------------
        if doSimulation:

            L0, GC = np.meshgrid(l0_array, Gc_array)
            results = results_e[e]

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
