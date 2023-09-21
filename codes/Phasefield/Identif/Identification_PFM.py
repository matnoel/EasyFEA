import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
import os
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
doIdentif = True # False -> Plot grid values
detectL0 = False

optimMesh = True

pltIter = False

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

solver = 0 # least_squares
# solveur = 1 # minimize

# ftol = 1e-12
# ftol = 1e-5
# ftol = 1e-3
ftol = 1e-2
# ftol = 1e-1

# split = "AnisotStress"
split = "He"
# split = "Zhang"
regu = "AT2"

# convOption = 0 # bourdin
# convOption = 1 # energie crack
convOption = 2 # energie tot

# tolConv = 1e-0
tolConv = 1e-2
# tolConv = 1e-3

# ----------------------------------------------
# Données
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

folder_FCBA = Folder.New_File("Essais FCBA",results=True)

if doIdentif:
    folder = Folder.Join([folder_FCBA, "Identification"])
else:
    folder = Folder.Join([folder_FCBA, "Grille"])

for idxEssai in range(1,18):

    # Dossier de l'essai

    add = "0" if idxEssai < 10 else ""

    essai = f"Essai{add}{idxEssai}"

    folder_essai = Folder.Join([folder, essai])

    if test:
        folder_essai = Folder.Join([folder_essai, "Test"])

    simu_name = f"{split} {regu} tolConv{tolConv} optimMesh{optimMesh}"

    if doIdentif:
        simu_name += f" ftol{ftol}"
        if not detectL0:
            simu_name += f" nL{nL}"

    folder_save = Folder.Join([folder_essai, simu_name])
    
    print()
    print(folder_save.replace(folder, ''))

    # ----------------------------------------------
    # Données de l'essai
    # ----------------------------------------------
    forces = dfLoad["forces"][idxEssai]
    deplacements = dfLoad["deplacements"][idxEssai]

    f_max = np.max(forces)
    f_crit = dfLoadMax["Load [kN]"][idxEssai]
    # f_crit = 10
    print(f"fcrit = {f_crit}")
    
    idx_crit = np.where(forces >= f_crit)[0][0]
    dep_crit = deplacements[idx_crit]
    
    El = dfParams["El"][idxEssai]
    Et = dfParams["Et"][idxEssai]
    Gl = dfParams["Gl"][idxEssai]
    # vl = dfParams["vl"][idxEssai]
    vl = 0.02
    vt = 0.44    

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])
    axis_t = np.cross(np.array([0,0,1]), axis_l)

    material = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axis_l, axis_t, True, ep)

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

    # ----------------------------------------------
    # Phase field
    # ----------------------------------------------
    vectB = np.array([1,0])
    vectB = axis_t[:2]
    matB = np.einsum("i,j->ij", vectB, vectB)

    # Betha = El/Et # 5
    Betha = 0
    A = np.eye(2) + Betha * (np.eye(2) - matB)

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------

    dCible = 1
    evals = []
    simus: list[Simulations.Simu_PhaseField] = []

    def DoSimu(x: np.ndarray, mesh: Mesh) -> float:
        """Simulation pour les paramètres x=[Gc,l0]"""

        Gc = x[0]

        if x.size > 1:
            l0 = x[1]            
            print(f"\nGc = {x[0]:.5e}, l0 = {x[1]:.5e}")
            if detectL0:
                mesh = DoMesh(l0)
        else:
            l0 = l0_init
            print(f"\nGc = {x[0]:.5e}")
            
        yn = mesh.coordo[:, 1]
        nodes_Lower = mesh.Nodes_Tags(["L0"])
        nodes_Upper = mesh.Nodes_Tags(["L2"])
        nodes0 = mesh.Nodes_Tags(["P0"])
        
        # construit le modèle d'endommagement
        pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0, A=A)
        
        simu = Simulations.Simu_PhaseField(mesh, pfm)
        simus.clear(); simus.append(simu)

        ddlsY_Upper = simu.Bc_dofs_nodes(nodes_Upper, ["y"])

        dep = -inc0

        i = -1
        fr = 0
        
        while simu.damage.max() <= dCible:

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
            f = Kglob[ddlsY_Upper,:] @ u
            fr = - np.sum(f)/1000
            
            simu.Results_Set_Iteration_Summary(i, fr, "kN", 0, True)

        # calcul de l'erreur entre la force résultante calculée et la force expérimentale
        J = (fr - f_crit)/f_crit        
        if doIdentif:
            print(f'\nfr = {fr}')
            print(f"J = {J:.5e}")
            
            evals.append(J)
            Niter = len(evals)
            

            if pltIter:      
                ax_J.scatter(Niter, evals[-1], c="black", zorder=4)
                plt.figure(ax_J.figure)
                plt.pause(1e-12)
        
        return J
            

    # ----------------------------------------------
    # Simulations
    # ----------------------------------------------

    if doIdentif:

        if doSimulation:

            mesh = DoMesh(l0_init)

            # création de la figure pour tracer J
            ax_J = plt.subplots()[1]
            ax_J.set_xlabel("$N$"); ax_J.set_ylabel("$J$")
            ax_J.grid()

            GcMax = 2
            lb = [0] if not detectL0 else [0, 0]
            ub = [GcMax] if not detectL0 else [GcMax, L/20]
            x0 = [Gc_init] if not detectL0 else [Gc_init, l0_init]

            if solver == 0:
                res = least_squares(DoSimu, x0, bounds=(lb, ub), verbose=0, ftol=ftol, xtol=None, gtol=None, args=(mesh,))
            elif solver == 1:
                bounds = [(l, u) for l, u in zip(lb, ub)]
                res = minimize(DoSimu, x0, bounds=bounds, tol=ftol, args=(mesh,))

            Gc = res.x[0]
            if detectL0:
                l0 = res.x[1]
                print(f"Gc = {Gc:.10e}, l0 = {l0:.4e}")
            else:
                l0 = l0_init
                print(f"Gc = {Gc:.10e}")

            x = res.x

            print(res)
            
            # ----------------------------------------------
            # Sauvegarde des données
            # ----------------------------------------------
            if pltIter:
                plt.figure(ax_J.figure)
            else:                
                ax_J.scatter(np.arange(len(evals)), evals, c='black', zorder=4)
            Display.Save_fig(folder_save, "iterations")
            
            # Récupère la simulation            
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
                    # "A": A,
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

        # reconstruction de la courbe force déplacement
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

        def Calc_a_b(forces, deplacements, fmax):
            """Calcul des coefs de f(x) = a x + b"""

            idxElas = np.where((forces <= fmax))[0]
            idx1, idx2 = idxElas[0], idxElas[-1]
            x1, x2 = deplacements[idx1], deplacements[idx2]
            f1, f2 = forces[idx1], forces[idx2]
            vect_ab = np.linalg.inv(np.array([[x1, 1],[x2, 1]])).dot(np.array([f1, f2]))
            a, b = vect_ab[0], vect_ab[1]

            return a, b

        k_exp, __ = Calc_a_b(forces, deplacements, 15)
        k_mat, __ = Calc_a_b(forcesIdentif, deplacementsIdentif, 15)
        k_montage = 1/(1/k_exp - 1/k_mat)
        deplacements = deplacements-forces/k_montage

        # trace la fonction force déplacement
        axLoad = plt.subplots()[1]
        axLoad.set_xlabel("x [mm]"); axLoad.set_ylabel("f [kN]"); axLoad.grid()
        axLoad.plot(deplacements, forces, label="exp")
        axLoad.scatter(deplacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=3)
        axLoad.plot(deplacementsIdentif, forcesIdentif, label="identif")
        axLoad.legend()
        Display.Save_fig(folder_save, "load")

        Display.Plot_Result(simu, "damage", folder=folder_save, colorbarIsClose=True)

    else:

        N = 10
        # N = 4

        Gc_array = np.linspace(0.01, 0.2, N)
        l0_array = np.linspace(L/100, L/10, N)
        # l0_array = np.linspace(L/20, L/10, N)

        mesh = DoMesh(l0_array.min()) # ici on prend le meme maillage pour toutes les simulations

        L0, GC = np.meshgrid(l0_array, Gc_array)        

        path = Folder.New_File("data.pickle", folder_save)

        if doSimulation:
            # Sauvegarde les données

            results = np.zeros_like(GC)

            for g, gc in enumerate(Gc_array):
                for l, l0 in enumerate(l0_array):
                    
                    ecart = DoSimu(np.array([gc,l0]), mesh)

                    results[g,l] = ecart

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

        lMax = L/10
        lMax = L0[0,-4]
        lMax = L0.max()

        cols = np.where(L0[0] <= lMax)[0]
        GC = GC[:,cols]
        L0 = L0[:,cols]
        results = results[:,cols]

        levels = np.linspace(results.min(), results.max(), 255)
        ticks = np.linspace(results.min(), results.max(), 11)
        argmin = np.where(results == results.min())

        axeX = "$G_c \ [mJ \ mm^{-2}]$"
        axeY = "$\ell_0 \ [mm]$"

        fig = plt.figure()
        ax1 = fig.add_subplot(projection="3d")
        cc = ax1.plot_surface(GC, L0, results, cmap='jet')
        fig.colorbar(cc, ticks=ticks)
        ax1.set_xlabel(axeX, fontsize=14)
        ax1.set_ylabel(axeY, fontsize=14)
        ax1.set_title("$J^2$", fontsize=14)
        ax1.scatter(GC[argmin], L0[argmin], results[argmin], c='red', marker='.', zorder=10)
        # ax1.contour(GC, L0, results)        
        
        # for gc in Gc_array:
        #     ecart, fr = DoSimu(np.array([gc]))
        #     ax1.scatter(gc, l00, ecart, c='black')
        #     plt.pause(1e-12)

        Display.Save_fig(folder_save, "J surface")

        ax2 = plt.subplots()[1]        
        cc = ax2.contourf(GC, L0, results,levels,  cmap='jet')
        ax2.set_xlabel(axeX, fontsize=14)
        ax2.set_ylabel(axeY, fontsize=14)
        ax2.set_title("$J^2$", fontsize=14)
        ax2.scatter(GC[argmin], L0[argmin], 200, c='red', marker='.', zorder=10,edgecolors='white')
        ax2.figure.colorbar(cc, ticks=ticks)

        Display.Save_fig(folder_save, "J contourf")

        # Display.Save_fig(folder_Save, "J_grid")
        # plt.show()

        pass



    plt.close('all')
