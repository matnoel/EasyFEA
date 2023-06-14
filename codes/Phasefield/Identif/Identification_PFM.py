import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
import os
import pandas as pd

import Folder
import Affichage
from Interface_Gmsh import Interface_Gmsh
from Geom import Point, Domain, Circle
import Materials
import Simulations
import PostTraitement

Affichage.Clear()

folder_file = Folder.Get_Path(__file__)

# ----------------------------------------------
# Config
# ----------------------------------------------

test = True
useContact = False
detectL0 = False
optimMesh = True

solveur = 0 # least_squares
# solveur = 1 # minimize
# solveur = 2 # regle de 3

# split = "AnisotStress"
# split = "He"
split = "Zhang"

# tolConv = 1e-0
tolConv = 1e-3

# Affichage
pltLoad = True
pltIter = True
pltContact = False

doSimulation = True

# inc0 = 8e-3 # incrément platewith hole
# inc1 = 2e-3
inc0 = 1e-2/2
inc1 = inc0/3

h = 90
l = 45
ep = 20
d = 10

Gc0 = 0.07
GcMax = 2

nL = 100
l00 = l/nL

# ftol = 1e-12
ftol = 1e-4

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

for idxEssai in range(18):

    # Dossier de l'essai
    essai = f"Essai{idxEssai}"
    folderIdentif = Folder.New_File(Folder.Join(["Essais FCBA", "Identification"]), results=True)
    folder_Save = Folder.Join([folderIdentif, essai])
    if test:
        folder_Save = Folder.Join([folder_Save, "Test"])

    if not doSimulation:
        simu = Simulations.Load_Simu(folder_Save)
        Affichage.Plot_Mesh(simu.mesh)
        Affichage.Plot_Result(simu, "damage")
        plt.show()
        continue
    
    # ----------------------------------------------   
    # Données de l'essai
    # ----------------------------------------------   

    forces = dfLoad["forces"][idxEssai]
    deplacements = dfLoad["deplacements"][idxEssai]

    f_max = np.max(forces)
    f_crit = dfLoadMax["Load [kN]"][idxEssai]
    print(f"fcrit = {f_crit}")
    # f_crit = 10
    idx_crit = np.where(forces >= f_crit)[0][0]
    dep_crit = deplacements[idx_crit]

    
    El = dfParams["El"][idxEssai]
    Et = dfParams["Et"][idxEssai]
    Gl = dfParams["Gl"][idxEssai]
    # vl = dfParams["vl"][idxEssai]
    vl = 0.02
    vt = 0.44

    # axis_l = np.array([0,1,0])
    # axis_t = np.array([1,0,0])

    rot = 90 * np.pi/180
    axis_l = np.array([np.cos(rot), np.sin(rot), 0])

    matRot = np.array([ [np.cos(np.pi/2), -np.sin(np.pi/2), 0],
                        [np.sin(np.pi/2), np.cos(np.pi/2), 0],
                        [0, 0, 1]])

    axis_t = matRot @ axis_l

    comp = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axis_l, axis_t, True, ep)

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------    

    def DoMesh(l0: float):

        meshSize = l0 if test else l0/2

        if optimMesh:
            epRefine = d
            refineGeom = Domain(Point(l/2-epRefine), Point(l/2+epRefine, h), meshSize)
            meshSize *= 3
        else:
            refineGeom = None

        domain = Domain(Point(), Point(l, h), meshSize)
        circle = Circle(Point(l/2, h/2), d, meshSize)

        mesh = Interface_Gmsh().Mesh_2D(domain, [circle], "TRI3", refineGeom=refineGeom)

        return mesh

    mesh = DoMesh(l00)
    # Affichage.Plot_Mesh(mesh)
    # Affichage.Plot_Model(mesh)

    # ----------------------------------------------
    # phase field
    # ----------------------------------------------

    a1 = np.array([1,0])
    a2 = axis_t[:2]
    M1 = np.einsum("i,j->ij", a1, a1)

    a2 = np.array([0,1])
    a2 = axis_l[:2]
    M2 = np.einsum("i,j->ij", a2, a2)

    A = np.eye(2)

    # coef = Et/El
    # A += 0 * M1 + 1/coef * M2
    # A += + 1/coef * M1 + 0 * M2
    # A = np.array([[coef, 0],[0, 1-coef]])
    # A = np.array([[1-coef, 0],[0, coef]])

    # ----------------------------------------------
    # Simu
    # ----------------------------------------------

    dCible = 1

    returnSimu = False

    axEcart = plt.subplots()[1]
    axEcart.set_xlabel("iter"); axEcart.set_ylabel("ecart")

    ecarts = []

    def DoSimu(x) -> float:

        Gc = x[0]

        if detectL0:
            l0 = x[1]
            # reconstruit le maillage
            meshSimu = DoMesh(l0)

        else:
            l0 = l00
            meshSimu = mesh
            
        yn = meshSimu.coordo[:, 1]
        nodes_Lower = meshSimu.Nodes_Tags(["L0"])
        nodes_Upper = meshSimu.Nodes_Tags(["L2"])
        nodes0 = meshSimu.Nodes_Tags(["P0"])    
        ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_Upper, ["y"])
        
        # construit le modèle d'endommagement
        pfm = Materials.PhaseField_Model(comp, split, "AT2", Gc, l0, A=A)
        
        simu = Simulations.Simu_PhaseField(meshSimu, pfm)

        dep = -inc0

        if detectL0:
            print(f"\nGc = {x[0]:.10e}, l0 = {x[1]:.5e}")
        else:
            print(f"\nGc = {x[0]:.10e}")

        i = -1
        fr = 0
        # while fr <= f_max*2:
        while True:

            i += 1
            # dep += inc0 if simu.damage.max() <= 0.6 else inc1
            dep += inc0 if simu.damage.max() <= 0.2 else inc1

            simu.Bc_Init()
            simu.add_dirichlet(nodes_Lower, [0], ["y"])
            simu.add_dirichlet(nodes0, [0], ["x"])

            if useContact:
                frontiere = 90 - dep

                yn_c = yn[nodes_Upper] + simu.displacement[ddlsY_Upper]

                ecart = frontiere - yn_c

                idxContact = np.where(ecart < 0)[0]

                if len(idxContact) > 0:
                    simu.add_dirichlet(nodes_Upper[idxContact], [ecart[idxContact]], ["y"])        
                
            else:
                simu.add_dirichlet(nodes_Upper, [-dep], ["y"])

            u, d, Kglob, convergence = simu.Solve(tolConv, convOption=2)

            simu.Save_Iteration()

            f = Kglob[ddlsY_Upper,:] @ u        

            fr = - np.sum(f)/1000

            simu.Resultats_Set_Resume_Iteration(i, fr, "kN", fr/f_crit, True)

            # if not convergence or True in (d[nodes_Boundary] >= 0.98):
            #     print("\nPas de convergence")
            #     break        

            if d.max() >= dCible:
                break

        list_Gc.append(x)

        if solveur in [0,1]:        
            ecart = (fr - f_crit)/f_crit
            # ecart = (fr - f_crit)
            print(f"\necart = {ecart:.10e}")
        
        if solveur == 2:
            ecart = fr

        if returnSimu:
            return simu
        else:

            ecarts.append(ecart)
            Niter = len(ecarts)
                    
            axEcart.scatter(Niter, ecarts[-1], c="black")
            plt.figure(axEcart.figure)
            plt.pause(1e-12)

            return ecart

    # ----------------------------------------------
    # Optimization
    # ----------------------------------------------

    list_Gc = []
    
    lb = [0] if not detectL0 else [0, 0]
    ub = [GcMax] if not detectL0 else [GcMax, np.inf]
    x0 = [Gc0] if not detectL0 else [Gc0, l00]

    if solveur in [0,1]:
        
        if solveur == 0:
            res = least_squares(DoSimu, x0, bounds=(lb, ub), verbose=0, ftol=ftol)
        elif solveur == 1:
            # res = minimize(DoSimu, x0, tol=tol)

            bounds = [(l, u) for l, u in zip(lb, ub)]

            res = minimize(DoSimu, x0, bounds=bounds, tol=ftol)

        Gc = res.x[0]
        if detectL0:
            l0 = res.x[1]
            print(f"Gc = {Gc:.10e}, l0 = {l0:.4e}")
        else:
            l0 = l00
            print(f"Gc = {Gc:.10e}")

        x = res.x

    elif solveur == 2:

        ecart = 1
        Gc = Gc0
        while ecart >= ftol:
            fr = DoSimu([Gc])

            ecart = (f_crit - fr)/f_crit

            print(f"\nGc = {Gc:.4f}, ecart = {ecart:.5e}")
            Gc = f_crit * Gc/fr

        x = [Gc,l00]

    # ----------------------------------------------
    # PostTraitement
    # ----------------------------------------------

    returnSimu = True
    simu = DoSimu(x)

    assert isinstance(simu, Simulations.Simu_PhaseField)

    deplacementsIdentif = []
    forcesIdentif = []

    for iter in range(len(simu._results)):

        simu.Update_iter(iter)

        displacement = simu.displacement
        ddlsY = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", simu.mesh.Nodes_Conditions(lambda x,y,z: y==h), ["y"])

        deplacementsIdentif.append(-np.mean(displacement[ddlsY]))
        forcesIdentif.append(-np.sum(simu.Get_K_C_M_F()[0][ddlsY,:] @ displacement)/1000)

    deplacementsIdentif = np.asarray(deplacementsIdentif)
    forcesIdentif = np.asarray(forcesIdentif)

    axLoad = plt.subplots()[1]
    axLoad.set_xlabel("displacement [mm]")
    axLoad.set_ylabel("load [kN]")

    axLoad.grid()

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

    axLoad.plot(deplacements, forces, label="exp")
    axLoad.scatter(deplacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=3)
    axLoad.plot(deplacementsIdentif, forcesIdentif, label="identif")
    axLoad.legend()
    Affichage.Save_fig(folder_Save, "load")    

    # Sauvegarde les données identifiées

    pathData = Folder.Join([folderIdentif, "identification.xlsx"])

    data = [
        {
            "Essai": essai,
            "split": simu.phaseFieldModel.split,
            "regu": simu.phaseFieldModel.regularization,        
            "A": A,
            "tolConv": tolConv,
            "optimMesh": optimMesh,
            "solveur": solveur,
            "ftol": ftol,
            "detectL0": detectL0,
            "f_crit": f_crit,
            "fr": forcesIdentif[-1],
            "err": np.abs(f_crit-forcesIdentif[-1])/f_crit,
            "Gc": Gc,
            "l0": l0
        }
    ]

    if os.path.exists(pathData):

        df = pd.read_excel(pathData)

        newDf = pd.DataFrame(data)
        df = pd.concat([df,newDf])        

    else:

        df = pd.DataFrame(data)

    df.to_excel(pathData, index=False)

    plt.figure(axEcart.figure)
    Affichage.Save_fig(folder_Save, "iterations")

    # PostTraitement.Make_Paraview(folder_Save, simu)

    simu.Save(folder_Save)

    del simu

    plt.close('all')