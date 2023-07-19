import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

import Folder
import Display
from Mesh import Mesh
from Interface_Gmsh import Interface_Gmsh
from Geom import Point, Domain, Circle
import Materials
import Simulations
import PostTraitement

Display.Clear()

folder_file = Folder.Get_Path(__file__)

# ----------------------------------------------
# Config
# ----------------------------------------------

doSimulation = True

test = True
optimMesh = True
useContact = False
GcHeterogene = False

pltLoad = True
pltIter = True
pltContact = False
makeParaview = False
makeMovie = False

idxEssai = 4

nL = 100

solveur = 0 # least_squares
# solveur = 1 # minimize
# solveur = 2 # regle de 3

split = "AnisotStress"
# split = "He"
# split = "Zhang"
regu = "AT2"

# tolConv = 1e-0
# tolConv = 1e-1
tolConv = 1e-2

# convOption = 0 # bourdin
# convOption = 1 # energie crack
convOption = 2 # energie tot

folderSource = Folder.New_File(Folder.Join(["Essais FCBA","Simu", f"Essai{idxEssai}"]), results=True)    

folder_Save = Folder.PhaseField_Folder(folderSource, "", split, regu, "", tolConv, "", test, optimMesh, nL=nL)

pathSimu = Folder.Join([folder_Save, "simulation.pickle"])
if not os.path.exists(pathSimu) and not doSimulation:
    print(folder_Save)
    print("la simulation n'exsite pas")
    doSimulation = True

# inc0 = 8e-3 # incréments utilisés pour platwith hole
# inc1 = 2e-3

# inc0 = 1e-2
# inc1 = 1e-2/3

inc0 = 1e-2/2
inc1 = inc0/3

inc0 = 8e-6

inc0 = 8e-3; tresh0 = 0.2
inc1 = 2e-3; tresh1 = 0.6

h = 90
l = 45
ep = 20
d = 10

l0 = l/nL

if not doSimulation:
    simu = Simulations.Load_Simu(folder_Save)
    assert isinstance(simu, Simulations.Simu_PhaseField)

# ----------------------------------------------
# Forces & Déplacements
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

forces = dfLoad["forces"][idxEssai]
deplacements = dfLoad["deplacements"][idxEssai]

f_max = np.max(forces)
f_crit = dfLoadMax["Load [kN]"][idxEssai]
# f_crit = 10
idx_crit = np.where(forces >= f_crit)[0][0]
dep_crit = deplacements[idx_crit]

def Calc_a_b(forces, deplacements, fmax):
    """Calcul des coefs de f(x) = a x + b"""

    idxElas = np.where((forces <= fmax))[0]
    idx1, idx2 = idxElas[0], idxElas[-1]
    x1, x2 = deplacements[idx1], deplacements[idx2]
    f1, f2 = forces[idx1], forces[idx2]
    vect_ab = np.linalg.inv(np.array([[x1, 1],[x2, 1]])).dot(np.array([f1, f2]))
    a, b = vect_ab[0], vect_ab[1]

    return a, b

# calcul la raideur expérimentale
k_exp, __ = Calc_a_b(forces, deplacements, 15)

# ----------------------------------------------
# Mesh
# ----------------------------------------------

def DoMesh(l0: float) -> Mesh:

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


if doSimulation:
    mesh = DoMesh(l0)
else:
    mesh = simu.mesh


nodes_Lower = mesh.Nodes_Tags(["L0"])
nodes_Upper = mesh.Nodes_Tags(["L2"])
nodes0 = mesh.Nodes_Tags(["P0"])
nodes_Boundary = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

ddlsX_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_Upper, ["x"])
ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_Upper, ["y"])

Display.Plot_Mesh(mesh)
# Affichage.Plot_Nodes(mesh, nodes_Upper, True)

# ----------------------------------------------
# Comp and Simu
# ----------------------------------------------

# récupère les proritétés identifiées
pathParams = Folder.Join([folder_file, "params_Essais.xlsx"])

dfParams = pd.read_excel(pathParams)

# print(dfParams)

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
# comp = Materials.Elas_IsotTrans(2, El, El, Gl, vl, vl, axis_l, axis_t, True, ep)

# Calcul de la pente numérique
simuElas = Simulations.Simu_Displacement(mesh, comp)
simuElas.add_dirichlet(nodes_Lower, [0,0], ["x","y"])
simuElas.add_surfLoad(nodes_Upper, [-f_crit*1000/l/ep], ["y"])
u_num = simuElas.Solve()
fr_num = - np.sum(simuElas.Get_K_C_M_F()[0][ddlsY_Upper] @ u_num)/1000

k_mat, __ = Calc_a_b(np.linspace(0, fr_num, 50), np.linspace(0, -np.mean(u_num[ddlsY_Upper]), 50), f_crit)

k_montage = 1/(1/k_exp - 1/k_mat)

if GcHeterogene:

    coord_e = np.mean(mesh.coordo[mesh.connect], axis=1)
    x_e = coord_e[:,0]

    Nc = 10
    dc = l/Nc

    l1 = l0*2; elems1 = []
    l2 = l1*4; elems2 = []

    ax = Display.Plot_Mesh(mesh, alpha=0)

    x = 0
    i = 0

    elems1 = []; elems2 = []

    while x < l:
        i += 1
        ll = l1 if i % 2 == 0 else l2
        b1 = x
        b2 = x + ll
        x += ll
        elems = np.where((x_e-1e-12 >= b1) & (x_e+1e-12 <= b2))[0]

        if i % 2 == 0:
            elems1.extend(elems)
            ax.scatter(coord_e[elems,0], coord_e[elems,1], c="red")
        else:
            elems2.extend(elems)
            ax.scatter(coord_e[elems,0], coord_e[elems,1], c="blue")
        pass

    Gc = np.ones(mesh.Ne) * 0.05*10
    Gc[elems1] = 0.03

else:

    Gc = 0.07    
    # Gc = 0.05
    # Gc = 0.01

vectFibre = axis_t[:2]
M = np.einsum("i,j->ij", vectFibre, vectFibre)

Kic_L = 410 # RL kPa m^(1/2)
Kic_T = 260 # TL 

# coefK = 1e-3 * 1e6 * 1000**(1/2) # kPa m^(1/2) -> MPa mm^(1/2)
# Kic_L *= coefK
# Kic_T *= coefK
Betha = El/Et
Betha = 0
A = np.eye(2) + Betha * (np.eye(2) - M)

# A = np.eye(2)

# coef = Et/El
# A += 0 * M1 + 1/coef * M2
# A += + 1/coef * M1 + 0 * M2
# A = np.array([[coef, 0],[0, 1-coef]])
# A = np.array([[1-coef, 0],[0, coef]])

pfm = Materials.PhaseField_Model(comp, split, regu, Gc, l0, A=A)

if doSimulation:
    simu = Simulations.Simu_PhaseField(mesh, pfm)
else:
    pfm = simu.phaseFieldModel
    comp = pfm.comportement

damageMax = []
list_fr = []
list_dep = []

if doSimulation:

    if pltLoad:
        axLoad = plt.subplots()[1]
        # axLoad.plot(deplacements, forces, label="exp")
        axLoad.set_xlabel("x [mm]")
        axLoad.set_ylabel("f [kN]")

        deplMat = np.linspace(0, -np.mean(u_num[ddlsY_Upper]), 20)
        forcesMat = np.linspace(forces[0], fr_num, 20)
        # axLoad.scatter(deplMat, forcesMat, label="mat", marker=".", c="black", zorder=10)

        # coef_a = k_mat/k_exp    
        # axLoad.plot(deplacements/coef_a, forces, label="(1)")    
        
        deplacements = deplacements-forces/k_montage
        axLoad.scatter(deplacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=10)
        axLoad.plot(deplacements, forces, label="redim")
        # axLoad.legend()
        axLoad.grid()
        Display.Save_fig(folder_Save, "load")

    
    if pltContact:
        xn = mesh.coordo[:,0]
        yn = mesh.coordo[:,1]
        axContact = plt.subplots()[1]
        axContact.set_xlabel("xn [mm]"), axContact.set_ylabel("f [kN]")
        idxSort = np.argsort(xn[nodes_Upper])

    dep = -inc0
    fr = 0
    i = -1

    fStop = f_max*1.1
    fStop = f_crit*1.2

    while fr <= fStop:

        i += 1
        dep += inc0 if simu.damage.max() <= tresh0 else inc1

        simu.Bc_Init()
        simu.add_dirichlet(nodes_Lower, [0], ["y"])
        simu.add_dirichlet(nodes0, [0], ["x"])

        if useContact:
            frontiere = 90 - dep # coordonnée y du plan maitre

            yn_c = simu.mesh.coordo[nodes_Upper,1] + simu.displacement[ddlsY_Upper] # coordonnées y du plan esclave
            # yn_c = simu.mesh.coordo[nodes_Upper,1] # coordonnées y du plan esclave

            ecart = frontiere - yn_c

            idxContact = np.where(ecart < 0)[0]

            if len(idxContact) > 0:
                simu.add_dirichlet(nodes_Upper[idxContact], [ecart[idxContact]], ["y"])                

            # axContact.clear()
            # axContact.plot([-10, l+10], [frontiere, frontiere])
            # axContact.scatter(simu.mesh.coordo[nodes_Upper,0], yn_c)
            
        else:

            simu.add_dirichlet(nodes_Upper, [-dep], ["y"])

            # simu.add_dirichlet(nodes_Upper, [-10/(90*45)], ["y"])

        # Affichage.Plot_BoundaryConditions(simu)

        u, d, Kglob, convergence = simu.Solve(tolConv, convOption=2)

        damageMax.append(np.max(d))

        simu.Save_Iteration()

        f = Kglob @ u

        f_Upper = f[ddlsY_Upper]

        fr = - np.sum(f_Upper)/1000

        simu.Resultats_Set_Resume_Iteration(i, fr, "kN", fr/fStop, True)

        # if fr != -0.0 and pltContact:
        #     Affichage.Plot_Result(simu, f.reshape(-1,2)[:,1])
        #     ax = Affichage.Plot_Mesh(simu, alpha=0)    
        #     ax.quiver(xn[nodes_Upper], yn[nodes_Upper], f[ddlsX_Upper]*5/fr, f[ddlsY_Upper]*5/fr, color='red', width=1e-3)

        #     axContact.plot(mesh.coordo[nodes_Upper[idxSort],0], f_Upper[idxSort]/1000)
        #     plt.figure(axContact.figure)
        #     pass

        list_fr.append(fr)
        list_dep.append(dep)

        if pltLoad:

            depp = -np.mean(simu.displacement[ddlsY_Upper]) if useContact else dep

            plt.figure(axLoad.figure)
            axLoad.scatter(depp, fr, c='black', marker='.')        
            # axLoad.scatter(depp, fr, marker='.')        
            if np.max(d) >= 1:
                axLoad.scatter(depp, fr, c='red', marker='.')
            plt.pause(1e-12)

        if pltIter:
            if i == 0:
                _, axIter, cbIter = Display.Plot_Result(simu, "damage")
            else:
                cbIter.remove()
                _, axIter, cbIter = Display.Plot_Result(simu, "damage", ax=axIter)

            plt.figure(axIter.figure)        
            
        plt.pause(1e-12)    

        if not convergence or True in (d[nodes_Boundary] >= 0.98):
            print("\nPas de convergence")
            break

    damageMax = np.array(damageMax)
    list_fr = np.array(list_fr)
    list_dep = np.array(list_dep)

    PostTraitement.Save_Load_Displacement(list_fr, list_dep, folder_Save)

    fDamageSimu = list_fr[np.where(damageMax >= 0.95)[0][0]]
    
    if pltLoad:
        plt.figure(axLoad.figure)
        Display.Save_fig(folder_Save, "forcedep")

    Display.Plot_ResumeIter(simu, folder_Save)

    simu.Save(folder_Save)

Display.Plot_Result(simu, 'damage', folder=folder_Save, colorbarIsClose=True)

if makeParaview:
    PostTraitement.Make_Paraview(folder_Save, simu)

if makeMovie:
    PostTraitement.Make_Movie(folder_Save, "damage", simu)



plt.show()