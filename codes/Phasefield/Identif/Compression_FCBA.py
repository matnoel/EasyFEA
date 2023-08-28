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
import PostProcessing

Display.Clear()

folder_file = Folder.Get_Path(__file__)

# ----------------------------------------------
# Configuration
# ----------------------------------------------
idxEssai = 4

solve = True
test = True
optimMesh = True
useContact = False

# geom
h = 90
l = 45
ep = 20
d = 10

nL = 100
l0 = l/nL

# posProcessing
pltLoad = True
pltIter = True
pltContact = False
makeParaview = False
makeMovie = False

# phase field
split = "AnisotStress" # he, Zhang
regu = "AT2"
tolConv = 1e-2 # 1e-0, 1e-1, 1e-2
convOption = 2
# (0, bourdin)
# (1, crack energy)
# (2, crack + strain energy)

folder_essai = Folder.New_File(Folder.Join(["Essais FCBA","Simu", f"Essai{idxEssai}"]), results=True)
folder_save = Folder.PhaseField_Folder(folder_essai, "", split, regu, "", tolConv, "", test, optimMesh, nL=nL)

pathSimu = Folder.Join([folder_save, "simulation.pickle"])
if not os.path.exists(pathSimu) and not solve:
    print(folder_save)
    print("la simulation n'existe pas")
    solve = True

# ----------------------------------------------
# Loading
# ----------------------------------------------
inc0 = 8e-3; tresh0 = 0.2
inc1 = 2e-3; tresh1 = 0.6

if not solve:
    simu = Simulations.Load_Simu(folder_save)
    assert isinstance(simu, Simulations.Simu_PhaseField)

# ----------------------------------------------
# Import Loading
# ----------------------------------------------
# recovers force-displacement curves
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
displacements = dfLoad["deplacements"][idxEssai]

f_max = np.max(forces)
f_crit = dfLoadMax["Load [kN]"][idxEssai]
# f_crit = 10
idx_crit = np.where(forces >= f_crit)[0][0]
dep_crit = displacements[idx_crit]

def Calc_a_b(forces, deplacements, fmax):
    """Calculating the coefficients of f(x) = a x + b"""

    idxElas = np.where((forces <= fmax))[0]
    idx1, idx2 = idxElas[0], idxElas[-1]
    x1, x2 = deplacements[idx1], deplacements[idx2]
    f1, f2 = forces[idx1], forces[idx2]
    vect_ab = np.linalg.inv(np.array([[x1, 1],[x2, 1]])).dot(np.array([f1, f2]))
    a, b = vect_ab[0], vect_ab[1]

    return a, b

# calculation of experimental stiffness
k_exp, __ = Calc_a_b(forces, displacements, 15)

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


if solve:
    mesh = DoMesh(l0)
else:
    mesh = simu.mesh

nodes_lower = mesh.Nodes_Tags(["L0"])
nodes_upper = mesh.Nodes_Tags(["L2"])
nodes0 = mesh.Nodes_Tags(["P0"])
nodes_edges = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

dofsX_upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_upper, ["x"])
dofsY_upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_upper, ["y"])

Display.Plot_Mesh(mesh)

# ----------------------------------------------
# Material
# ----------------------------------------------
# recovers identified properties
pathParams = Folder.Join([folder_file, "params_Essais.xlsx"])
dfParams = pd.read_excel(pathParams)
# print(dfParams)

El = dfParams["El"][idxEssai]
Et = dfParams["Et"][idxEssai]
Gl = dfParams["Gl"][idxEssai]
# vl = dfParams["vl"][idxEssai]
vl = 0.02
vt = 0.44

# set axis
rot = np.pi/2
axis_l = np.array([np.cos(rot), np.sin(rot), 0])
axis_t = np.cross(np.array([0,0,1]), axis_l)

material = Materials.Elas_IsotTrans(2, El, Et, Gl, vl, vt, axis_l, axis_t, True, ep)

# Numerical slope calculation
simuElas = Simulations.Simu_Displacement(mesh, material)
simuElas.add_dirichlet(nodes_lower, [0,0], ["x","y"])
simuElas.add_surfLoad(nodes_upper, [-f_crit*1000/l/ep], ["y"])
u_num = simuElas.Solve()
fr_num = - np.sum(simuElas.Get_K_C_M_F()[0][dofsY_upper] @ u_num)/1000
k_mat, __ = Calc_a_b(np.linspace(0, fr_num, 50), np.linspace(0, -np.mean(u_num[dofsY_upper]), 50), f_crit)

k_montage = 1/(1/k_exp - 1/k_mat)

Gc = 0.07 # mJ/mm2

fibreVector = axis_t[:2]
M = np.einsum("i,j->ij", fibreVector, fibreVector)
Betha = El/Et
Betha = 0
A = np.eye(2) + Betha * (np.eye(2) - M)

pfm = Materials.PhaseField_Model(material, split, regu, Gc, l0, A=A)

if solve:
    simu = Simulations.Simu_PhaseField(mesh, pfm)
else:
    pfm = simu.phaseFieldModel
    material = pfm.material

damageMax = []
list_fr = []
list_dep = []

if solve:

    if pltLoad:
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
        axLoad.scatter(displacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=10)
        axLoad.plot(displacements, forces, label="redim")
        # axLoad.legend()
        axLoad.grid()
        Display.Save_fig(folder_save, "load")
    
    if pltContact:
        xn = mesh.coordo[:,0]
        yn = mesh.coordo[:,1]
        axContact = plt.subplots()[1]
        axContact.set_xlabel("xn [mm]"), axContact.set_ylabel("f [kN]")
        idxSort = np.argsort(xn[nodes_upper])

    dep = -inc0
    fr = 0
    i = -1
    
    fStop = f_crit*1.2

    while fr <= fStop:

        i += 1
        dep += inc0 if simu.damage.max() <= tresh0 else inc1

        simu.Bc_Init()
        simu.add_dirichlet(nodes_lower, [0], ["y"])
        simu.add_dirichlet(nodes0, [0], ["x"])

        if useContact:
            frontiere = 90 - dep # coordonnée y du plan maitre

            yn_c = simu.mesh.coordo[nodes_upper,1] + simu.displacement[dofsY_upper] # coordonnées y du plan esclave
            # yn_c = simu.mesh.coordo[nodes_Upper,1] # coordonnées y du plan esclave

            ecart = frontiere - yn_c

            idxContact = np.where(ecart < 0)[0]

            if len(idxContact) > 0:
                simu.add_dirichlet(nodes_upper[idxContact], [ecart[idxContact]], ["y"])                

            # axContact.clear()
            # axContact.plot([-10, l+10], [frontiere, frontiere])
            # axContact.scatter(simu.mesh.coordo[nodes_Upper,0], yn_c)
            
        else:

            simu.add_dirichlet(nodes_upper, [-dep], ["y"])        

        # solve and save iter
        u, d, Kglob, convergence = simu.Solve(tolConv, convOption=2)
        simu.Save_Iter()

        damageMax.append(np.max(d))

        f = Kglob @ u

        f_Upper = f[dofsY_upper]

        fr = - np.sum(f_Upper)/1000

        simu.Results_Set_Iteration_Summary(i, fr, "kN", fr/fStop, True)

        # if fr != -0.0 and pltContact:
        #     Display.Plot_Result(simu, f.reshape(-1,2)[:,1])
        #     ax = Display.Plot_Mesh(simu, alpha=0)    
        #     ax.quiver(xn[nodes_Upper], yn[nodes_Upper], f[ddlsX_Upper]*5/fr, f[ddlsY_Upper]*5/fr, color='red', width=1e-3)

        #     axContact.plot(mesh.coordo[nodes_Upper[idxSort],0], f_Upper[idxSort]/1000)
        #     plt.figure(axContact.figure)
        #     pass

        list_fr.append(fr)
        list_dep.append(dep)

        if pltLoad:

            depp = -np.mean(simu.displacement[dofsY_upper]) if useContact else dep

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

        if not convergence or True in (d[nodes_edges] >= 0.98):
            print("\nPas de convergence")
            break

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

Display.Plot_Result(simu, 'damage', folder=folder_save, colorbarIsClose=True)

if makeParaview:
    PostProcessing.Make_Paraview(folder_save, simu)

if makeMovie:
    PostProcessing.Make_Movie(folder_save, "damage", simu)

plt.show()