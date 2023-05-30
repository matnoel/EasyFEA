import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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

idxEssai = 4

nL = 100
tolConv = 1e-0

split = "AnisotStress"
# split = "He"
# split = "Zhang"

useContact = False
test = True
optimMesh = True

folderSource = Folder.New_File(Folder.Join(["Essais FCBA","Simu", f"Essai{idxEssai}"]), results=True)    

folder_Save = Folder.PhaseField_Folder(folderSource, "", split, "", "", tolConv, "", test, optimMesh, nL=nL)

pltLoad = True
pltIter = True
pltContact = False

GcHeterogene = False

# inc0 = 1e-2
# inc1 = 1e-2/3

inc0 = 1e-2/2
inc1 = inc0/4

h = 90
l = 45
ep = 20
d = 10

l0 = l/nL

meshSize = l0 if test else l0/2

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
# f_crit = dfLoadMax["Load [kN]"][idxEssai]
f_crit = 10
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

k_exp, __ = Calc_a_b(forces, deplacements, 15)

# ----------------------------------------------
# Mesh
# ----------------------------------------------

if optimMesh:
    epRefine = d
    refineGeom = Domain(Point(l/2-epRefine), Point(l/2+epRefine, h), meshSize)
    meshSize *= 3
else:
    refineGeom = None

domain = Domain(Point(), Point(l, h), meshSize)
circle = Circle(Point(l/2, h/2), d, meshSize)

mesh = Interface_Gmsh().Mesh_2D(domain, [circle], "TRI3", refineGeom=refineGeom)
xn = mesh.coordo[:,0]
yn = mesh.coordo[:,1]

nodes_Lower = mesh.Nodes_Tags(["L0"])
nodes_Upper = mesh.Nodes_Tags(["L2"])
nodes0 = mesh.Nodes_Tags(["P0"])
nodes_Boundary = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])

ddlsX_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_Upper, ["x"])
ddlsY_Upper = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_Upper, ["y"])

Affichage.Plot_Mesh(mesh)
# Affichage.Plot_Model(mesh)

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

# Calcul de la pente numérique
simuElas = Simulations.Simu_Displacement(mesh, comp)
simuElas.add_dirichlet(nodes_Lower, [0,0], ["x","y"])
simuElas.add_surfLoad(nodes_Upper, [-f_crit*1000/l/ep], ["y"])
u_num = simuElas.Solve()
fr_num = - np.sum(simuElas.Get_K_C_M_F()[0][ddlsY_Upper] @ u_num)/1000

k_mat, __ = Calc_a_b(np.linspace(0, fr_num, 50), np.linspace(0, -np.mean(u_num[ddlsY_Upper]), 50), f_crit)

k_montage = 1/(1/k_exp - 1/k_mat)

if pltLoad:
    axLoad = plt.subplots()[1]
    axLoad.plot(deplacements, forces, label="exp")
    axLoad.set_xlabel("displacement [mm]")
    axLoad.set_ylabel("load [kN]")

    deplMat = np.linspace(0, -np.mean(u_num[ddlsY_Upper]), 20)
    forcesMat = np.linspace(forces[0], fr_num, 20)
    # axLoad.scatter(deplMat, forcesMat, label="mat", marker=".", c="black", zorder=10)    

    coef_a = k_mat/k_exp    
    # axLoad.plot(deplacements/coef_a, forces, label="(1)")    
       
    deplacements = deplacements-forces/k_montage
    # axLoad.scatter(deplacements[idx_crit], forces[idx_crit], marker='+', c='red', zorder=2)
    axLoad.plot(deplacements, forces, label="(2)")
    axLoad.legend()
    axLoad.grid()
    Affichage.Save_fig(folder_Save, "load")

if GcHeterogene:

    coord_e = np.mean(mesh.coordo[mesh.connect], axis=1)
    x_e = coord_e[:,0]

    Nc = 10
    dc = l/Nc

    l1 = l0*2; elems1 = []
    l2 = l1*4; elems2 = []

    ax = Affichage.Plot_Mesh(mesh, alpha=0)

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

# Gc = 1e-2 # -> 32
# # Gc = 42 * 1e-2/32

# # 1e-2 -> 22.86724860286298
# #      -> 39.356
# Gc = 39.565*1e-2/18.1109745566610
# Gc *= 2

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

pfm = Materials.PhaseField_Model(comp, split, "AT2", Gc, l0, A=A)

simu = Simulations.Simu_PhaseField(mesh, pfm)

damageMax = []
list_fr = []
list_dep = []

if pltContact:
    axContact = plt.subplots()[1]
    axContact.set_xlabel("xn [mm]"), axContact.set_ylabel("f [kN]")
    idxSort = np.argsort(xn[nodes_Upper])

dep = -inc0
fr = 0
i = -1
while fr <= f_max*1.05:

    i += 1
    dep += inc0 if simu.damage.max()<=0.6 else inc1

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

    u, d, Kglob, convergence = simu.Solve(tolConv, convOption=1)

    damageMax.append(np.max(d))    

    simu.Resultats_Set_Resume_Iteration(i, dep, "mm", dep/dep_crit, True)

    simu.Save_Iteration()

    f = Kglob @ u

    f_Upper = f[ddlsY_Upper]

    fr = - np.sum(f_Upper)/1000

    if fr != -0.0 and pltContact:
        # Affichage.Plot_Result(simu, f.reshape(-1,2)[:,1])
        # ax = Affichage.Plot_Mesh(simu, alpha=0)    
        # ax.quiver(xn[nodes_Upper], yn[nodes_Upper], f[ddlsX_Upper]*5/fr, f[ddlsY_Upper]*5/fr, color='red', width=1e-3)

        axContact.plot(mesh.coordo[nodes_Upper[idxSort],0], f_Upper[idxSort]/1000)
        plt.figure(axContact.figure)
        pass

    list_fr.append(fr)
    list_dep.append(dep)

    if pltLoad:
        plt.figure(axLoad.figure)
        axLoad.scatter(dep, fr, c='black', marker='.')
        # if fr >= f_crit:
        if np.max(d) >= 0.9:
            axLoad.scatter(dep, fr, c='red', marker='.')
        plt.pause(1e-12)

    if pltIter:
        if i == 0:
            _, axIter, cbIter = Affichage.Plot_Result(simu, "damage")
        else:
            cbIter.remove()
            _, axIter, cbIter = Affichage.Plot_Result(simu, "damage", ax=axIter)

        plt.figure(axIter.figure)        
        
    plt.pause(1e-12)    

    if not convergence or True in (d[nodes_Boundary] >= 0.98):
        print("\nPas de convergence")
        break

damageMax = np.array(damageMax)
list_fr = np.array(list_fr)

fDamageSimu = list_fr[np.where(damageMax >= 0.95)[0][0]]

 
if pltLoad:
    plt.figure(axLoad.figure)
    Affichage.Save_fig(folder_Save, "forcedep")

if pltIter:    
    plt.figure(axIter.figure)
    Affichage.Save_fig(folder_Save, "damage")

PostTraitement.Make_Paraview(folder_Save, simu)

Affichage.Plot_ResumeIter(simu, folder_Save)

plt.show()