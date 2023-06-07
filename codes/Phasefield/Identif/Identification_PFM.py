import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

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

idxEssai = 1

nL = 100
# tolConv = 1e-0
tolConv = 1e-6

# option = 0 # least_squares
option = 1 # regle de 3

# split = "AnisotStress"
# split = "He"
split = "Zhang"

useContact = False
detectL0 = False
test = True
optimMesh = True

folder_Save = Folder.New_File(Folder.Join(["Essais FCBA", "Identification", f"Essai{idxEssai}"]), results=True)

if test:
    folder_Save = Folder.Join([folder_Save, "Test"])

pltLoad = True
pltIter = True
pltContact = False

# inc0 = 8e-3 # incrément platewith hole
# inc1 = 2e-3

# inc0 = 1e-2
# inc1 = 1e-2/3
# inc0 = 1e-2/3
# inc1 = inc0/3

inc0 = 1e-2/2
inc1 = inc0/3

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
print(f"fcrit = {f_crit}")
# f_crit = 10
idx_crit = np.where(forces >= f_crit)[0][0]
dep_crit = deplacements[idx_crit]

h = 90
l = 45
ep = 20
d = 10

# ----------------------------------------------
# Comportement
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

# ----------------------------------------------
# Mesh
# ----------------------------------------------

l00 = l/nL

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

Gc0 = 0.07
# Gc0 = 6.4339e-02
# Gc0 = 0.0758867295417314
# 0.0719550

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
        print(f"\nGc = {x[0]:.5f}, l0 = {x[1]:.5f}")
    else:
        print(f"\nGc = {x[0]:.5f}")

    i = -1
    fr = 0
    while fr <= f_max*1.05:

        i += 1
        dep += inc0 if simu.damage.max() <= 0.6 else inc1

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

    if option == 0:
        ecart = (f_crit - fr)/f_crit
        # ecart = f_crit - fr

        print(f"\necart = {ecart:.4e}")
    
    elif option == 1:
        ecart = fr

    if returnSimu:
        return simu
    else:
        return ecart

list_Gc = []

tol = 1e-12

if option == 0:
    
    GcMax = 2
    lb = [0] if not detectL0 else [0, 0]
    ub = [GcMax] if not detectL0 else [GcMax, np.inf]
    x0 = [Gc0] if not detectL0 else [Gc0, l00]

    res = least_squares(DoSimu, x0, bounds=(lb, ub), verbose=0, ftol=tol)
    
    # res = minimize(DoSimu, x0, tol=tol)
    # res = minimize(DoSimu, x0, method="trust-constr", bounds=(lb, ub), tol=tol)
    if detectL0:
        print(f"Gc = {res.x[0]:.4e}, l0 = {res.x[1]:.4e}")
    else:
        print(f"Gc = {res.x[0]:.4e}")

    x = res.x

elif option == 1:

    ecart = 1
    Gc = Gc0
    while ecart >= tol:
        fr = DoSimu([Gc])

        ecart = np.abs(f_crit - fr)/f_crit

        print(f"\nGc = {Gc:.4f}, ecart = {ecart:.5e}")
        Gc = f_crit * Gc/fr

    x = [Gc,l00]

# ----------------------------------------------
# PostTraitement
# ----------------------------------------------

returnSimu = True
simu = DoSimu(x)

assert isinstance(simu, Simulations.Simu_PhaseField)

PostTraitement.Make_Paraview(folder_Save, simu)

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

simu.Save(folder_Save)


plt.show()