from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Simulations
import Affichage
import Interface_Gmsh
import Materials
import Geom

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

pltVerif = False
useRescale = True

l = 45
h = 90
b = 20
d = 10

meshSize = l/15
elemType = "TRI3"

mat = "bois" # "acier" "bois"

tol = 1e-14

sig = 10

# ----------------------------------------------
# Maillage
# ----------------------------------------------

gmshInterface = Interface_Gmsh.Interface_Gmsh()

pt1 = Geom.Point()
pt2 = Geom.Point(l, 0)
pt3 = Geom.Point(l, h)
pt4 = Geom.Point(0, h)
points = Geom.PointsList([pt1, pt2, pt3, pt4], meshSize)

circle = Geom.Circle(Geom.Point(l/2, h/2), d, meshSize, isCreux=True)

mesh = gmshInterface.Mesh_2D(points, elemType, [circle])

Affichage.Plot_Model(mesh)

nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodesp0 = mesh.Nodes_Tags(["P0"])
nodesBas = mesh.Nodes_Tags(["L0"])
nodesHaut = mesh.Nodes_Tags(["L2"])

ddlsX = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["x"])
ddlsY = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes, ["y"])

assert nodes.size*2 == (ddlsX.size + ddlsY.size)

if useRescale:
    ddlsBasX = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesBas, ["x"])
    ddlsBasY = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesBas, ["y"])
    
    ddlsHautX = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesHaut, ["x"])
    ddlsHautY = Simulations.BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodesHaut, ["y"])


Affichage.Plot_Mesh(mesh)
Affichage.Plot_Model(mesh)
# Affichage.Plot_Nodes(mesh, nodesX0)

# ----------------------------------------------
# Comportement
# ----------------------------------------------

tol0 = 1e-6
bSup = np.inf

if mat == "acier":
    E_exp, v_exp = 210000, 0.3
    comp = Materials.Elas_Isot(2, epaisseur=b)

    Emax=300000
    vmax=0.49
    E0, v0 = Emax, vmax
    x0 = [E0, v0]
    
    compIdentif = Materials.Elas_Isot(2, E0, v0, epaisseur=b)
    bounds=([tol0]*2, [bSup, vmax])
    
elif mat == "bois":
    EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3
    
    EL0 = EL_exp * 10
    GL0 = GL_exp * 10
    ET0 = ET_exp * 10
    vL0 = vL_exp

    EL0 = np.random.uniform(tol0, EL_exp*100)
    GL0 = np.random.uniform(tol0, GL_exp*100)
    ET0 = np.random.uniform(tol0, ET_exp*100)
    vL0 = np.random.uniform(tol0, 0.5-tol0)
    vL0 = 0.3

    lb = [tol0]*4
    ub = (bSup, bSup, bSup, 0.5-tol0)
    bounds = (lb, ub)
    x0 = [EL0, GL0, ET0, vL0]

    comp = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), contraintesPlanes=True, epaisseur=b)

    compIdentif = Materials.Elas_IsotTrans(2, El=EL0, Et=ET0, Gl=GL0, vl=vL0, vt=0.3,
    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), contraintesPlanes=True, epaisseur=b)

# ----------------------------------------------
# Simulation et chargement
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesBas, [0], ["y"])
simu.add_dirichlet(nodesp0, [0], ["x"])
simu.add_surfLoad(nodesHaut, [-sig], ["y"])

Affichage.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()

Affichage.Plot_Result(simu, "uy")
# Affichage.Plot_Result(simu, "Syy", coef=1/sig, nodeValues=False)
# Affichage.Plot_Result(simu, np.linalg.norm(vectRand.reshape((mesh.Nn), 2), axis=1), title="bruit")
# Affichage.Plot_Result(simu, u_exp.reshape((mesh.Nn,2))[:,1], title='uy bruit')
# simu.Resultats_Resume()

# ----------------------------------------------
# Identification
# ----------------------------------------------

Affichage.NouvelleSection("Identification")

# perturbations = [0, 0.02]
perturbations = np.linspace(0, 0.05, 6)

simuIdentif = Simulations.Simu_Displacement(mesh, compIdentif)

def func(x):
    # Fonction coût

    # Mise à jour des paramètres
    if mat == "acier":
        # x0 = [E0, v0]
        E = x[0]
        v = x[1]
        compIdentif.E = E
        compIdentif.v = v
    elif mat == "bois":
        # x0 = [EL0, GL0, ET0, vL0]
        compIdentif.El = x[0]
        compIdentif.Gl = x[1]
        compIdentif.Et = x[2]
        compIdentif.vl = x[3]

    simuIdentif.Need_Update()

    u = simuIdentif.Solve()

    # diff = u[ddlsInconnues] - u_exp_bruit[ddlsInconnues]
    diff = u - u_exp_bruit

    return diff

list_dict = []
# liste de dictionnaire qui va contenir pour les différentes perturbations les
# propriétés identifiées

for perturbation in perturbations:

    # bruitage de la solution
    bruit = np.abs(u_exp).max() * (np.random.rand(u_exp.shape[0]) - 1/2) * perturbation
    u_exp_bruit = u_exp + bruit

    if mat == "acier":
        compIdentif.E = E0
        compIdentif.v = v0
    elif mat == "bois":
        compIdentif.El = EL0
        compIdentif.Gl = GL0
        compIdentif.Et = ET0
        compIdentif.vl = vL0

    simuIdentif.Bc_Init()
    
    if useRescale:        
        simuIdentif.add_dirichlet(nodesBas, [u_exp_bruit[ddlsBasX], u_exp_bruit[ddlsBasY]], ["x","y"])
        simuIdentif.add_dirichlet(nodesHaut, [u_exp_bruit[ddlsHautX]], ["x"])
        simuIdentif.add_surfLoad(nodesHaut, [-sig], ["y"])
    else:    
        simuIdentif.add_dirichlet(nodes, [u_exp_bruit[ddlsX], u_exp_bruit[ddlsY]], ["x","y"])

    ddlsConnues, ddlsInconnues = simuIdentif.Bc_ddls_connues_inconnues(simuIdentif.problemType)
    # Affichage.Plot_BoundaryConditions(simuIdentif)

    # res = least_squares(func, x0, bounds=bounds, verbose=2, ftol=tol, gtol=tol, xtol=tol, jac='3-point')
    res = least_squares(func, x0, bounds=bounds, verbose=1, ftol=tol, gtol=tol, xtol=tol)

    dict = {
        "perturbation": perturbation        
    }

    if mat == "acier":
        dict["E"]=res.x[0]
        dict["v"]=res.x[1]
    elif mat == "bois":
        dict["EL"]=res.x[0]
        dict["GL"]=res.x[1]
        dict["ET"]=res.x[2]
        dict["vL"]=res.x[3]

    dict["bruit"]= bruit
    dict["u_exp_bruit"]= u_exp_bruit

    list_dict.append(dict)

df = pd.DataFrame(list_dict)

# print(df)

ax = plt.subplots()[1]

if mat == "acier":
    params = ["E","v"]
elif mat == "bois":
    params = ["EL", "GL", "ET", "vL"]

for param in params:

    values = df[param].values

    err = np.abs(values - values[0])/values[0]

    label = f"| {param} - {param}_exp | / {param}_exp"

    ax.plot(df["perturbation"], err, label=label)

ax.legend()
ax.set_xlabel("perturbation")

if mat == "acier":
    print(f"\nE = {res.x[0]:.3e}")
    print(f"v = {res.x[1]}")
elif mat == "bois":
    print(f"\nEL = {compIdentif.El}")
    print(f"GL = {compIdentif.Gl}")
    print(f"ET = {compIdentif.Et}")
    print(f"vL = {compIdentif.vl}")

diff_n = np.reshape(simuIdentif.displacement - u_exp, (mesh.Nn, 2))

# err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp.reshape((mesh.Nn,2)), axis=1)
err_n = np.linalg.norm(diff_n, axis=1)/np.linalg.norm(u_exp)
# err_n = np.linalg.norm(diff_n, axis=1)

Affichage.Plot_Result(simuIdentif, err_n, title=r"$\dfrac{\Vert u(p) - u_{exp} \Vert^2}{\Vert u_{exp} \Vert^2}$")

# print(np.linalg.norm(diff_n)/np.linalg.norm(u_exp))

plt.show()
