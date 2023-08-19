from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Simulations
import Display
from Interface_Gmsh import Interface_Gmsh, ElemType
import Materials
import Geom

Display.Clear()

built_b_tild = False
verifOrtho = False

l = 45
h = 90
b = 20
d = 10

meshSize = l/15
elemType = ElemType.TRI3

mat = "bois" # "acier" "bois"

tol = 1e-14

sig = 10

gmshInterface = Interface_Gmsh()

pt1 = Geom.Point()
pt2 = Geom.Point(l, 0)
pt3 = Geom.Point(l, h)
pt4 = Geom.Point(0, h)
points = Geom.PointsList([pt1, pt2, pt3, pt4], meshSize)
circle = Geom.Circle(Geom.Point(l/2, h/2), d, meshSize, isHollow=True)

mesh = gmshInterface.Mesh_2D(points, [circle], elemType)

nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
nodes_p1 = mesh.Nodes_Tags(["P0"])
nodesBas = mesh.Nodes_Tags(["L0"])
nodesHaut = mesh.Nodes_Tags(["L2"])

Display.Plot_Mesh(mesh)
Display.Plot_Model(mesh)
# Display.Plot_Nodes(mesh, nodesX0)

tol0 = 1e-6
bSup = np.inf

if mat == "acier":
    E_exp, v_exp = 210000, 0.3
    comp = Materials.Elas_Isot(2, thickness=b)
    
elif mat == "bois":
    EL_exp, GL_exp, ET_exp, vL_exp = 12000, 450, 500, 0.3

    comp = Materials.Elas_IsotTrans(2, El=EL_exp, Et=ET_exp, Gl=GL_exp, vl=vL_exp, vt=0.3,
    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]), planeStress=True, thickness=b)

simu = Simulations.Simu_Displacement(mesh, comp)

simu.add_dirichlet(nodesBas, [0], ["y"])
simu.add_dirichlet(nodes_p1, [0], ["x"])
simu.add_surfLoad(nodesHaut, [-sig], ["y"])

Display.Plot_BoundaryConditions(simu)

u_exp = simu.Solve()

perturbation = 0.02
coefBruit = np.abs(u_exp).max()
bruit = coefBruit * (np.random.rand(u_exp.shape[0]) - 1/2) * perturbation
u_exp_bruit = u_exp + bruit

f_exp = simu.Get_K_C_M_F()[0] @ u_exp_bruit
# ici fonctionne parce que l'on connait les données matériaux a ce moment la.

# Display.Plot_Result(simu, "uy")
# Display.Plot_Result(simu, "Syy", coef=1/sig, nodeValues=False)
# Display.Plot_Result(simu, np.linalg.norm(vectRand.reshape((mesh.Nn), 2), axis=1), title="bruit")
# Display.Plot_Result(simu, u_exp_bruit.reshape((mesh.Nn,2))[:,1], title='uy bruit')

Display.Section("Identification")

compIdentif = Materials.Elas_Anisot(2, comp.C, useVoigtNotation=False, thickness=comp.thickness)

simuIdentif = Simulations.Simu_Displacement(mesh, compIdentif)

mat_c11 = np.array([[1,0,0],[0,0,0],[0,0,0]])
mat_c22 = np.array([[0,0,0],[0,1,0],[0,0,0]])
mat_c33 = np.array([[0,0,0],[0,0,0],[0,0,1]])
mat_c23 = np.array([[0,0,0],[0,0,1],[0,1,0]])
mat_c13 = np.array([[0,0,1],[0,0,0],[1,0,0]])
mat_c12 = np.array([[0,1,0],[1,0,0],[0,0,0]])

list_c = [mat_c11, mat_c22, mat_c33, mat_c23, mat_c13, mat_c12]

list_b = []

for m_c in list_c:

    compIdentif.Set_C(m_c, useVoigtNotation=False, update_S=False)

    simuIdentif.Need_Update()

    K_c = simuIdentif.Get_K_C_M_F()[0]

    list_b.append(K_c @ u_exp_bruit)

list_b = np.asarray(list_b)

if built_b_tild:

    # algo de Gram-Shmidt
    # https://fr.wikipedia.org/wiki/Algorithme_de_Gram-Schmidt

    list_b_tild = []

    for i in range(6):

        indexes = np.arange(i)    

        bj_tild = list(np.asarray(list_b_tild)[indexes])

        bi = list_b[i] - np.sum([(bj.T @ list_b[i])/(bj.T @ bj) * bj for bj in bj_tild])    

        list_b_tild.append(bi)

    list_b_tild = np.asarray(list_b_tild)

    difff = list_b - list_b_tild

    Gamma = np.einsum("in,jn->ij", list_b, list_b_tild)
else:

    Gamma = np.einsum("in,jn->ij", list_b, list_b)

if verifOrtho:
    # Verif orthogonalité
    bList = list_b_tild.copy()
    for i, b in enumerate(bList):
        
        indexes = list(range(len(bList)))
        indexes.remove(i)

        verif = np.abs([b.T @ bList[ind] for ind in indexes])

        tests = np.array(verif) <= 1e-3

        assert False not in tests, "Les vecteurs b ne sont pas ortogonaux"

if built_b_tild:
    p = np.einsum("i,ni->n", f_exp, list_b_tild)
else:
    p = np.einsum("i,ni->n", f_exp, list_b)

c = np.linalg.solve(Gamma, p)

C = comp.C
c_exp = [C[0,0],C[1,1],C[2,2],C[1,2],C[0,2],C[0,1]]

diff = c - c_exp

print(diff)

plt.show()
