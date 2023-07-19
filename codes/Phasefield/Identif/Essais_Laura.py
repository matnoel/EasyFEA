import Display
from Geom import Point, PointsList, Circle, Domain, normalize_vect
from Interface_Gmsh import Interface_Gmsh
import Materials
import Simulations
import Folder
import PostTraitement

import matplotlib.pyplot as plt
import numpy as np

Display.Clear()

folder_FCBA = Folder.New_File("Essais FCBA",results=True)
folder = Folder.Join([folder_FCBA, "Essais Laura"])

dim = 3
test = True
optimMesh = True

# loadType = 0 # nodesLower
loadType = 1 # in hole

# ----------------------------------------------
# GEOM
# ----------------------------------------------

H = 120 # mm
L = 90
h = 35 
d = 10
ep = 20

# nL = 50
# l0 = L/nL
l0 = 1
nL = L//l0

# ----------------------------------------------
# Mesh
# ----------------------------------------------

clC = l0 if test else l0/2
clD = clC*2 if optimMesh else clC

if optimMesh:
    pr0 = Point(L/2-d, 0, 0)
    pr1 = Point(L/2+d, H, ep)
    refineGeom = Domain(pr0, pr1, clC)
else:
    refineGeom = None

p1 = Point(0,0)
p2 = Point(L,0)
p3 = Point(L,H)
p4 = Point(0,H)
contour = PointsList([p1,p2,p3,p4], clD)

pC = Point(L/2, H-h); pc = pC.coordo
circle = Circle(pC, d, clC, True)

if dim == 2:
    mesh = Interface_Gmsh().Mesh_2D(contour, [circle], "TRI6", refineGeom=refineGeom)
    directions = ['x','y']
else:
    mesh = Interface_Gmsh().Mesh_3D(contour, [circle], [0,0,-ep], 3, "HEXA8", refineGeom=refineGeom)
    directions = ['x','y','z']


mesh.Resume()

# ----------------------------------------------
# Materials
# ----------------------------------------------

# Propriétés pour l'essai 4
Gc = 0.075 # mJ/mm2

psiC = (3*Gc)/(16*l0)

El = 15716.16722094732 
Et = 232.6981580878141
Gl = 557.3231495541391
vl = 0.02
vt = 0.44

rot = 90 * np.pi/180
axis_l = np.array([np.cos(rot), np.sin(rot), 0])
axis_t = np.cross(np.array([0,0,1]), axis_l)

split = "AnisotStress"
regu = "AT1"

comp = Materials.Elas_IsotTrans(dim, El, Et, Gl, vl, vt, axis_l, axis_t, True, ep)
pfm = Materials.PhaseField_Model(comp, split, regu, Gc, l0)

# ----------------------------------------------
# Simulation
# ----------------------------------------------

simu = Simulations.Simu_Displacement(mesh, comp)

nodesLower = mesh.Nodes_Conditions(lambda x,y,z: y==0)

if loadType == 0:
    surf = ep * L
    nodesLoad = mesh.Nodes_Conditions(lambda x,y,z: y==H)    

elif loadType == 1:
    surf = np.pi * d/2 * ep
    nodesLoad = mesh.Nodes_Cylindre(circle, [0,0,-ep])
    nodesLoad = nodesLoad[mesh.coordo[nodesLoad,1] <= pC.y]
    # Affichage.Plot_Nodes(mesh, nodesLoad)

    group = mesh.Get_list_groupElem(dim-1)[0]
    elems = group.Get_Elements_Nodes(nodesLoad)

    aire = np.einsum('ep,p->', group.Get_jacobien_e_pg("masse")[elems], group.Get_poid_pg("masse"))

    if dim == 2:
        aire *= ep 

    print(f"errSurf = {np.abs(surf-aire)/surf:.3e}")

    def FuncEval(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Evaluation de la fonction sig cos(theta)^2 vect_n"""
        
        # Calcul de l'angle
        theta = np.arctan((x-pc[0])/(y-pc[1]))

        # Coordonnées des points de gauss sous forme de matrice
        coord = np.zeros((x.shape[0],x.shape[1],3))
        coord[:,:,0] = x
        coord[:,:,1] = y
        coord[:,:,2] = z

        # Construction du vecteur normal
        vect = coord - pc
        vectN = np.einsum('npi,np->npi', vect, 1/np.linalg.norm(vect, axis=2))
        
        # Chargement
        loads = f/surf * np.einsum('np,npi->npi',np.cos(theta)**2, vectN)

        return loads

    funcEvalX = lambda x,y,z: FuncEval(x,y,z)[:,:,0]
    funcEvalY = lambda x,y,z: FuncEval(x,y,z)[:,:,1]
    
    # # Affichage
    # ax = plt.subplots()[1]
    # ax.axis('equal')
    # angle = np.linspace(0, np.pi*2, 360)
    # ax.scatter(0,0,marker='+', c='black')
    # ax.plot(d/2*np.cos(angle),d/2*np.sin(angle), c="black")
    
    # sig = d/2
    # angle = np.linspace(0, np.pi, 21)

    # x = - d/2 * np.cos(angle)
    # y = - d/2 * np.sin(angle)

    # coord = np.zeros((x.size,2))
    # coord[:,0] = x; coord[:,1] = y
    # # ax.plot(x,y, c="red")

    # vectN = normalize_vect(coord)


    # f = sig * np.einsum("n,ni->ni", np.sin(angle)**2, vectN)
    # f[np.abs(f)<=1e-12] = 0

    # [ax.arrow(x[i], y[i], f[i,0], f[i,1], color='red', head_width=1e-1*2, length_includes_head=True) for i in range(angle.size)]
    # ax.plot((coord+f)[:,0], (coord+f)[:,1], c='red')
    # ax.set_axis_off()

    # Affichage.Save_fig(folder, 'illustration')

    # # ax.annotate("$x$",xy=(1,0),xytext=(0,0),arrowprops=dict(arrowstyle="->"), c='black')

    pass
    







# simu.Solve()

array_f = np.linspace(0, 4, 10)*1000
# array_f = np.array([2000])

list_psiP = []

for f in array_f:

    simu.Bc_Init()
    simu.add_dirichlet(nodesLower, [0]*dim, directions)
    if loadType == 0:
        simu.add_surfLoad(nodesLoad, [-f/surf], ['y'])
    elif loadType == 1:
        simu.add_surfLoad(nodesLoad, [funcEvalX, funcEvalY], ["x","y"],
                          description=r"$\mathbf{q}(\theta) = \sigma \ sin^2(\theta) \ \mathbf{n}(\theta)$")

    simu.Solve()
    simu.Save_Iteration()

    # Calcul l'energie
    Epsilon_e_pg = simu._Calc_Epsilon_e_pg(simu.displacement, "masse")
    psiP_e_pg, psiM_e_pg = pfm.Calc_psi_e_pg(Epsilon_e_pg)
    psiP_e = np.max(psiP_e_pg, axis=1)

    list_psiP.append(np.max(psiP_e))

    print(f"f = {f/1000:.3f} kN -> psiP/psiC = {list_psiP[-1]/psiC:.2e}")


if len(list_psiP) > 1:
    axLoad = plt.subplots()[1]
    axLoad.set_xlabel("$f \ [kN]$"); axLoad.set_ylabel("$\psi^+ \ / \ \psi_c$")
    axLoad.grid() 


    array_psiP = np.array(list_psiP)

    axLoad.plot([0,array_f[-1]/1000], [1, 1], zorder=3, c='black')
    axLoad.plot(array_f/1000, array_psiP/psiC, zorder=3, c='blue')
 
    Display.Save_fig(folder, "Load")



Display.Plot_Mesh(mesh)
ax = Display.Plot_BoundaryConditions(simu, folder=folder)
# f_v = simu.Get_K_C_M_F()[0] @ simu.displacement
# f_m = f_v.reshape(-1,2)
# f_m *= 1
# nodes = np.concatenate([nodesLoad, nodesLower])
# xn,yn,zn = mesh.coordo[:,0], mesh.coordo[:,1], mesh.coordo[:,2]
# if dim == 2:    
#     # ax.quiver(xn[nodes], yn[nodes], f_m[nodes,0], f_m[nodes,1], color='red', width=1e-3, scale=1e3)
#     ax.quiver(xn, yn, f_m[:,0], f_m[:,1], color='red', width=1e-3, scale=1e4)


Display.Plot_Result(simu, psiP_e, title="$\psi^+$", nodeValues=False)
ax = Display.Plot_Result(simu, psiP_e/psiC, nodeValues=True, title="$\psi^+ \ / \ \psi_c$", colorbarIsClose=False)[1]

elemtsDamage = np.where(psiP_e >= psiC)[0]
if elemtsDamage.size > 0:
    nodes = np.unique(mesh.connect[elemtsDamage])
    # Affichage.Plot_Elements(mesh, nodes, alpha=0.2, edgecolor='black', ax=ax)
Display.Save_fig(folder, "psiPpsiC")

Display.Plot_Result(simu, "Sxx", plotMesh=False)
Display.Plot_Result(simu, "Syy", plotMesh=False)
Display.Plot_Result(simu, "Sxy", plotMesh=False)

simu.Resultats_Resume()

PostTraitement.Make_Paraview(folder, simu)

plt.show()