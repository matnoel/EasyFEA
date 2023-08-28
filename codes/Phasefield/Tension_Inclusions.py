import Display
from Interface_Gmsh import Interface_Gmsh, ElemType
from Geom import Point, Domain, Circle
import Materials
import Simulations

import numpy as np
import matplotlib.pyplot as plt

Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
L = 1 # mm
h = 0.2*2

# nL = 150
nL = 100
clC = L/nL # taille de maille
l0 = 2 * clC

# ----------------------------------------------
# Mesh
# ----------------------------------------------
domain = Domain(Point(-1/2,-1/2), Point(1/2, 1/2), clC)
inclusion = Circle(Point(), h, clC, isHollow=False)

mesh = Interface_Gmsh().Mesh_2D(domain, [inclusion], ElemType.TRI3)

nodesLeftRight = mesh.Nodes_Conditions(lambda x,y,z: (x==-L/2) | (x==L/2))
nodesLower = mesh.Nodes_Conditions(lambda x,y,z: y==-L/2)
nodesUpper = mesh.Nodes_Conditions(lambda x,y,z: y==L/2)
dofsY_Upper = Simulations.BoundaryCondition.Get_dofs_nodes(2, 'displacement', nodesUpper, ['y'])

nodes_inclu = mesh.Nodes_Circle(inclusion)
elem_inclu = mesh.Elements_Nodes(nodes_inclu)
elem_matrice = np.array(set(np.arange(mesh.Ne)) - set(elem_inclu))

# ----------------------------------------------
# Material
# ----------------------------------------------
E_mat = 52 # MPa
E_inclu = 10000
v = 0.3

sig_inclu = 10000 # MPa
sig_mat = 0.03

Gc_mat = l0 * sig_mat **2 / E_mat # mJ/mm^2
Gc_inclu = l0 * sig_inclu **2 / E_inclu

E = np.ones(mesh.Ne) * E_mat
v = np.ones_like(E) * v
Gc = np.ones_like(E) * Gc_mat

if elem_inclu.size > 0:    
    E[elem_inclu] = E_inclu
    Gc[elem_inclu] = Gc_inclu

Display.Plot_Result(mesh, E, nodeValues=False, title='$E$')
Display.Plot_Result(mesh, Gc, nodeValues=False, title='$G_c$')

comp = Materials.Elas_Isot(2, E, v, False, 1)

pfm = Materials.PhaseField_Model(comp, "AnisotStress", "AT2", Gc, l0)

# ----------------------------------------------
# Simulation
# ----------------------------------------------
simu = Simulations.Simu_PhaseField(mesh, pfm)

# nPg = mesh.groupElem.Get_gauss("rigi").poids.size
# Epsilon_e_pg = np.random.rand(mesh.Ne, nPg, 3)
# cP, cM = pfm.Calc_C(Epsilon_e_pg)

N = 300
displacements = np.linspace(0, 5e-4, N)

axLoad = plt.subplots()[1]
axLoad.set_xlabel('u [mm]'); axLoad.set_ylabel('f [N/mm]')
__, axDamage, cb = Display.Plot_Result(simu, 'damage')

forces = []

nDetect=0 # number of times an edge has been damaged

for i, ud in enumerate(displacements):

    # Boundary conditions
    simu.Bc_Init()
    simu.add_dirichlet(nodesLower, [0,0], ['x','y'])
    simu.add_dirichlet(nodesLeftRight, [0], ['x'])
    simu.add_dirichlet(nodesUpper, [ud], ['y'])

    # solve and save
    u, d, Kglob, convergence  = simu.Solve(1e-2)
    simu.Save_Iter()

    # print iteration
    simu.Results_Set_Iteration_Summary(i, 0, '', i/N, True)

    # load on the upper edge
    fr = np.sum(Kglob[dofsY_Upper,:]@u)/1000
    forces.append(fr)

    # plot iter
    axLoad.scatter(ud, fr, c='black')
    plt.figure(axLoad.figure)
    plt.pause(1e-12)

    # plot damage
    cb.remove()
    cb = Display.Plot_Result(simu, 'damage', ax=axDamage)[2]
    plt.figure(axDamage.figure)
    plt.pause(1e-12)

    if simu.damage[nodesLeftRight].max() >= 0.95:
        nDetect += 1

    if nDetect==10:
        break

forces = np.array(forces)

Display.Plot_BoundaryConditions(simu)
Display.Plot_Iter_Summary(simu)
Display.Plot_Energy(simu, forces, displacements)

plt.show()