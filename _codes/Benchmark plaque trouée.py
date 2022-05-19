
from sympy import reshape
from TicTac import TicTac
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import PostTraitement
import Dossier

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

test=True
solve=True

comp = "Elas_Isot"
split = "Miehe" # ["Bourdin","Amor","Miehe"]
regu = "AT1" # "AT1", "AT2"


nom="_".join([comp, split, regu])

nomDossier = "Benchmarck_Compression"

if test:
    folder = Dossier.Append([nomDossier, "Test", nom])
else:
    folder = Dossier.Append([nomDossier, nom])

folder = Dossier.NewFile(folder, results=True)

# Data

L=15e-3
h=30e-3
ep=1
diam=6e-3

E=12e9
v=0.2
Sig = 10 #Pa

gc = 1.4
l_0 = 0.12e-3

# Création de la simulations

umax = 25e-6

if test:
    clD = l_0*2
    clC = l_0

    inc0 = 8e-8
    inc1 = 2e-8
else:
    clD = l_0/2
    clC = l_0/2

    inc0 = 16e-8
    inc1 = 4e-8

if solve:



    point = Point()
    domain = Domain(point, Point(x=L, y=h), clD)
    circle = Circle(Point(x=L/2, y=h/2), diam, clC)

    interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
    mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

    # Affichage.Plot_Maillage(mesh)
    # plt.show()

    comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
    phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
    materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel)

    simu = Simu.Simu(mesh, materiau, verbosity=False)

    # Récupérations des noeuds

    B_lower = Line(point,Point(x=L))
    B_upper = Line(Point(y=h),Point(x=L, y=h))
    B_left = Line(point,Point(y=h))
    B_right = Line(Point(x=L),Point(x=L, y=h))


    c = diam/10
    domainA = Domain(Point(x=(L-c)/2, y=h/2+0.8*diam/2), Point(x=(L+c)/2, y=h/2+0.8*diam/2+c))
    domainB = Domain(Point(x=L/2+0.8*diam/2, y=(h-c)/2), Point(x=L/2+0.8*diam/2+c, y=(h+c)/2))

    nodes_lower = mesh.Get_Nodes_Line(B_lower)
    nodes_upper = mesh.Get_Nodes_Line(B_upper)
    nodes_left = mesh.Get_Nodes_Line(B_left)
    nodes_right = mesh.Get_Nodes_Line(B_right)

    # noeuds_bord = np.array().reshape(-1)
    noeuds_bord = []
    for ns in [nodes_lower, nodes_upper, nodes_left, nodes_right]:
        noeuds_bord.extend(ns)
    noeuds_bord = np.unique(noeuds_bord)
    
    node00 = mesh.Get_Nodes_Point(point)
    nodesA = mesh.Get_Nodes_Domain(domainA)
    nodesB = mesh.Get_Nodes_Domain(domainB)

    # fig, ax = Affichage.Plot_Maillage(mesh)

    # for ns in [nodes0, nodesh, node00, nodesA, nodesB]:
    #     Affichage.Plot_NoeudsMaillage(mesh, ax=ax, noeuds=ns)   

    def Chargement():
        simu.Clear_Bc_Dirichlet()
        simu.add_dirichlet("displacement", nodes_lower, ["y"], [0])
        simu.add_dirichlet("displacement", node00, ["x"], [0])
        simu.add_dirichlet("displacement", nodes_upper, ["y"], [-ud])

    ud=0
    damage_t=[]
    displacement_t=[]

    resol=1
    bord=0

    while ud <= umax:

        Chargement()

        # Damage

        simu.Assemblage_d()

        damage = simu.Solve_d()

        damage_t.append(damage)

        # Displacement

        Kglob = simu.Assemblage_u()

        displacement = simu.Solve_u(useCholesky=True)

        displacement_t.append(displacement)

        max_d = np.round(damage.max(),3)
        min_d = np.round(damage.min(),3)
        
        f = np.sum(np.einsum('ij,j->i', Kglob[nodes_upper*2, nodes_upper*2], displacement[nodes_upper*2], optimize=True))

        print(f"{resol} : ud = {np.round(ud*1e6,2)} µm, f = {np.round(f/1e6)} kN/mm,  max d = {max_d}, min d = {min_d}")

        if max_d<0.6:
            ud += inc0
        else:
            ud += inc1
        
        if np.any(damage[noeuds_bord] >= 0.8):
            bord +=1
        
        if bord == 20:
            break
        
        resol += 1
    
    
    PostTraitement.Save_Simulation(folder, simu)

else:

    simu = PostTraitement.Load_Simulation(folder)


Affichage.Plot_Result(simu, "damage", folder=folder)

PostTraitement.Save_Simulation_in_Paraview(folder, simu, displacement_t, damage_t)







TicTac.getResume()







plt.show()



pass


