from TicTac import TicTac
import Materiau
from Geom import *
import Affichage
import Interface_Gmsh
import Simu
import Dossier
import PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()
# mpirun -np 4 python3 PlateWithHole_Benchmark.py

# Options

test=True
solve=True
saveParaview=False

comp = "Elas_IsotTrans" # ["Elas_Isot", "Elas_IsotTrans"]
split = "AnisotStress" # ["Bourdin","Amor","Miehe","AnisotMiehe","Stress","AnisotStress"]
regu = "AT1" # "AT1", "AT2"
simpli2D = "DP" # ["CP","DP"]

# Data

L=15e-3
h=30e-3
ep=1
diam=6e-3

if comp == "Elas_Isot":
    E=12e9
    v=0.3
elif comp == "Elas_IsotTrans":
    El=11580*1e6
    Et=500*1e6
    Gl=450*1e6
    vl=0.02
    vt=0.44

gc = 1.4
l_0 = 0.12e-3

# Création de la simulations

umax = 25e-6

if test:
    cc = 1.2
    # cc = 1
    clD = 0.25e-3*cc
    clC = 0.12e-3*cc
    # clD = l_0*2
    # clC = l_0

    # inc0 = 16e-8
    # inc1 = 4e-8

    # inc0 = 8e-84
    inc0 = 8e-8
    inc1 = 2e-8 
else:
    clD = l_0
    clC = l_0/2

    inc0 = 8e-8
    inc1 = 2e-8 

nom="_".join([comp, split, regu, simpli2D])
if comp == "Elas_Isot":
    nom = f"{nom} pour v={v}"

nomDossier = "PlateWithHole_Benchmark"

folder = Dossier.NewFile(nomDossier, results=True)

if test:
    folder = Dossier.Join([folder, "Test", nom])
else:
    folder = Dossier.Join([folder, nom])

if solve:

    print(folder)

    point = Point()
    domain = Domain(point, Point(x=L, y=h), clD)
    circle = Circle(Point(x=L/2, y=h/2), diam, clC)

    interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
    mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")
    
    if comp == "Elas_Isot":
        comportement = Materiau.Elas_Isot(2,
        E=E, v=v, contraintesPlanes=False, epaisseur=ep)
    elif comp == "Elas_IsotTrans":
        comportement = Materiau.Elas_IsotTrans(2,
                    El=11580, Et=500, Gl=450, vl=0.02, vt=0.44,
                    contraintesPlanes=True, epaisseur=ep,
                    axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

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

    ud=0
    damage_t=[]    
    

    def Chargement():
        simu.Init_Bc()
        simu.add_dirichlet("displacement", nodes_lower, [0], ["y"])
        simu.add_dirichlet("displacement", node00, [0], ["x"])
        simu.add_dirichlet("displacement", nodes_upper, [-ud], ["y"])

    Chargement()
    
    # Affichage.Plot_BoundaryConditions(simu)
    # plt.show()

    Affichage.NouvelleSection("Simulation")

    maxIter = 250
    # tolConv = 0.0025
    # tolConv = 0.005
    tolConv = 0.01
    resol = 0
    bord = 0
    displacement = []
    load = []

    while ud <= umax:

        resol += 1
        
        tic = TicTac()

        iterConv=0
        convergence = False
        dold = simu.damage

        Chargement()

        while not convergence:
            
            iterConv += 1            

            # Damage
            simu.Assemblage_d()
            d = simu.Solve_d()

            # Displacement
            Kglob = simu.Assemblage_u()            
            u = simu.Solve_u()

            dincMax = np.max(np.abs(d-dold))
            # TODO faire en relatif np.max(np.abs((damage-dold)/dold))?
            convergence = dincMax <= tolConv
            # if damage.min()>1e-5:
            #     convergence=False
            dold = d.copy()

            if iterConv == maxIter:
                break

            # convergence=True
        
        # TODO Comparer avec code matlab

        if iterConv == maxIter:
            print(f'On converge pas apres {iterConv} itérations')
            break

        simu.Save_Iteration()

        temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
        temps = np.round(temps,3)

        max_d = d.max()
        min_d = d.min()        
        f = np.sum(np.einsum('ij,j->i', Kglob[nodes_upper*2, nodes_upper*2], u[nodes_upper*2], optimize=True))

        print(f"{resol:4} : ud = {np.round(ud*1e6,3)} µm,  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{temps} s")

        # if ud>12e-6:
        #     inc0 = 1e-8

        if max_d<0.6:
            ud += inc0
        else:
            ud += inc1
        
        if np.any(d[noeuds_bord] >= 0.95):
            bord +=1
        
        if bord == 1:
            break

        displacement.append(ud)
        load.append(f)



    # Sauvegarde
    PostTraitement.Save_Simu(simu, folder)

    PostTraitement.Save_Load_Displacement(load, displacement, folder)
        
else:

    simu = PostTraitement.Load_Simu(folder)

    PostTraitement.Load_Load_Displacement(folder)

fig, ax = plt.subplots()
ax.plot(displacement, np.abs(load)/1e6, c='black')
ax.set_xlabel("ud en m")
ax.set_ylabel("f en kN/mm")
PostTraitement.Save_fig(folder, "forcedep")

if comp == "Elas_Isot":
    filenameDamage = f"{split} damage_n pour v={v}"
else:
    filenameDamage = f"{split} damage_n"


Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=True,
folder=folder, filename=filenameDamage, 
title=fr"$\phi \ pour \ \nu ={v}$")

if saveParaview:
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)

TicTac.getResume()







plt.show()



pass


