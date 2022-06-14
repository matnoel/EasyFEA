
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

# Options

test=True
solve=True
saveParaview=True

comp = "Elas_Isot"
split = "Stress" # ["Bourdin","Amor","Miehe","Stress"]
regu = "AT1" # "AT1", "AT2"

# Data

L=9e-2
H=12e-2
h=3.5e-2
ep=2e-2
diam=1e-2

E=11.580e9
v=0.3

gc = 1
l_0 = L/50

# Création de la simulations

umax = 3e-3
N=400

if test:
    cc = 0.5
    clD = l_0*2*cc
    clC = l_0*cc
    # clD = l_0*2
    # clC = l_0

    # inc0 = 16e-8
    # inc1 = 4e-8
    # inc0 = 8e-8
    # inc1 = 2e-8

    inc0 = umax/N
    inc1 = inc0/3

else:
    clD = l_0/2
    clC = l_0/2

    inc0 = 8e-8
    inc1 = 2e-8 

nom="_".join([comp, split, regu])
nom = f"{nom} pour v={v}"

nomDossier = "PlateWithHole_FCBA"

folder = Dossier.NewFile(nomDossier, results=True)

if test:
    folder = Dossier.Append([folder, "Test", nom])
else:
    folder = Dossier.Append([folder, nom])

if solve:

    print(folder)

    point = Point()
    domain = Domain(point, Point(x=L, y=H), clD)
    circle = Circle(Point(x=L/2, y=H-h), diam, clC)

    interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
    mesh = interfaceGmsh.PlaqueTrouée(domain, circle, "TRI3")

    comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=False, epaisseur=ep)
    phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
    materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel)

    simu = Simu.Simu(mesh, materiau, verbosity=False)

    # Récupérations des noeuds

    B_lower = Line(point,Point(x=L))
    B_upper = Line(Point(y=H),Point(x=L, y=H))
    B_left = Line(point,Point(y=H))
    B_right = Line(Point(x=L),Point(x=L, y=H))

    c = diam/10
    domainA = Domain(Point(x=(L-c)/2, y=H/2+0.8*diam/2), Point(x=(L+c)/2, y=H/2+0.8*diam/2+c))
    domainB = Domain(Point(x=L/2+0.8*diam/2, y=(H-c)/2), Point(x=L/2+0.8*diam/2+c, y=(H+c)/2))

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
        simu.Init_Bc_Dirichlet()
        simu.add_dirichlet("displacement", nodes_lower, [0], ["y"])
        simu.add_dirichlet("displacement", node00, [0], ["x"])
        simu.add_dirichlet("displacement", nodes_upper, [-ud], ["y"])

    Chargement()
    
    # Affichage.Plot_BoundaryConditions(simu)
    # plt.show()

    Affichage.NouvelleSection("Simulation")

    maxIter = 200
    tolConv = 0.01
    resol = 1
    bord = 0
    dep = []
    forces = []

    fig, ax = plt.subplots()

    while ud <= umax:
        
        tic = TicTac()

        iterConv=0
        convergence = False
        dold = simu.damage

        Chargement()

        while not convergence:
            
            iterConv += 1            

            # Damage
            simu.Assemblage_d()
            damage = simu.Solve_d()

            # Displacement
            Kglob = simu.Assemblage_u()
            displacement = simu.Solve_u()

            dincMax = np.max(np.abs(damage-dold))
            convergence = dincMax <= tolConv
            # print(dincMax)
            dold = damage.copy()

            if iterConv == maxIter:
                break
        
        # TODO Comparer avec code matlab

        if iterConv == maxIter:
            print(f'On converge pas apres {iterConv} itérations')
            break

        simu.Save_solutions()

        temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
        temps = np.round(temps,3)

        max_d = damage.max()
        min_d = damage.min()
        
        f = np.sum(np.einsum('ij,j->i', Kglob[nodes_upper*2, nodes_upper*2], displacement[nodes_upper*2], optimize=True))/1e6

        print(f"{resol:4} : ud = {ud*1e3:5.2} µm,  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{temps} s")

        if max_d<0.5:
            ud += inc0
        else:
            ud += inc1
        
        if np.any(damage[noeuds_bord] >= 0.8):
            bord +=1
        
        if bord == 1:
            break

        dep.append(ud)
        forces.append(f)

        ax.cla()
        ax.plot(dep, np.abs(forces), c='black')
        ax.set_xlabel("ud en µm")
        ax.set_ylabel("f en kN/mm")
        plt.pause(0.0000001)
        
        resol += 1
    
    plt.savefig(Dossier.Append([folder,"forcedep.png"]))

    # Sauvegarde
    PostTraitement.Save_Simu(simu, folder)
        
else:   

    simu = PostTraitement.Load_Simu(folder)


Affichage.Plot_Result(simu, "damage", folder=folder, unite=f" pour v ={v}",  valeursAuxNoeuds=True)

# Affichage.Plot_Result(simu, "psiP", folder=folder, unite=f" pour v ={v}", valeursAuxNoeuds=True,affichageMaillage=True)

if saveParaview:
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)







TicTac.getResume()







plt.show()



pass


