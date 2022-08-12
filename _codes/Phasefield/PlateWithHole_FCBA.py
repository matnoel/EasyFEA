from PythonEF.TicTac import Tic
import PythonEF.Materiau as Materiau
from PythonEF.Geom import *
import PythonEF.Affichage as Affichage
import PythonEF.Interface_Gmsh as Interface_Gmsh
import PythonEF.Simu as Simu
import PythonEF.Dossier as Dossier
import pandas as pd
import PythonEF.PostTraitement as PostTraitement

import matplotlib.pyplot as plt

Affichage.Clear()

# Options

test=True
solve=False
saveParaview=True

comp = "Elas_Isot"
split = "AnisotStress" # ["Bourdin","Amor","Miehe","Stress","AnisotStress"]
regu = "AT1" # "AT1", "AT2"

loadInHole = True


# Data

L=9e-2
H=12e-2
h=3.5e-2
ep=2e-2
diam=1e-2
r=diam/2

E=12e9
# v=0.3
# gc = 1.4

# E=1.4015e8
# E=E/10.7098
v=0.44
# gc = 2 0.9 a la fin
gc = 1


# l_0 = L/50
l_0 = L/70

# Création de la simulations

if loadInHole:
    loadMax = 2e3 #N
else:
    # umax = 25e-6
    # umax = 0.5302e-3
    umax = 2e-3 #m

if test:
    cc = 0.5
    clD = l_0*2*cc
    clC = l_0*cc
    # clD = l_0*2
    # clC = l_0

    if loadInHole:
        N=300
        inc0 = loadMax/N
        inc1 = inc0/6
    else:
        inc0 = 8e-6
        inc1 = 2e-6

        # inc0 = 16e-8
        # inc1 = 4e-8

        # inc0 = 8e-8
        # inc1 = 2e-8
else:

    clD = l_0/2
    clC = l_0/2

    if loadInHole:
        raise "Erreur"
    else:        
        inc0 = 8e-8
        inc1 = 2e-8 

nom="_".join([comp, split, regu])
nom = f"{nom} pour v={v}"

nomDossier = "PlateWithHole_FCBA"
if loadInHole:
    nomDossier += "_loadInHole"

folder = Dossier.NewFile(nomDossier, results=True)

if test:
    folder = Dossier.Join([folder, "Test", nom])
else:
    folder = Dossier.Join([folder, nom])

if solve:

    print(folder)

    point = Point()
    domain = Domain(point, Point(x=L, y=H), clD)
    circle = Circle(Point(x=L/2, y=H-h), diam, clC)

    interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
    mesh = interfaceGmsh.PlaqueAvecCercle(domain, circle, "TRI3")

    comportement = Materiau.Elas_Isot(2, E=E, v=v, contraintesPlanes=True, epaisseur=ep)
    phaseFieldModel = Materiau.PhaseFieldModel(comportement, split, regu, gc, l_0)
    materiau = Materiau.Materiau(phaseFieldModel=phaseFieldModel)

    simu = Simu.Simu(mesh, materiau, verbosity=False)

    # Récupérations des noeuds

    B_lower = Line(point,Point(x=L))
    B_upper = Line(Point(y=H),Point(x=L, y=H))
    B_left = Line(point,Point(y=H))
    B_right = Line(Point(x=L),Point(x=L, y=H))

    nodes_lower = mesh.Nodes_Line(B_lower)
    nodes_upper = mesh.Nodes_Line(B_upper)
    nodes_left = mesh.Nodes_Line(B_left)
    nodes_right = mesh.Nodes_Line(B_right)

    noeuds_cercle = mesh.Nodes_Circle(circle)
    noeuds_cercle = noeuds_cercle[np.where(mesh.coordo[noeuds_cercle,1]<=circle.center.y)]
    noeud_contact = mesh.Nodes_Point(Point(circle.center.x, circle.center.y - r))
    # essai = (mesh.coordo[noeuds_cercle,1]-circle.center.y)/r
    

    # noeuds_bord = np.array().reshape(-1)
    noeuds_bord = []
    for ns in [nodes_lower, nodes_upper, nodes_left, nodes_right]:
        noeuds_bord.extend(ns)
    noeuds_bord = np.unique(noeuds_bord)
    
    node00 = mesh.Nodes_Point(point)

    ud=0
    load=0

    def Chargement():
        simu.Init_Bc()        
        simu.add_dirichlet("displacement", nodes_lower, [0], ["y"])
        simu.add_dirichlet("displacement", node00, [0], ["x"])

        # fmax = -load/(2*np.pi*r*ep)
        # simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: fmax*(y-circle.center.y)/r], ["y"])

        if loadInHole:
            SIG = load/(np.pi * r * ep) #Pa
            simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(x-circle.center.x)/r * np.abs((y-circle.center.y)/r)], ["x"])
            simu.add_surfLoad("displacement",noeuds_cercle, [lambda x,y,z: SIG*(y-circle.center.y)/r * np.abs((y-circle.center.y)/r)], ["y"])
        else:
            simu.add_dirichlet("displacement", nodes_upper, [-ud], ["y"])

    Chargement()
    
    # ax = Affichage.Plot_BoundaryConditions(simu,folder)
    # Affichage.Plot_NoeudsMaillage(mesh, ax=ax, noeuds=noeuds_cercle, showId=True)
    # plt.show()

    Affichage.NouvelleSection("Simulation")

    maxIter = 200
    tolConv = 0.005
    resol = 1
    bord = 0
    dep = []
    forces = []

    

    iterSansEndomagement=0

    # while ud <= umax:
    while load <= loadMax:
        
        tic = Tic()

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

        if iterConv == maxIter:
            print(f'On converge pas apres {iterConv} itérations')
            break

        simu.Save_Iteration()

        temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
        temps = np.round(temps,3)

        max_d = damage.max()
        min_d = damage.min()

        if loadInHole:
            depl = float(displacement[noeud_contact*2])
            force = load
        else:
            depl = ud
            force = np.sum(np.einsum('ij,j->i', Kglob[nodes_upper*2, nodes_upper*2], displacement[nodes_upper*2], optimize='optimal'))

        # depl = np.abs(np.max(displacement[noeuds_cercle*2]))
        if loadInHole:
            print(f"{resol:4} : load = {load*1e-3:5.2} kN,  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{temps} s")
        else:
            print(f"{resol:4} : ud = {depl*1e3:5.2} mm,  d = [{min_d:.2e}; {max_d:.2e}], {iterConv}:{temps} s")

        if max_d<0.5:
            if loadInHole:
                load += inc0
            else:
                ud += inc0
        else:
            if loadInHole:
                load += inc1
            else:
                ud += inc1
        
        if np.any(damage[noeuds_bord] >= 0.8):
            bord +=1
        
        dep.append(depl)        
        forces.append(force)

        # Arret de la simulation si le bord est endommagé
        if bord == 1:
            break
        
        resol += 1
    
    
    # Sauvegarde
    PostTraitement.Save_Simu(simu, folder)

    PostTraitement.Save_Load_Displacement(forces, dep, folder)
        
else:   

    simu = PostTraitement.Load_Simu(folder)

    forces, dep = PostTraitement.Load_Load_Displacement(folder)

fig, ax = plt.subplots()
# list_iterd0 = np.arange(iterSansEndomagement)
ax.plot(np.array(dep)*1e3, np.abs(np.array(forces))/1e3)
# ax.plot(np.array(dep)[list_iterd0]*1e3, np.abs(np.array(forces)[list_iterd0])/1e3, c='black')
ax.set_xlabel("ud en mm")
ax.set_ylabel("f en kN")
plt.savefig(Dossier.Join([folder,"forcedep.png"]))

# plt.pause(0.0000001)

Affichage.Plot_Result(simu, "damage", title=fr"$\phi \ pour \ \nu={v}$",
valeursAuxNoeuds=True, colorbarIsClose=True, affichageMaillage=False,
folder=folder, filename=f"damage_n v={v} FCBA")

if saveParaview:
    PostTraitement.Save_Simulation_in_Paraview(folder, simu)







Tic.getResume()





plt.show()



pass


