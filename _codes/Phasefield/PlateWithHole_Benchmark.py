from TicTac import Tic
import Materials
from BoundaryCondition import BoundaryCondition
from Geom import *
import Affichage as Affichage
import Interface_Gmsh as Interface_Gmsh
import Simu as Simu
import Dossier
import PostTraitement as PostTraitement
import PhaseFieldSimulation

import matplotlib.pyplot as plt

# Affichage.Clear()
# mpirun -np 4 python3 PlateWithHole_Benchmark.py

# Options

test=True
solve=True
saveParaview=False

comp = "Elas_Isot" # ["Elas_Isot", "Elas_IsotTrans"]
regu = "AT1" # "AT1", "AT2"
simpli2D = "DP" # ["CP","DP"]
useHistory=False

useNumba=True

# Convergence
maxIter = 400
# tolConv = 0.01
# tolConv = 0.05
tolConv = 0.1

if comp == "Elas_Isot":
    umax = 25e-6
    # umax = 35e-6    
else:
    umax = 80e-6

#["Bourdin","Amor","Miehe","He","Stress"]
#["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]
#["AnisotStress","AnisotStress_NoCross"]
for split in ["Amor"]:
    
    if split == "AnisotStress" and comp == "Elas_Isot":
        umax = 45e-6

    # Data

    L=15e-3
    h=30e-3
    ep=1
    diam=6e-3

    if comp == "Elas_Isot":
        E=12e9
        v=0.3
    elif comp == "Elas_IsotTrans":
        # El=11580*1e6
        El=12e9
        Et=500*1e6
        Gl=450*1e6
        vl=0.02
        vt=0.44
        v=0

    gc = 1.4
    l_0 = 0.12e-3
    
    if test:
        cc = 1
        clD = 0.25e-3*cc
        clC = 0.12e-3*cc
        # clD = l_0*2
        # clC = l_0

        # inc0 = 16e-8
        # inc1 = 4e-8

        inc0 = 8e-8
        inc1 = 2e-8
        
        # inc0 = 2e-8
        # inc1 = 1e-8
    else:
        clD = l_0/2
        clC = l_0/2

        inc0 = 8e-8
        inc1 = 2e-8

    # Nom du dossier
    nomDossier = "PlateWithHole_Benchmark"
    folder = PhaseFieldSimulation.ConstruitDossier(dossierSource=nomDossier,
    comp=comp, split=split, regu=regu, simpli2D=simpli2D,
    tolConv=tolConv, useHistory=useHistory, test=test, openCrack=False, v=v)

    
    if solve:

        print()

        point = Point()
        domain = Domain(point, Point(x=L, y=h), clD)
        circle = Circle(Point(x=L/2, y=h/2), diam, clC, isCreux=True)

        interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)
        mesh = interfaceGmsh.PlaqueAvecCercle2D(domain, circle, "TRI3")
        # mesh = interfaceGmsh.PlaqueAvecCercle(domain, circle, "QUAD4")
        # mesh = interfaceGmsh.PlaqueAvecCercle3D(domain, circle, [0,0,10e-3], 4, elemType="HEXA8", isOrganised=True)

        # Affichage.Plot_Maillage(mesh)
        # # plt.show()

        if simpli2D == "CP":
            isCp = True
        else:
            isCp = False
        
        if comp == "Elas_Isot":
            comportement = Materials.Elas_Isot(2,
            E=E, v=v, contraintesPlanes=isCp, epaisseur=ep)
        elif comp == "Elas_IsotTrans":
            comportement = Materials.Elas_IsotTrans(2,
                        El=El, Et=Et, Gl=Gl, vl=vl, vt=vt,
                        contraintesPlanes=isCp, epaisseur=ep,
                        axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

        phaseFieldModel = Materials.PhaseFieldModel(comportement, split, regu, gc, l_0, useHistory=useHistory)
        materiau = Materials.Materiau(phaseFieldModel=phaseFieldModel, verbosity=False)

        simu = Simu.Simu(mesh, materiau, verbosity=False, useNumba=useNumba)

        # Récupérations des noeuds

        B_lower = Line(point,Point(x=L))
        B_upper = Line(Point(y=h),Point(x=L, y=h))
        B_left = Line(point,Point(y=h))
        B_right = Line(Point(x=L),Point(x=L, y=h))

        c = diam/10
        domainA = Domain(Point(x=(L-c)/2, y=h/2+0.8*diam/2), Point(x=(L+c)/2, y=h/2+0.8*diam/2+c))
        domainB = Domain(Point(x=L/2+0.8*diam/2, y=(h-c)/2), Point(x=L/2+0.8*diam/2+c, y=(h+c)/2))

        nodes_lower = mesh.Nodes_Line(B_lower)
        nodes_upper = mesh.Nodes_Line(B_upper)
        nodes_left = mesh.Nodes_Line(B_left)
        nodes_right = mesh.Nodes_Line(B_right)

        # noeuds_bord = np.array().reshape(-1)
        noeuds_bord = []
        for ns in [nodes_lower, nodes_upper, nodes_left, nodes_right]:
            noeuds_bord.extend(ns)
        noeuds_bord = np.unique(noeuds_bord)
        
        node00 = mesh.Nodes_Point(point)
        nodesA = mesh.Get_Nodes_Domain(domainA)
        nodesB = mesh.Get_Nodes_Domain(domainB)

        ud=0
        damage_t=[]

        ddls_upper = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_upper, ["y"])

        def Chargement():
            simu.Init_Bc()
            simu.add_dirichlet("displacement", nodes_lower, [0], ["y"])
            simu.add_dirichlet("displacement", node00, [0], ["x"])
            simu.add_dirichlet("displacement", nodes_upper, [-ud], ["y"])

        Chargement()
        
        # Affichage.Plot_BoundaryConditions(simu)
        # plt.show()

        PhaseFieldSimulation.ResumeChargement(simu, umax,[inc0, inc1], [0.6, 1])

        resol = 0
        bord = 0
        displacement = []
        load = []

        while ud <= umax:

            resol += 1
            
            tic = Tic()
            
            Chargement()

            u, d, Kglob, iterConv = PhaseFieldSimulation.ResolutionIteration(simu=simu, tolConv=tolConv, maxIter=maxIter)

            if iterConv == maxIter:
                print(f'On converge pas apres {iterConv} itérations')
                break

            simu.Save_Iteration()

            temps = tic.Tac("Resolution phase field", "Resolution Phase Field", False)
            max_d = d.max()
            f = np.sum(np.einsum('ij,j->i', Kglob[ddls_upper, :].toarray(), u, optimize='optimal'))

            PhaseFieldSimulation.ResumeIteration(simu, resol, ud*1e6, d, iterConv, temps, "µm", ud/umax, True)

            if max_d<0.6:
                ud += inc0
            else:
                ud += inc1

            # Detection si on a touché le bord
            if np.any(d[noeuds_bord] >= 0.95):
                bord +=1
            
            if bord == 50:
                break

            displacement.append(ud)
            load.append(f)

        load = np.array(load)
        displacement = np.array(displacement)

        # Sauvegarde
        print()
        PostTraitement.Save_Load_Displacement(load, displacement, folder)
        PostTraitement.Save_Simu(simu, folder)
            
    else:
 
        load, displacement = PostTraitement.Load_Load_Displacement(folder)
        simu = PostTraitement.Load_Simu(folder)


    Affichage.Plot_ForceDep(displacement*1e3, load*1e-6, 'ud en mm', 'f en kN/mm', folder)

    filenameDamage = f"{split} damage_n"
    # titleDamage = fr"$\phi$"
    titleDamage = f"{split}"


    Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=True,
    folder=folder, filename=filenameDamage,
    title=titleDamage)

    if saveParaview:
        PostTraitement.Save_Simulation_in_Paraview(folder, simu)

    # Tic.getResume()

    if solve:
        Tic.getGraphs(folder, details=False)
    else:
        # Tic.getGraphs()
        plt.show()

    # plt.show()
    
    Tic.Clear()
    plt.close('all')