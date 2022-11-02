from TicTac import Tic
import Materials
from BoundaryCondition import BoundaryCondition
from Geom import *
import Affichage as Affichage
import Interface_Gmsh as Interface_Gmsh
import Simu as Simu
import PostTraitement as PostTraitement
import PhaseFieldSimulation

import matplotlib.pyplot as plt



# Affichage.Clear()
# mpirun -np 4 python3 PlateWithHole_Benchmark.py

# Options

test = False
solve = True
plotMesh = False
plotIter = False
plotResult = True
showFig = True
plotEnergie = False
saveParaview = False; NParaview=200
makeMovie = False; NMovie = 300


problem = "CompressionFCBA2" # ["Benchmark" , "CompressionFCBA", "CompressionFCBA2", "CompressionFCBA3"]
comp = "Elas_IsotTrans" # ["Elas_Isot", "Elas_IsotTrans"]
regu = "AT2" # ["AT1", "AT2"]
solveur = "History" # ["History", "HistoryDamage", "BoundConstrain"]
optimMesh = True

useNumba = True

# Convergence
maxIter = 500
tolConv = 1e-0
# TODO Faire la convergence sur l'energie ?

if comp == "Elas_Isot":
    umax = 25e-6
    # umax = 35e-6    
else:
    umax = 80e-6

if "CompressionFCBA" in problem:
    nL=200
else:
    nL=0



#["Bourdin","Amor","Miehe","He","Stress"]
#["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]
#["AnisotStress","AnisotStress_NoCross"]
for split in ["AnisotStress"]:
    
    # if split == "AnisotStress" and comp == "Elas_Isot":
    #     umax = 45e-6

    # Data

    if problem == "Benchmark":
        L=15e-3
        h=30e-3
        ep=1
        diam=6e-3

        gc = 1.4
        l_0 = 0.12e-3

        inc0 = 8e-8
        inc1 = 2e-8

        tresh0 = 0.6
        tresh1 = 1

        simpli2D = "DP" # ["CP","DP"]

        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]

    elif "CompressionFCBA" in problem:
        L=9e-2
        if problem in ["CompressionFCBA2","CompressionFCBA3"]:
            # Pour un carré
            h=L
        else:
            h=12e-2
        ep=2e-2
        
        diam=1e-2
        if problem == "CompressionFCBA2":
            diam=2e-2
        if problem == "CompressionFCBA3":
            diam=1e-2
        r=diam/2
        
        gc = 1.4/2
        # l_0 = 0.12e-3
        l_0 = L/nL

        inc0 = 8e-7
        inc1 = 2e-7
        inc2 = inc1/2

        tresh0 = 0.2
        tresh1 = 1
        tresh2 = 130

        simpli2D = "CP" # ["CP","DP"]

        listInc = [inc0, inc1, inc2]
        listTresh = [tresh0, tresh1, tresh2]

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
    
    if test:

        if optimMesh:
            clD = l_0*3
            clC = l_0
        else:
            if problem == "Benchmark":
                clD = 0.25e-3
                clC = 0.12e-3
            else:
                clD = l_0*2
                clC = l_0


    else:
        
        if optimMesh:
            clD = l_0*2
            clC = l_0/2
        else:
            clD = l_0/2
            clC = l_0/2

    # Nom du dossier
    nomDossier = "PlateWithHole_" + problem
    folder = PhaseFieldSimulation.ConstruitDossier(dossierSource=nomDossier, comp=comp, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solveur=solveur, test=test, closeCrack=False, v=v, nL=nL)
    
    if solve:

        print()

        point = Point()
        domain = Domain(point, Point(x=L, y=h), clD)
        circle = Circle(Point(x=L/2, y=h/2), diam, clC, isCreux=True)
        
        interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False, verbosity=False)

        if optimMesh:
            # Concentration de maillage sur la fissure
            if problem == "Benchmark":
                ecartZone = diam*1.5/2
            elif "CompressionFCBA" in problem:
                ecartZone = diam
            if split in ["Bourdin", "Amor"]:
                domainFissure = Domain(Point(y=h/2-ecartZone, x=0), Point(y=h/2+ecartZone, x=L), clC)
            else:
                domainFissure = Domain(Point(x=L/2-ecartZone, y=0), Point(x=L/2+ecartZone, y=h), clC)
            mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3", domainFissure)
        else:
            mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")

        # mesh = interfaceGmsh.PlaqueAvecCercle(domain, circle, "QUAD4")
        # mesh = interfaceGmsh.PlaqueAvecCercle3D(domain, circle, [0,0,10e-3], 4, elemType="HEXA8", isOrganised=True)
        if plotMesh:
            Affichage.Plot_Maillage(mesh)
            plt.show()

        if simpli2D == "CP":
            isCp = True
        else:
            isCp = False
        
        if comp == "Elas_Isot":
            comportement = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=isCp, epaisseur=ep)
        elif comp == "Elas_IsotTrans":
            comportement = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=isCp, epaisseur=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

        phaseFieldModel = Materials.PhaseFieldModel(comportement, split, regu, gc, l_0, solveur=solveur)
        materiau = Materials.Materiau(phaseFieldModel, verbosity=False)

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
        nodesA = mesh.Nodes_Domain(domainA)
        nodesB = mesh.Nodes_Domain(domainB)

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

        PhaseFieldSimulation.ResumeChargement(simu, umax, listInc, listTresh)

        resol = 0
        bord = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True)


        def Condition():
            if problem == "Benchmark":
                return ud <= umax
            elif "CompressionFCBA" in problem:
                return simu.damage[noeuds_bord].max() <= 1
                # return simu.damage.max() <= 0.5

        while Condition():

            resol += 1
            
            Chargement()

            u, d, Kglob, nombreIter, dincMax, temps = PhaseFieldSimulation.ResolutionIteration(simu=simu, tolConv=tolConv, maxIter=maxIter)

            max_d = d.max()
            f = np.sum(np.einsum('ij,j->i', Kglob[ddls_upper, :].toarray(), u, optimize='optimal'))

            if problem == "Benchmark":
                pourcentage = ud/umax
            else: 
                pourcentage = 0

            PhaseFieldSimulation.ResumeIteration(simu, resol, ud*1e6, d, nombreIter, dincMax,  temps, "µm", pourcentage, True)

            if "CompressionFCBA" in problem:
                if ud >= tresh2:
                    ud += inc2
                elif max_d<tresh0:
                    ud += inc0
                else:
                    ud += inc1
            else:
                if max_d<tresh0:
                    ud += inc0
                else:
                    ud += inc1

            # Detection si on a touché le bord
            if np.any(d[noeuds_bord] >= 0.95):
                bord +=1

            if plotIter:
                cb.remove()
                figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, ax=axIter)
                plt.pause(1e-12)
            
            if bord == 10:
                break

            displacement.append(ud)
            load.append(f)

            if nombreIter == maxIter:
                print(f'\nOn converge pas apres {nombreIter} itérations')
                break

        load = np.array(load)
        displacement = np.array(displacement)

        # Sauvegarde
        print()
        PostTraitement.Save_Load_Displacement(load, displacement, folder)
        PostTraitement.Save_Simu(simu, folder)
            
    else:
 
        load, displacement = PostTraitement.Load_Load_Displacement(folder)
        simu = PostTraitement.Load_Simu(folder)

    if plotEnergie:    
        PostTraitement.Plot_Energie(simu, load, displacement, Niter=400, folder=folder)

    if plotResult:
        Affichage.Plot_ResumeIter(simu, folder, None, None)

        Affichage.Plot_ForceDep(displacement*1e3, load*1e-6, 'ud en mm', 'f en kN/mm', folder)

        filenameDamage = f"{split} damage_n"
        # titleDamage = fr"$\phi$"
        titleDamage = f"{split}"


        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True,colorbarIsClose=True, folder=folder, filename=filenameDamage, title=titleDamage)


    if saveParaview:
        PostTraitement.Save_Simulation_in_Paraview(folder, simu, Niter=NParaview)
        if not solve:
            Tic.getGraphs(details=True)

    if makeMovie:
        PostTraitement.MakeMovie(folder, "damage", simu, Niter=NMovie, affichageMaillage=False, deformation=False, NiterFin=0, facteurDef=6)

    # Tic.getResume()

    if solve:
        Tic.getGraphs(folder, details=False)

    if showFig:
        plt.show()
    
    Tic.Clear()
    plt.close('all')