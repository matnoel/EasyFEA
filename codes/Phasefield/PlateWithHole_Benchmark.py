from TicTac import Tic
import Materials
from BoundaryCondition import BoundaryCondition
from Geom import *
import Affichage as Affichage
import Interface_Gmsh as Interface_Gmsh
import Simulations
import PostTraitement as PostTraitement
import Folder

import matplotlib.pyplot as plt

# Affichage.Clear()
# mpirun -np 4 python3 PlateWithHole_Benchmark.py

# Options

test = False
solve = True

plotMesh = False
plotIter = False
plotResult = True
showFig = False
plotEnergie = True

saveParaview = False; NParaview=200
makeMovie = False; NMovie = 300


problem = "FCBA" # ["Benchmark","FCBA"]
comp = "Elas_IsotTrans" # ["Elas_Isot", "Elas_IsotTrans"]
regu = "AT2" # ["AT1", "AT2"]
svType = Materials.PhaseField_Model.SolveurType
solveur = svType.History # ["History", "HistoryDamage", "BoundConstrain"]
optimMesh = True

useNumba = True

# Convergence
maxIter = 1000
tolConv = 1e-0

# TODO Faire la convergence sur l'energie ?

for tolConv in [1e-0, 1e-1, 1e-2]:
    split = "Zhang"
# for split in ["Zhang"]:
# for split in ["Bourdin","Amor","Miehe","Stress"]:
# for split in ["Zhang","AnisotStress_PM","He","AnisotStrain","AnisotStress"]:

    # Data

    if problem == "Benchmark":
        L=15e-3
        h=30e-3
        ep=1
        diam=6e-3

        gc = 1.4
        nL=0
        l_0 = 0.12e-3

        if comp == "Elas_Isot":
            # u_max = 25e-6
            u_max = 35e-6    
        else:
            u_max = 80e-6

        inc0 = 8e-8
        inc1 = 2e-8

        tresh0 = 0.6
        tresh1 = 1

        simpli2D = "DP" # ["CP","DP"]

        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]
        listOption = ["damage"]*len(listTresh)

    elif "FCBA" in problem:

        L=9e-2
        h=L
        ep=2e-2
        
        u_max=0

        diam=1e-2
        r=diam/2
        
        gc = 1.4/2
        # l_0 = 0.12e-3
        nL=100
        l_0 = L/nL

        inc0 = 8e-7
        inc1 = 2e-7
        inc2 = 2e-8

        tresh0 = 0.2
        tresh1 = 1
        tresh2 = 130

        simpli2D = "CP" # ["CP","DP"]

        listInc = [inc0, inc1, inc2]
        listTresh = [tresh0, tresh1, tresh2]
        listOption = (["damage"]*2).append("displacement")

    # Matériau

    if simpli2D == "CP":
        isCp = True
    else:
        isCp = False

    if comp == "Elas_Isot":
        E=12e9
        v=0.3
        comportement = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=isCp, epaisseur=ep)

    elif comp == "Elas_IsotTrans":
        # El=11580*1e6
        El=12e9
        Et=500*1e6
        Gl=450*1e6
        vl=0.02
        vt=0.44
        v=0
        comportement = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=isCp, epaisseur=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

    # Taille element
    
    if test:

        if optimMesh:
            clD = l_0*3*5
            clC = l_0*5
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
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comp, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solveur=solveur, test=test, closeCrack=False, v=v, nL=nL)
    
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
            elif "FCBA" in problem:
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

        phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l_0, solveur=solveur)
        materiau = Materials.Create_Materiau(phaseFieldModel, verbosity=False)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False, useNumba=useNumba)
        
        # Récupérations des noeuds
        B_lower = Line(point,Point(x=L)); nodes_lower = mesh.Nodes_Line(B_lower)
        B_upper = Line(Point(y=h),Point(x=L, y=h)); nodes_upper = mesh.Nodes_Line(B_upper)
        B_left = Line(point,Point(y=h)); nodes_left = mesh.Nodes_Line(B_left)
        B_right = Line(Point(x=L),Point(x=L, y=h)); nodes_right = mesh.Nodes_Line(B_right)
        node00 = mesh.Nodes_Point(point)

        noeuds_bord = []
        for ns in [nodes_lower, nodes_upper, nodes_left, nodes_right]:
            noeuds_bord.extend(ns)
        noeuds_bord = np.unique(noeuds_bord)
        
        ud=0
        damage_t=[]

        ddls_upper = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_upper, ["y"])

        def Chargement():
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(node00, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])

        # Premier Chargement
        Chargement()

        simu.Resultats_Set_Resume_Chargement(u_max, listInc, listTresh, listOption)

        resol = 0
        bord = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True)

        def Condition():
            if problem == "Benchmark":
                return ud <= u_max
            elif "FCBA" in problem:
                return simu.damage[noeuds_bord].max() <= 1
                # return simu.damage.max() <= 0.5

        while Condition():

            resol += 1
            
            Chargement()

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            simu.Save_Iteration()

            # Si on converge pas on arrête la simulation
            if not convergence: break

            max_d = d.max()
            f = np.einsum('ij,j->', Kglob[ddls_upper, :].toarray(), u, optimize='optimal')

            if problem == "Benchmark":
                pourcentage = ud/u_max
            else: 
                pourcentage = 0
            simu.Resultats_Set_Resume_Iteration(resol, ud*1e6, "µm", pourcentage, True)

            if "FCBA" in problem:
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
            if np.any(d[noeuds_bord] >= 1):
                bord +=1
                if bord == 10:
                    break

            if plotIter:
                cb.remove()
                figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, ax=axIter)
                plt.pause(1e-12)

            displacement.append(ud)
            load.append(f)

        load = np.array(load)
        displacement = np.array(displacement)

        # Sauvegarde
        print()
        PostTraitement.Save_Load_Displacement(load, displacement, folder)
        simu.Save(folder)
            
    else:
 
        load, displacement = PostTraitement.Load_Load_Displacement(folder)
        simu = Simulations.Load_Simu(folder)

    if plotEnergie:    
        PostTraitement.Plot_Energie(simu, load, displacement, Niter=400, folder=folder)

    if plotResult:
        Affichage.Plot_ResumeIter(simu, folder, None, None)

        Affichage.Plot_ForceDep(displacement*1e3, load*1e-6, 'ud en mm', 'f en kN/mm', folder)

        filenameDamage = f"{split} damage_n"
        # titleDamage = fr"$\phi$"
        titleDamage = f"{split}"


        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True,colorbarIsClose=True, folder=folder, filename=filenameDamage, title=titleDamage)
        
    # Affichage.Plot_BoundaryConditions(simu)
    # plt.show()

    if saveParaview:
        PostTraitement.Make_Paraview(folder, simu, Niter=NParaview)
        if not solve:
            Tic.getGraphs(details=True)

    if makeMovie:
        # Affichage.Plot_Result(simu, "damage", deformation=True, facteurDef=20)
        # plt.show()
        PostTraitement.Make_Movie(folder, "damage", simu, Niter=NMovie, affichageMaillage=False, deformation=True, NiterFin=0, facteurDef=20)

    # Tic.getResume()

    if solve:
        Tic.getGraphs(folder, details=True)

    if showFig:
        plt.show()
    
    Tic.Clear()
    plt.close('all')