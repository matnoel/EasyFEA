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

# ----------------------------------------------
# Simulation
# ----------------------------------------------
problem = "FCBA" # ["Benchmark","FCBA"]
dim = 3
if dim == 3:
    problem += "_3D"

test = True
solve = True

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
plotMesh = True
plotIter = True
plotResult = True
plotEnergie = False
showFig = True

# ----------------------------------------------
# Animation
# ----------------------------------------------
saveParaview = False; NParaview=300
makeMovie = False; NMovie = 200

# ----------------------------------------------
# Comportement 
# ----------------------------------------------
comp = "Elas_Isot" # ["Elas_Isot", "Elas_IsotTrans"]
regu = "AT2" # ["AT1", "AT2"]
svType = Materials.PhaseField_Model.SolveurType
solveur = svType.History # ["History", "HistoryDamage", "BoundConstrain"]

# ----------------------------------------------
# Maillage
# ----------------------------------------------
optimMesh = True

# ----------------------------------------------
# Convergence
# ----------------------------------------------
maxIter = 1000
tolConv = 1e-0

# for tolConv in [1e-0, 1e-1, 1e-2]:
#     split = "Zhang"

for split in ["Zhang"]:
#for split in ["Bourdin","Amor","Miehe","Stress"]: # Splits Isotropes
# for split in ["He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes sans bourdin
# for split in ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes

    # ----------------------------------------------
    # Geometrie et chargement de la simulation
    # ----------------------------------------------
    if "Benchmark" in problem:
        L = 15e-3
        h = 30e-3
        ep = 1 
        diam = 6e-3

        gc = 1.4
        nL = 0
        l0 = 0.12e-3

        if comp == "Elas_Isot":
            # u_max = 25e-6
            u_max = 35e-6    
        else:
            u_max = 80e-6

        inc0 = 8e-8; tresh0 = 0.6
        inc1 = 2e-8; tresh1 = 1

        simpli2D = "DP" # ["CP","DP"]

        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]
        listOption = ["damage"]*len(listTresh)

    elif "FCBA" in problem:

        L = 4.5e-2
        h = 9e-2
        ep = 2e-2

        diam = 1e-2
        r = diam/2
        
        gc = 3000
        # l_0 = 0.12e-3
        nL = 100
        l0 = h/nL        

        u_max = 2e-3

        inc0 = 8e-6; tresh0 = 0.2
        inc1 = 2e-6; tresh1 = 0.6

        simpli2D = "CP" # ["CP","DP"]

        listInc = [inc0, inc1]
        listTresh = [tresh0, tresh1]
        listOption = ["damage"]*2

    # ----------------------------------------------
    # Taille d'elements
    # ----------------------------------------------
    if test:
        coef = 1 if dim == 2 else 3
        if optimMesh:
            clD = l0*3*coef
            clC = l0*coef
        else:
            if "Benchmark" in problem:
                clD = 0.25e-3
                clC = 0.12e-3
            else:
                clD = l0*2
                clC = l0
    else:        
        if optimMesh:
            clD = l0*2
            clC = l0/2
        else:
            clD = l0/2
            clC = l0/2

    # ----------------------------------------------
    # Modlèle en deplacement
    # ----------------------------------------------
    if dim == 2:
        if simpli2D == "CP":
            isCp = True
        else:
            isCp = False
    else:
        isCp = False

    if comp == "Elas_Isot":
        E = 12e9
        v = 0.3
        comportement = Materials.Elas_Isot(dim, E=E, v=v, contraintesPlanes=isCp, epaisseur=ep)

    elif comp == "Elas_IsotTrans":
        # El = 11580*1e6
        cc = 1
        El = 12e9*cc
        Et = 500*1e6*cc
        Gl = 450*1e6*cc
        vl = 0.02
        vt = 0.44
        v = 0
        comportement = Materials.Elas_IsotTrans(dim, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=isCp, epaisseur=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

    # Nom du dossier
    nomDossier = "PlateWithHole_" + problem
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comp, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solveur=solveur, test=test, closeCrack=False, v=v, nL=nL)
    
    if solve:

        # ----------------------------------------------
        # Maillage
        # ----------------------------------------------
        point = Point()
        domain = Domain(point, Point(x=L, y=h), clD)
        circle = Circle(Point(x=L/2, y=h/2), diam, clC, isCreux=True)
        
        interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
        
        if optimMesh:
            # Concentration de maillage sur la fissure
            if "Benchmark" in problem:
                ecartZone = diam*1.5/2
            elif "FCBA" in problem:
                ecartZone = diam

            if split in ["Bourdin", "Amor"]:
                domainFissure = Domain(Point(y=h/2-ecartZone, x=0), Point(y=h/2+ecartZone, x=L), clC)
            else:
                domainFissure = Domain(Point(x=L/2-ecartZone, y=0), Point(x=L/2+ecartZone, y=h), clC)
            if dim == 2:
                mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3", domainFissure)
            elif dim == 3:
                mesh = interfaceGmsh.Mesh_PlaqueAvecCercle3D(domain, circle, extrude=[0,0,ep], nCouches=4, elemType="HEXA8", refineGeom=domainFissure)

        else:
            mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")
        
        if plotMesh:
            Affichage.Plot_Maillage(mesh)
            # plt.show()

        # Récupérations des noeuds
        nodes_lower = mesh.Nodes_Conditions(conditionY = lambda y: y==0)
        nodes_upper = mesh.Nodes_Conditions(conditionY = lambda y: y==h)
        nodes_left = mesh.Nodes_Conditions(lambda x: x==0)
        nodes_right = mesh.Nodes_Conditions(lambda x: x==L)
        nodesX00 = mesh.Nodes_Conditions(lambda x: x==0, lambda y: y==0)
        nodesY00 = mesh.Nodes_Conditions(conditionY=lambda y: y==0, conditionZ = lambda z: z==0)

        noeuds_bord = []
        for ns in [nodes_lower, nodes_upper, nodes_left, nodes_right]:
            noeuds_bord.extend(ns)
        noeuds_bord = np.unique(noeuds_bord)

        # ----------------------------------------------
        # Matériau
        # ----------------------------------------------
        if dim == 2:
            if simpli2D == "CP":
                isCp = True
            else:
                isCp = False
        else:
            isCp = False       

        phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l0, solveur=solveur)
        materiau = Materials.Create_Materiau(phaseFieldModel, verbosity=False)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)        

        ddls_upper = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_upper, ["y"])

        def Chargement(ud: float):
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodesX00, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])
            if dim == 3:
                simu.add_dirichlet(nodesY00, [0], ["z"])

        # Premier Chargement
        Chargement(0)

        # Affichage.Plot_BoundaryConditions(simu)
        # plt.show()

        simu.Resultats_Set_Resume_Chargement(u_max, listInc, listTresh, listOption)

        ud=0
        resol = 0
        bord = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True)

            arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)
            if "Benchmark" in problem:
                figLoad, axLoad = Affichage.Plot_ForceDep(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]')
            elif "FCBA" in problem:
                figLoad, axLoad = Affichage.Plot_ForceDep(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]')

        def Condition():
            """Fonction qui traduit la condition de chargement"""
            if "Benchmark" in problem :
                return ud <= u_max
            elif "FCBA" in problem:
                # On va charger jusqua la rupture
                return True

        while Condition():

            resol += 1

            if dim == 3:
                if resol > 200:
                    break
            
            Chargement(ud)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            simu.Save_Iteration()

            max_d = d.max()
            f = float(np.einsum('ij,j->', Kglob[ddls_upper, :].toarray(), u, optimize='optimal'))

            if "Benchmark" in problem:
                pourcentage = ud/u_max
            else: 
                pourcentage = 0

            if "Benchmark" in problem:
                simu.Resultats_Set_Resume_Iteration(resol, ud*1e6, "µm", pourcentage, True)
            elif "FCBA" in problem:
                simu.Resultats_Set_Resume_Iteration(resol, ud*1e3, "mm", pourcentage, True)            
            
            # Si on converge pas on arrête la simulation
            if not convergence: break

            if max_d<tresh0:
                ud += inc0
            else:
                ud += inc1

            # Detection si on a touché le bord
            if np.any(d[noeuds_bord] >= 0.98):
                bord += 1
                if bord == 10:                    
                    break

            displacement.append(ud)
            load.append(f)

            if plotIter:
                cb.remove()
                figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, ax=axIter)

                plt.figure(figIter)

                plt.pause(1e-12)

                arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)

                if "Benchmark" in problem:
                    axLoad = Affichage.Plot_ForceDep(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]', ax=axLoad)[1]
                elif "FCBA" in problem:
                    axLoad = Affichage.Plot_ForceDep(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]', ax=axLoad)[1]

                plt.figure(axLoad.figure)

                plt.pause(1e-12)

        load = np.array(load)
        displacement = np.array(displacement)

        # ----------------------------------------------
        # Sauvegarde
        # ----------------------------------------------
        print()
        PostTraitement.Save_Load_Displacement(load, displacement, folder)
        simu.Save(folder)
            
    else:
        # ----------------------------------------------
        # Chargement
        # ----------------------------------------------
        load, displacement = PostTraitement.Load_Load_Displacement(folder)
        simu = Simulations.Load_Simu(folder)

    # ----------------------------------------------
    # Post Traitement
    # ---------------------------------------------

    if plotEnergie:
        Affichage.Plot_Energie(simu, load, displacement, Niter=400, folder=folder)

    if plotResult:
        Affichage.Plot_BoundaryConditions(simu)

        Affichage.Plot_ResumeIter(simu, folder, None, None)

        if "Benchmark" in problem:
            Affichage.Plot_ForceDep(displacement*1e3, load*1e-6, 'ud [mm]', 'f [kN/mm]', folder)
        elif "FCBA" in problem:
            Affichage.Plot_ForceDep(displacement*1e3, load*1e-3, 'ud [mm]', 'f [kN]', folder)

        filenameDamage = f"{split} damage_n"
        # titleDamage = fr"$\phi$"
        titleDamage = f"{split}"

        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=False, folder=folder, filename=filenameDamage, title=titleDamage)

    if saveParaview:
        PostTraitement.Make_Paraview(folder, simu, Niter=NParaview)        

    if makeMovie:
        # Affichage.Plot_Result(simu, "damage", deformation=True, facteurDef=20)
        # plt.show()
        PostTraitement.Make_Movie(folder, "damage", simu, Niter=NMovie, affichageMaillage=False, deformation=True, NiterFin=0, facteurDef=20)

    # Tic.getResume()

    if solve:
        Tic.Plot_History(folder, details=True)
    else:        
        Tic.Plot_History(details=True)

    if showFig:
        plt.show()
    
    Tic.Clear()
    plt.close('all')

    if solve:
        del simu
        del mesh
    else:        
        del simu