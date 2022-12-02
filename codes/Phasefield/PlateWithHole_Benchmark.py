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
problem = "Benchmark" # ["Benchmark","FCBA"]
test = False
solve = True

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
plotMesh = False
plotIter = False
plotResult = True
plotEnergie = True
showFig = False

# ----------------------------------------------
# Animation
# ----------------------------------------------
saveParaview = False; NParaview=200
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
optimMesh = False

# ----------------------------------------------
# Convergence
# ----------------------------------------------
maxIter = 1000
tolConv = 1e-0

# for tolConv in [1e-0, 1e-1, 1e-2]:
#     split = "Zhang"
# for split in ["Zhang"]:
#for split in ["Bourdin","Amor","Miehe","Stress"]: # Splits Isotropes
for split in ["He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes sans bourdin
# for split in ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes

    # ----------------------------------------------
    # Geometrie et chargement de la simulation
    # ----------------------------------------------
    if problem == "Benchmark":
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

        L = 9e-2
        h = L
        ep = 2e-2

        diam = 1e-2
        r = diam/2
        
        gc = 1.4/2
        # l_0 = 0.12e-3
        nL = 150
        l0 = L/nL

        u_max = "crack bord"

        inc0 = 8e-7; tresh0 = 0.2
        inc1 = 2e-7; tresh1 = 1
        inc2 = 2e-8; tresh2 = 130

        simpli2D = "CP" # ["CP","DP"]

        listInc = [inc0, inc1, inc2]
        listTresh = [tresh0, tresh1, tresh2]
        listOption = (["damage"]*2).append("displacement")

    # ----------------------------------------------
    # Taille d'elements
    # ----------------------------------------------
    if test:
        l0 *= 1
        if optimMesh:
            clD = l0*3
            clC = l0
        else:
            if problem == "Benchmark":
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
    # Matériau
    # ----------------------------------------------    
    if simpli2D == "CP":
        isCp = True
    else:
        isCp = False

    if comp == "Elas_Isot":
        E = 12e9
        v = 0.3
        comportement = Materials.Elas_Isot(2, E=E, v=v, contraintesPlanes=isCp, epaisseur=ep)

    elif comp == "Elas_IsotTrans":
        # El = 11580*1e6
        El = 12e9
        Et = 500*1e6
        Gl = 450*1e6
        vl = 0.02
        vt = 0.44
        v = 0
        comportement = Materials.Elas_IsotTrans(2, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, contraintesPlanes=isCp, epaisseur=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

    # Nom du dossier
    nomDossier = "PlateWithHole_" + problem
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comp, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solveur=solveur, test=test, closeCrack=False, v=v, nL=nL)
    
    if solve:

        # ----------------------------------------------
        # Construction du maillage
        # ----------------------------------------------
        point = Point()
        domain = Domain(point, Point(x=L, y=h), clD)
        circle = Circle(Point(x=L/2, y=h/2), diam, clC, isCreux=True)
        
        interfaceGmsh = Interface_Gmsh.Interface_Gmsh(affichageGmsh=False)
        
        if optimMesh:
            # Concentration de maillage sur la fissure
            if problem == "Benchmark":
                ecartZone = diam*1.5/2
            elif "FCBA" in problem:
                ecartZone = diam*1.5
            if split in ["Bourdin", "Amor"]:
                domainFissure = Domain(Point(y=h/2-ecartZone, x=0), Point(y=h/2+ecartZone, x=L), clC)
            else:
                domainFissure = Domain(Point(x=L/2-ecartZone, y=0), Point(x=L/2+ecartZone, y=h), clC)
            mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3", domainFissure)
        else:
            mesh = interfaceGmsh.Mesh_PlaqueAvecCercle2D(domain, circle, "TRI3")
        
        if plotMesh:
            Affichage.Plot_Maillage(mesh)
            # plt.show()

        # ----------------------------------------------
        # Modèle physique
        # ----------------------------------------------
        phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l0, solveur=solveur)
        materiau = Materials.Create_Materiau(phaseFieldModel, verbosity=False)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)
        
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

        ddls_upper = BoundaryCondition.Get_ddls_noeuds(2, "displacement", nodes_upper, ["y"])

        def Chargement(ud: float):
            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(node00, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])

        # Premier Chargement
        Chargement(0)

        simu.Resultats_Set_Resume_Chargement(u_max, listInc, listTresh, listOption)

        resol = 0
        bord = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True)

        def Condition():
            """Fonction qui traduit la condition de chargement"""
            if problem == "Benchmark":
                return ud <= u_max
            elif "FCBA" in problem:
                # On va charger jusqua la rupture
                return True

        while Condition():

            resol += 1
            
            Chargement(ud)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            simu.Save_Iteration()

            max_d = d.max()
            f = np.einsum('ij,j->', Kglob[ddls_upper, :].toarray(), u, optimize='optimal')

            if problem == "Benchmark":
                pourcentage = ud/u_max
            else: 
                pourcentage = 0

            simu.Resultats_Set_Resume_Iteration(resol, ud*1e6, "µm", pourcentage, True)
            
            # Si on converge pas on arrête la simulation
            if not convergence: break

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
            if np.any(d[noeuds_bord] >= 0.98):
                bord += 1
                if bord == 10:
                    # Si le bord à été touché depuis 5 iter on arrête la simulation
                    break

            if plotIter:
                cb.remove()
                figIter, axIter, cb = Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, ax=axIter)
                plt.pause(1e-12)

            displacement.append(ud)
            load.append(f)

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

        Affichage.Plot_ForceDep(displacement*1e3, load*1e-6, 'ud en mm', 'f en kN/mm', folder)

        filenameDamage = f"{split} damage_n"
        # titleDamage = fr"$\phi$"
        titleDamage = f"{split}"

        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=False, folder=folder, filename=filenameDamage, title=titleDamage)

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