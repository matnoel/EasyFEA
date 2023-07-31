from TicTac import Tic
import Materials
from BoundaryCondition import BoundaryCondition
from Geom import *
import Display as Display
from Interface_Gmsh import Interface_Gmsh
from Mesh import Mesh, Calc_projector
import Simulations
import PostProcessing as PostProcessing
import Folder

import matplotlib.pyplot as plt

# Affichage.Clear()

# ----------------------------------------------
# Simulation
# ----------------------------------------------
problem = "FCBA" # ["Benchmark","FCBA"]
dim = 2
if dim == 3:
    problem += "_3D"

test = True
solve = True

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
plotMesh = False
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
comp = "Elas_IsotTrans" # ["Elas_Isot", "Elas_IsotTrans"]

svType = Materials.PhaseField_Model.SolverType
solveur = svType.History # ["History", "HistoryDamage", "BoundConstrain"]

# ----------------------------------------------
# Maillage
# ----------------------------------------------
optimMesh = True
updateMesh = False
damagedNodes = []

# ----------------------------------------------
# Convergence
# ----------------------------------------------
maxIter = 1000
tolConv = 1e-0

# ----------------------------------------------
# Configurations
# ----------------------------------------------

# for tolConv in [1e-0, 1e-1, 1e-2]:
#     split = "Zhang"

# splits = ["Zhang"]
splits = ["He"]
# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes

regularisations = ["AT2"] # ["AT1", "AT2"]
# regularisations = ["AT1", "AT2"]

nSplit = len(splits)
nRegu = len(regularisations)

regularisations = regularisations * nSplit
splits = np.repeat(splits, nRegu)

for split, regu in zip(splits, regularisations):

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
        
        # gc = 3000
        gc = 0.07 * 1000
        # l_0 = 0.12e-3
        # nL = 50
        nL = 100
        # nL = 180
        l0 = L/nL

        u_max = 1e-3

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
            # clD = l0*3*coef
            # clC = l0*coef
            
            clD = l0*4
            clC = l0
            
        elif updateMesh:
            clD = l0*4
            clC = l0
        else:
            if "Benchmark" in problem:
                clD = 0.25e-3
                clC = 0.12e-3
            else:
                clD = l0*2
                clC = l0
    else:        
        if optimMesh:
            clD = l0*4
            # clD = l0*10
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
        comportement = Materials.Elas_Isot(dim, E=E, v=v, planeStress=isCp, thickness=ep)

    elif comp == "Elas_IsotTrans":
        # El = 11580*1e6        
        El = 12e9
        Et = 500*1e6
        Gl = 450*1e6
        vl = 0.02
        vt = 0.44
        v = 0
        comportement = Materials.Elas_IsotTrans(dim, El=El, Et=Et, Gl=Gl, vl=vl, vt=vt, planeStress=isCp, thickness=ep, axis_l=np.array([0,1,0]), axis_t=np.array([1,0,0]))

    # Nom du dossier
    nomDossier = "PlateWithHole_" + problem
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comp, split=split, regu=regu, simpli2D=simpli2D, optimMesh=optimMesh, tolConv=tolConv, solveur=solveur, test=test, closeCrack=False, nL=nL)
    
    if solve:

        # ----------------------------------------------
        # Maillage
        # ----------------------------------------------
        if optimMesh:
            ecartZone = diam*1.5/2
            if split in ["Bourdin", "Amor"]:
                domainFissure = Domain(Point(y=h/2-ecartZone, x=0), Point(y=h/2+ecartZone, x=L), clC)
            else:
                domainFissure = Domain(Point(x=L/2-ecartZone, y=0), Point(x=L/2+ecartZone, y=h), clC)
        else:
            domainFissure = None

        def DoMesh(refineGeom=None) -> Mesh:

            point = Point()
            domain = Domain(point, Point(x=L, y=h), clD)
            circle = Circle(Point(x=L/2, y=h/2), diam, clD, isCreux=True)
            
            interfaceGmsh = Interface_Gmsh(False, False)

            if dim == 2:
                mesh = interfaceGmsh.Mesh_2D(domain, [circle], "TRI3", refineGeom=refineGeom)
            elif dim == 3:
                mesh = interfaceGmsh.Mesh_3D(domain, [circle], [0,0,ep], 4, "HEXA8", refineGeom=refineGeom)

            return mesh
        
        mesh = DoMesh(domainFissure)
        
        if plotMesh:
            Display.Plot_Mesh(mesh)
            plt.show()        

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

        phaseFieldModel = Materials.PhaseField_Model(comportement, split, regu, gc, l0, solver=solveur)

        simu = Simulations.Simu_PhaseField(mesh, phaseFieldModel, verbosity=False)

        simu.Results_Set_Bc_Summary(u_max, listInc, listTresh, listOption)

        ud=0
        resol = 0
        bord = 0
        displacement = []
        load = []

        if plotIter:
            figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True)

            arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)
            if "Benchmark" in problem:
                figLoad, axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]')
            elif "FCBA" in problem:
                figLoad, axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]')

        def Chargement(ud: float):
            
            # Récupérations des noeuds
            nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y==0)
            nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h)            
            nodesX00 = mesh.Nodes_Conditions(lambda x,y,z: (x==0) & (y==0))
            nodesY00 = mesh.Nodes_Conditions(lambda x,y,z: (y==0) & (z==0))

            simu.Bc_Init()
            simu.add_dirichlet(nodes_lower, [0], ["y"])
            simu.add_dirichlet(nodesX00, [0], ["x"])
            simu.add_dirichlet(nodes_upper, [-ud], ["y"])
            if dim == 3:
                simu.add_dirichlet(nodesY00, [0], ["z"])

        def Condition():
            """Fonction qui traduit la condition de chargement"""
            if "Benchmark" in problem :
                return ud <= u_max
            elif "FCBA" in problem:
                # On va charger jusqua la rupture
                # return True
                return ud <= u_max

        while Condition():

            resol += 1
            
            noeuds_bord = simu.mesh.Nodes_Tags(["L0","L1","L2","L3"])
            nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y==h) 

            ddls_upper = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_upper, ["y"])

            if dim == 3:
                if resol > 1500:
                    break
            
            Chargement(ud)
            # Affichage.Plot_BoundaryConditions(simu)
            # plt.show()

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            # Si on converge pas on arrête la simulation
            if not convergence: break

            if updateMesh:

                meshSize_n = (clC-clD) * d + clD                

                # Affichage.Plot_Result(simu, meshSize_n)
                # nodes = np.where(d>=1e-3)[0]
                # test = [n for n in nodes if n not in damagedNodes]
                # if len(test) == 0: continue

                refineDomain = Interface_Gmsh().Create_posFile(simu.mesh.coordo, meshSize_n, folder)

                newMesh = DoMesh(refineDomain)

                if newMesh.Nn > simu.mesh.Nn:

                    # Affichage.Plot_Mesh(simu.mesh)
                    # Affichage.Plot_Mesh(newMesh)

                    proj = Calc_projector(simu.mesh, newMesh)

                    newU = np.zeros((newMesh.Nn, 2))

                    for i in range(dim):
                        newU[:,i] = proj @ u.reshape(-1,2)[:,i]

                    newD = proj @ d
                    
                    # Affichage.Plot_Result(simu.mesh, d, plotMesh=True)
                    # Affichage.Plot_Result(newMesh, newD, plotMesh=True)                    

                    # Affichage.Plot_Result(simu.mesh, u.reshape(-1,2)[:,0])
                    # Affichage.Plot_Result(newMesh, newU[:,0])

                    # plt.pause(1e-12)
                    # Tic.Plot_History()

                    simu.mesh = newMesh
                    mesh = newMesh
                    simu.set_u_n("displacement", newU.reshape(-1))
                    simu.set_u_n("damage", newD.reshape(-1))

                    # pass
                    # plt.close("all")

            simu.Save_Iter()

            max_d = d.max()
            f = np.sum(Kglob[ddls_upper, :] @ u)

            if "Benchmark" in problem:
                pourcentage = ud/u_max
            else: 
                pourcentage = 0

            if "Benchmark" in problem:
                simu.Results_Set_Iteration_Summary(resol, ud*1e6, "µm", pourcentage, True)
            elif "FCBA" in problem:
                simu.Results_Set_Iteration_Summary(resol, ud*1e3, "mm", pourcentage, True)
           
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
                figIter, axIter, cb = Display.Plot_Result(simu, "damage", nodeValues=True, ax=axIter)

                plt.figure(figIter)

                plt.pause(1e-12)

                arrayDisplacement, arrayLoad = np.array(displacement), np.array(load)

                if "Benchmark" in problem:
                    axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e6, arrayLoad*1e-6, 'ud [µm]', 'f [kN/mm]', ax=axLoad)[1]
                elif "FCBA" in problem:
                    axLoad = Display.Plot_Load_Displacement(arrayDisplacement*1e3, arrayLoad*1e-3, 'ud [mm]', 'f [kN]', ax=axLoad)[1]

                plt.figure(axLoad.figure)

                plt.pause(1e-12)

            

        load = np.array(load)
        displacement = np.array(displacement)

        # ----------------------------------------------
        # Sauvegarde
        # ----------------------------------------------
        print()
        PostProcessing.Save_Load_Displacement(load, displacement, folder)
        simu.Save(folder)
            
    else:
        # ----------------------------------------------
        # Chargement
        # ----------------------------------------------
        load, displacement = PostProcessing.Load_Load_Displacement(folder)
        simu = Simulations.Load_Simu(folder)

    # ----------------------------------------------
    # Post Traitement
    # ---------------------------------------------

    if plotEnergie:
        Display.Plot_Energy(simu, load, displacement, Niter=400, folder=folder)

    if plotResult:
        Display.Plot_BoundaryConditions(simu)

        Display.Plot_Iter_Summary(simu, folder, None, None)

        if "Benchmark" in problem:
            Display.Plot_Load_Displacement(displacement*1e3, load*1e-6, 'ud [mm]', 'f [kN/mm]', folder)
        elif "FCBA" in problem:
            Display.Plot_Load_Displacement(displacement*1e3, load*1e-3, 'ud [mm]', 'f [kN]', folder)

        Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=True, folder=folder, filename="damage")

    if saveParaview:
        PostProcessing.Make_Paraview(folder, simu, Niter=NParaview)        

    if makeMovie:
        # Affichage.Plot_Result(simu, "damage", deformation=True, facteurDef=20)
        # plt.show()
        PostProcessing.Make_Movie(folder, "ux", simu, Niter=NMovie, plotMesh=False, deformation=False, NiterFin=0, factorDef=1.5)

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