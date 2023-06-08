from BoundaryCondition import BoundaryCondition
import PostTraitement
import Folder
import Affichage
import Materials
from Geom import *
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic
from Mesh import Calc_projector

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Affichage.Clear()

# ----------------------------------------------
# Simulation
# ----------------------------------------------
dim = 3
simulation = "Shear" # "Shear" , "Tension"

if dim == 3:
    simulation += "_3D"
    ep = 0.1/1000
else:
    ep = 0

nomDossier = '_'.join([simulation,"Benchmark"])

test = True
solve = True

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
plotMesh = False
plotResult = True
plotEnergie = False
getFissure = False
showResult = True

# ----------------------------------------------
# Animation
# ----------------------------------------------
saveParaview = False; Nparaview=400
makeMovie = False

# ----------------------------------------------
# Maillage
# ----------------------------------------------
openCrack = True
optimMesh = True
updateMesh = False

# ----------------------------------------------
# Convergence
# ----------------------------------------------
maxIter = 1000
# tolConv = 0.0025
# tolConv = 0.005
tolConv = 1e-0

# ----------------------------------------------
# Comportement 
# ----------------------------------------------
comportement_str = "Elas_Isot" # "Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"
# regularisations = ["AT1", "AT2"]
regularisations = ["AT2"] # "AT1", "AT2"
solveurPhaseField = Simulations.PhaseField_Model.SolveurType.History

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes sans bourdin

# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
splits = ["Amor"]

nSplits = len(splits)
nRegus = len(regularisations)

regularisations = regularisations * nSplits
splits = np.repeat(splits, nRegus)

for split, regu in zip(splits, regularisations):

# splits = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*9 # Splits Anisotropes
# splits = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*3 # Splits Anisotropes
# # listTheta = [-0, -10, -20, -30, -45, -60]*5
# listTheta = [-70, -80, -90]*5
# listTheta.sort(); listTheta.reverse()
# for split, theta in zip(splits, listTheta):    

    # ----------------------------------------------
    # Construction du maillage
    # ----------------------------------------------

    # Paramètres géométrie
    L = 1e-3;  #m
    if comportement_str == "Elas_Anisot":
        theta = -45
        l0 = 8.5e-6
    else:
        theta = 0
        l0 = 1e-5 # taille fissure test FEMOBJECT ,7.5e-6, 1e-5        

    # Paramètres maillage
    if test:
        clC = l0 #taille maille test fem object        
    else:
        # On raffine pour avoir au moin 2 element par demie largeur de fissure
        clC = l0/2 #l0/2 2.5e-6 
        # taille = l0/1.2 #l0/2 2.5e-6
        # taille = 7.5e-6

    # Definition une zone pour raffiner le maillage
    if updateMesh:        
        clD = clC * 2
        refineDomain = None

    elif optimMesh:        
        zone = L*0.05
        if "Tension" in simulation:
            # On rafine horizontalement
            if comportement_str == "Elas_Isot":                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L/2+zone, z=ep), meshSize=clC)
            else:                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L*0.8, z=ep), meshSize=clC)
        if "Shear" in simulation:
            if split == "Bourdin":
                # On rafine en haut et en bas 
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L, z=ep), meshSize=clC)
            else:
                # On rafine en bas
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L/2+zone, z=ep), meshSize=clC)
        clD = clC * 3

    else:        
        clD = clC
        refineDomain = None

    # Construit le path vers le dossier en fonction des données du problèmes
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comportement_str, split=split, regu=regu, simpli2D='DP',tolConv=tolConv, solveur=solveurPhaseField, test=test, closeCrack= not openCrack, theta=theta, optimMesh=optimMesh)

    if solve:

        def DoMesh(refineDomain=None):

            elemType = "TRI3" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]            

            pt1 = Point()
            pt2 = Point(L)
            pt3 = Point(L,L)
            pt4 = Point(0,L)
            contour = PointsList([pt1, pt2, pt3, pt4], clD)

            if dim == 2:
                ptC1 = Point(0,L/2, isOpen=True)
                ptC2 = Point(L/2,L/2)
                cracks = [Line(ptC1, ptC2, clC, isOpen=True)]
            if dim == 3:
                ptC1 = Point(0,L/2,0, isOpen=True)
                ptC2 = Point(L/2,L/2, 0)
                ptC3 = Point(L/2,L/2, ep)
                ptC4 = Point(0,L/2, ep, isOpen=True)
                cracks = []

                l1 = Line(ptC1, ptC2, clC, True)
                l2 = Line(ptC2, ptC3, clC, False)
                l3 = Line(ptC3, ptC4, clC, True)
                l4 = Line(ptC4, ptC1, clC, True)
                
                cracks = [Contour([l1, l2, l3, l4])]
            
            if dim == 2:
                mesh = Interface_Gmsh().Mesh_2D(contour, cracks=cracks, elemType=elemType, refineGeom=refineDomain)
            elif dim == 3:
                # fichier = "/Users/matnoel/Desktop/gmsh_domain_single_edge_crack.msh"
                # mesh = Interface_Gmsh(True).Mesh_Import_msh(fichier)
                mesh = Interface_Gmsh(False, False).Mesh_3D(contour, [], [0,0,ep], 3, "TETRA4", cracks, refineGeom=refineDomain)

            return mesh
        
        mesh = DoMesh(refineDomain)
        
        if plotMesh:
            Affichage.Plot_Mesh(mesh)
            Affichage.Plot_Model(mesh, alpha=0)
            noeudsCracks = mesh.Nodes_Conditions(lambda x,y,z: (x<=L/2)&(y==L/2))            
            Affichage.Plot_Nodes(mesh, noeudsCracks, showId=True)            
            plt.show()

        # Simulation  -------------------------------------------------------------------------------------------        

        if comportement_str == "Elas_Isot":            
            comp = Materials.Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)
            Gc = 2.7e3 # J/m2
        elif comportement_str == "Elas_Anisot":
            if dim == 2:

                c11 = 65
                c22 = 260
                c33 = 30
                c12 = 20

                C_voigt = np.array([[c11, c12, 0],
                                    [c12, c22, 0],
                                    [0, 0, c33]])*1e9

                C_mandel = Materials.Displacement_Model.ApplyKelvinMandelCoefTo_Matrice(dim, C_voigt)

                
                theta_rad = theta * np.pi/180

                axis1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0])
                axis2 = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0])

            else:
                raise Exception("Pas implémenté")

            comp = Materials.Elas_Anisot(dim, C=C_voigt, axis1=axis1, contraintesPlanes=False)
            Gc = 1e3 # J/m2
        else:
            # comp = Elas_IsotTrans(2, El=210e9, Et=20e9, Gl=)
            raise Exception("Pas implémenté pour le moment")

        phaseFieldModel = Materials.PhaseField_Model(comp, split, regu, Gc=Gc, l_0=l0, solveur=solveurPhaseField)

        simu = Simulations.Simu_PhaseField(mesh, phaseFieldModel, verbosity=False)

        def Chargement(dep):
            """Renseignement des conditions limites"""

            simu.Bc_Init()            

            if not openCrack:
                simu.add_dirichlet(noeuds_Milieu, [1], ["d"], problemType="damage")            
            
            if "Shear" in simulation:
                # Conditions en déplacements à Gauche et droite
                simu.add_dirichlet(noeuds_Gauche, [0],["y"])
                simu.add_dirichlet(noeuds_Droite, [0],["y"])
                simu.add_dirichlet(noeuds_Haut, [dep,0], ["x","y"])
                # simu.add_dirichlet(noeuds_Haut, [dep], ["x"])

                # Conditions en déplacements en Bas
                if dim == 2:
                    simu.add_dirichlet(noeuds_Bas, [0,0],["x","y"])
                else:
                    simu.add_dirichlet(noeuds_Bas, [0,0,0],["x","y","z"])

            elif "Tension" in simulation:
                if dim == 2:
                    simu.add_dirichlet(noeuds_Haut, [0,dep], ["x","y"])                    
                elif dim == 3:
                    simu.add_dirichlet(noeuds_Haut, [0,dep,0], ["x","y","z"])
                simu.add_dirichlet(noeuds_Bas, [0],["y"])

            else:

                raise Exception("chargement inconnue pour cette simulation")

        Affichage.NewSection("Simulations")

        # ----------------------------------------------
        # Paramètres de chargement
        # ----------------------------------------------
        if "Shear" in simulation:
            u_inc = 5e-8 if test else 1e-8
            N = 400 if test else 2000 

            chargement = np.linspace(u_inc, u_inc*N, N, endpoint=True)
            
            listInc = [u_inc]
            listThreshold = [chargement[-1]]
            optionTreshold = ["displacement"]

        elif "Tension" in simulation:
            if test:
                u0 = 1e-7;  N0 = 40
                u1 = 1e-8;  N1 = 400
            else:
                u0 = 1e-8;  N0 = 400
                u1 = 1e-9;  N1 = 4000

            chargement = np.linspace(u0, u0*N0, N0, endpoint=True)
            chargement = np.append(chargement, np.linspace(u1, u1*N1, N1, endpoint=True)+chargement[-1])
            
            listInc = [u0, u1]
            listThreshold = [chargement[N0], chargement[N1]]
            optionTreshold = ["displacement"]*2

        if isinstance(comp, Materials.Elas_Anisot):

            if test:
                uinc0 = 12e-8
                uinc1 = 4e-8
            else:
                uinc0 = 6e-8
                uinc1 = 2e-8

            tresh0 = 0
            tresh1 = 0.6

            listInc = [uinc0, uinc1]
            listThreshold = [tresh0, tresh1]
            optionTreshold = ["damage"]*2
            # chargement = ["crack bord"]

        simu.Resultats_Set_Resume_Chargement(chargement[-1],listInc, listThreshold, optionTreshold)        

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        tic = Tic()
        
        def Condition():
            """Fonction qui traduit la condition de chargement"""
            if isinstance(comp, Materials.Elas_Isot):
                return dep < chargement[-1]
            else:
                # On va charger jusqua la rupture
                return True
        
        # INIT
        N = len(chargement)
        bord = 0 # variable pour savoir combien de l'endommagement à touché le bord
        deplacements=[]
        forces=[]
        uinc0 = chargement[0]
        dep = uinc0
        iter = 0
        while Condition():

            # Récupération des noeuds
            noeuds_Milieu = mesh.Nodes_Conditions(lambda x,y,z: (y==L/2) & (x<=L/2))
            noeuds_Haut = mesh.Nodes_Conditions(lambda x,y,z: y == L)
            noeuds_Bas = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
            noeuds_Gauche = mesh.Nodes_Conditions(lambda x,y,z: (x == 0) & (y>0) & (y<L))
            noeuds_Droite = mesh.Nodes_Conditions(lambda x,y,z: (x == L) & (y>0) & (y<L))

            # if noeuds_Milieu.size > 0:
            #     Affichage.Plot_Nodes(mesh, noeuds_Milieu, True)

            # Construit les noeuds du bord
            NoeudsBord=[]
            for noeuds in [noeuds_Bas,noeuds_Droite,noeuds_Haut]:
                NoeudsBord.extend(noeuds)

            # Récupération des ddls pour le calcul de la force
            if simulation == "Shear":
                ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["x"])
            else:
                ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["y"])

            Chargement(dep)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            simu.Save_Iteration()
            
            f = np.sum(Kglob[ddls_Haut, :] @ u)

            if isinstance(comp, Materials.Elas_Anisot):
                pourcentage = 0
            else:
                pourcentage = iter/N

            simu.Resultats_Set_Resume_Iteration(iter, dep*1e6, "µm", pourcentage, True)

            # Si on converge pas on arrête la simulation
            if not convergence: break

            if isinstance(comp, Materials.Elas_Anisot):
                if simu.damage.max() < tresh1:
                    dep += uinc0
                else:
                    dep += uinc1
            else:
                if iter == len(chargement)-1: break
                dep = chargement[iter]
    
            deplacements.append(dep)
            forces.append(f)            

            if np.any(simu.damage[NoeudsBord] >= 0.98):
                bord +=1
                if bord == 10:
                    # Si le bord à été touché depuis 5 iter on arrête la simulation
                    break

            iter += 1

            if updateMesh:

                meshSize_n = (clC-clD) * d + clD                

                # Affichage.Plot_Result(simu, meshSize_n)

                refineDomain = Interface_Gmsh().Create_posFile(simu.mesh.coordo, meshSize_n, folder)

                newMesh = DoMesh(refineDomain)

                if newMesh.Nn > simu.mesh.Nn:
                    
                    oldNodes = simu.mesh.Nodes_Conditions(lambda x,y,z: (x<L/2)&(y==L/2))
                    oldNodes = np.unique(oldNodes)
                    # oldNodes = oldNodes[np.argsort(simu.mesh.coordo[oldNodes,0])]

                    newNodes = newMesh.Nodes_Conditions(lambda x,y,z: (x<L/2)&(y==L/2))
                    newNodes = np.unique(newNodes)
                    # newNodes = newNodes[np.argsort(newMesh.coordo[newNodes,0])]

                    assert len(oldNodes) == len(newNodes)
                    
                    # axOld = Affichage.Plot_Mesh(simu.mesh, alpha=0)
                    # axNew = Affichage.Plot_Mesh(newMesh, alpha=0)
                    # for n in range(len(newNodes)):
                    #     if n > len(newNodes)//2:
                    #         c="black"
                    #     else:
                    #         c="red"
                    #     Affichage.Plot_Nodes(simu.mesh, [oldNodes[n]], True, ax=axOld, c=c)
                    #     # Affichage.Plot_Nodes(newMesh, [newNodes[n]], True, ax=axNew, c=c)
                    #     pass
                    
                    # for n in range(len(newNodes)):
                    #     plt.close("all")
                    #     Affichage.Plot_Nodes(simu.mesh, [oldNodes[n]], True)
                    #     Affichage.Plot_Nodes(newMesh, [newNodes[n]], True)
                    #     pass                    

                    proj = Calc_projector(simu.mesh, newMesh)
                    proj = proj.tolil()
                    proj[newNodes, :] = 0                    
                    proj[newNodes, oldNodes] = 1

                    newU = np.zeros((newMesh.Nn, 2))

                    for i in range(dim):
                        newU[:,i] = proj @ u.reshape(-1,2)[:,i]

                    newD = proj @ d

                    plt.close("all")
                    # Affichage.Plot_Result(simu.mesh, d, plotMesh=True)
                    # Affichage.Plot_Result(newMesh, newD, plotMesh=True)                    

                    Affichage.Plot_Result(simu.mesh, u.reshape(-1,2)[:,0])
                    Affichage.Plot_Result(newMesh, newU[:,0])

                    plt.pause(1e-12)
                    # Tic.Plot_History()

                    simu.mesh = newMesh
                    mesh = newMesh
                    simu.set_u_n("displacement", newU.reshape(-1))
                    simu.set_u_n("damage", newD.reshape(-1))


                    pass

            

            pass





                
        # ----------------------------------------------
        # Sauvegarde
        # ----------------------------------------------
        print()
        PostTraitement.Save_Load_Displacement(forces, deplacements, folder)
        simu.Save(folder)        

        forces = np.array(forces)
        deplacements = np.array(deplacements)

    else:
        # ----------------------------------------------
        # Chargement
        # ---------------------------------------------
        simu = Simulations.Load_Simu(folder)
        forces, deplacements = PostTraitement.Load_Load_Displacement(folder)

        

    # ----------------------------------------------
    # Post Traitement
    # ---------------------------------------------
    Affichage.NewSection("Affichage")
    

    if plotResult:

        Affichage.Plot_ResumeIter(simu, folder, None, None)

        Affichage.Plot_BoundaryConditions(simu)

        Affichage.Plot_ForceDep(deplacements*1e6, forces*1e-3, 'ud en µm', 'f en kN', folder)

        Affichage.Plot_Result(simu, "damage", nodeValues=True, plotMesh=False,deformation=False, folder=folder, filename="damage")
        

        # Affichage.Plot_Result(simu, "uy", folder=folder, deformation=True)
            
    if saveParaview:
        # ----------------------------------------------
        # Paraview
        # ---------------------------------------------
        PostTraitement.Make_Paraview(folder, simu, Nparaview)

    if makeMovie:
        # ----------------------------------------------
        # Movie
        # ---------------------------------------------
        PostTraitement.Make_Movie(folder, "damage", simu, deformation=True, NiterFin=0)
        # PostTraitement.MakeMovie(folder, "Svm", simu)        
        # PostTraitement.MakeMovie(filename, "Syy", simu, nodeValues=True, deformation=True)
            
    if plotEnergie:
        # ----------------------------------------------
        # Energie
        # ---------------------------------------------   
        # Affichage.Plot_Energie(simu, forces, deplacements, Niter=400, folder=folder)
        Affichage.Plot_Energie(simu, Niter=400, folder=folder)

    if getFissure:
        # ----------------------------------------------
        # Récupération de la fissure
        # ----------------------------------------------
        axDamage = Affichage.Plot_Result(simu, "damage")[1]

        coordoMesh = simu.mesh.coordo[:, :2] # coordonnées du maillage
        connectMesh = simu.mesh.connect

        # récupère la coordonnées des noeuds d'endommagement dans l'ordre
        knownNodes = []
        damagedElements = []
        unwantedElements = []    
        
        p0 = np.array([L/2, L/2])

        diamCercle = 2*meshSize
        tolDamageValide = 0.95

        vecteurs = []
        lignes = []
        listIter = []

        for iter in range(len(simu._results)):

            simu.Update_iter(iter)

            # récupére les endommagés pour l'iération
            noeuds = np.where(simu.damage >= 1)[0]        

            if len(noeuds) > 0:

                # Création des cercles
                nodesInCircles = []
                [nodesInCircles.extend(simu.mesh.Nodes_Circle(Circle(Point(x, y), diamCercle))) for x, y in zip(coordoMesh[noeuds, 0], coordoMesh[noeuds, 1])]

                # noeudsDans les cercles
                nodesInCircles = list(np.unique(nodesInCircles))            
                
                # dans les noeuds detectés enlève ceux déja connues
                nodesInCircles = list(set(nodesInCircles) - set(knownNodes))

                knownNodes.extend(nodesInCircles) # ajoute les noeuds inconnues

                if len(nodesInCircles) == 0: continue

                # axDamage.scatter(coordoMesh[nodesInCircles, 0], coordoMesh[nodesInCircles, 1], marker='+', c='white')

                # Récupère les noeuds avec un endommagement >= à tolDamageValide
                idxValables = np.where(simu.damage[nodesInCircles] >= tolDamageValide)
                nodesInCircles = np.array(nodesInCircles)[idxValables]

                if len(nodesInCircles) == 0: continue
                
                # récupère les elements associés a ces noeuds
                elements = simu.mesh.groupElem.Get_Elements_Nodes(nodesInCircles, exclusivement=False)            

                # récupère que les elements inconnues
                elements = list(set(elements) - set(damagedElements + unwantedElements))

                if len(elements) == 0: continue

                # Sur chaque element on va sommer la valeurs des endommagement et on va prendre le plus endommagé            
                localizedDamage = simu.mesh.Localises_sol_e(simu.damage)[elements] # localise l'endommagement sur tout les elements
                mostDamagedElement = elements[np.argmax(np.sum(localizedDamage, axis=1))] # element le plus endommagé
                
                # Renseigne l'itération ou un nouvelle element a ete detectée
                listIter.append(iter)

                # centrede gracité de cette element
                p1 = np.mean(coordoMesh[connectMesh[mostDamagedElement], :2], axis=0)

                vecteurs.append(p1-p0) # ajoute le vecteur de p0 à p1
                lignes.append(np.array([[p0],[p1]]).reshape(2,2)) # construit la ligne pour tracer sur la figure
                p0 = p1 # actualise le nouveau point 0

                # mets à jour les elements
                damagedElements.append(mostDamagedElement)
                elements.remove(mostDamagedElement)
                unwantedElements.extend(elements)                
        
        [Affichage.Plot_Elements(simu.mesh, connectMesh[element], dimElem=2, ax=axDamage, c='white') for element in damagedElements]

        lignes = np.array(lignes)
        collection = LineCollection(lignes, zorder=3, colors='white')
        axDamage.add_collection(collection)
        
        forces = forces[listIter]
        deplacements = deplacements[listIter]
        raideurs = forces/deplacements
        longeurs = np.linalg.norm(vecteurs, axis=1)
        ltot = np.sum(longeurs, axis=0)

        Longeurs = []
        for idx in range(len(longeurs)):
            
            if idx > 0:            
                lPrec = Longeurs[idx-1]
            else:
                lPrec = L/2

            Longeurs.append(longeurs[idx] + lPrec)

        Longeurs = np.array(Longeurs)
        
        aires = Longeurs*simu.phaseFieldModel.comportement.epaisseur

        plt.figure()
        plt.plot(aires, raideurs)
        # plt.show()

        delta_raideurs = raideurs[1:] - raideurs[:-1]
        delta_aires = aires[1:] - aires[:-1]

        G = 1/2 * delta_raideurs/delta_aires*deplacements[1:]**2

        plt.figure()
        plt.plot(aires[1:], G)

    Tic.Resume()

    if solve:
        Tic.Plot_History(folder, True)
    else:
        Tic.Plot_History(details=True)

    if showResult:
        plt.show()

    Tic.Clear()
    plt.close('all')

    if solve:
        del simu
        del mesh
    else:        
        del simu