# %%
from BoundaryCondition import BoundaryCondition

import PostTraitement as PostTraitement
import Folder
import Affichage as Affichage

import Materials
from Geom import *
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

import numpy as np
import matplotlib.pyplot as plt

# Affichage.Clear()

# ----------------------------------------------
# Simulation
# ----------------------------------------------
simulation = "Tension" # "Shear" , "Tension"
nomDossier = '_'.join([simulation,"Benchmark"])

test = False
solve = True

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
plotMesh = True
plotResult = True
plotEnergie = True
showResult = False

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
comportement_str = "Elas_Anisot" # "Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"
regularisation = "AT2" # "AT1", "AT2"
solveurPhaseField = Simulations.PhaseField_Model.SolveurType.History

# for split in ["Amor"]:
#for split in ["Bourdin","Amor","Miehe","Stress"]: # Splits Isotropes
#for split in ["He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes sans bourdin
#for split in ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]: # Splits Anisotropes sans bourdin
# listAnisotWithBourdin = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*9 # Splits Anisotropes
listAnisotWithBourdin = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*3 # Splits Anisotropes
# listTheta = [-0, -10, -20, -30, -45, -60]*5
listTheta = [-70, -80, -90]*5
listTheta.sort(); listTheta.reverse()
for split, theta in zip(listAnisotWithBourdin, listTheta):

    dim = 2

    # ----------------------------------------------
    # Construction du maillage
    # ----------------------------------------------

    # Paramètres géométrie
    L = 1e-3;  #m
    if comportement_str == "Elas_Anisot":
        # theta = -0        
        l0 = 8.5e-6
    else:
        theta = 0
        l0 = 1e-5 # taille fissure test FEMOBJECT ,7.5e-6, 1e-5        

    # Paramètres maillage
    if test:
        taille = l0 #taille maille test fem object
        # taille = 0.001  
        taille *= 3
    else:
        # On raffin pour avoir au moin 2 element par demie largeur de fissure
        taille = l0/2 #l0/2 2.5e-6 
        # taille = l0/1.2 #l0/2 2.5e-6
        # taille = 7.5e-6

    # Definition une zone pour raffiner le maillage
    if optimMesh:
        zone = L*0.05
        if simulation == "Tension":
            # On rafine horizontalement
            if comportement_str == "Elas_Isot":                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L/2+zone), taille=taille)
            else:                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L*0.8), taille=taille)
        elif simulation == "Shear":
            if split == "Bourdin":
                # On rafine en haut et en bas 
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L), taille=taille)
            else:
                # On rafine en bas
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L/2+zone), taille=taille)
        taille *= 3
    else:
        refineDomain = None

    # Construit le path vers le dossier en fonction des données du problèmes
    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comportement_str, split=split, regu=regularisation, simpli2D='DP',tolConv=tolConv, solveur=solveurPhaseField, test=test, closeCrack= not openCrack, v=0, theta=theta, optimMesh=optimMesh)

    if solve:

        elemType = "TRI3" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]

        interfaceGmsh = Interface_Gmsh(False)

        domain = Domain(Point(), Point(x=L, y=L), taille=taille)       

        line = Line(Point(y=L/2, isOpen=True), Point(x=L/2, y=L/2), taille=taille, isOpen=openCrack)
        line2 = Line(Point(y=L/4, isOpen=True), Point(x=3*L/4, y=L/4), taille=taille, isOpen=openCrack)

        # cracks = [line, line2]
        cracks = [line]

        mesh = interfaceGmsh.Mesh_Rectangle2D_Avec_Fissures(domain=domain, cracks=cracks, elemType=elemType, refineGeom=refineDomain)
        
        if plotMesh:
            Affichage.Plot_Maillage(mesh)
            # Affichage.Plot_Model(mesh)
            # noeudsCracks = list(mesh.Nodes_Line(line2))
            # noeudsCracks.extend(mesh.Nodes_Line(line))
            # Affichage.Plot_Noeuds(mesh, noeudsCracks, showId=True)
            # print(len(noeudsCracks))
            # plt.show()

        # Récupération des noeuds
        noeuds_Milieu = mesh.Nodes_Line(line)        
        noeuds_Haut = mesh.Nodes_Conditions(conditionY=lambda y: y == L)
        noeuds_Bas = mesh.Nodes_Conditions(conditionY=lambda y: y == 0)
        noeuds_Gauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
        noeuds_Droite = mesh.Nodes_Conditions(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

        # Construit les noeuds du bord
        NoeudsBord=[]
        for noeuds in [noeuds_Bas,noeuds_Droite,noeuds_Haut]:
            NoeudsBord.extend(noeuds)

        # Récupération des ddls pour le calcul de la force
        if simulation == "Shear":
            ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["x"])
        else:
            ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["y"])

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

                
                theta_rad = theta * np.pi/180

                axis1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0])
                axis2 = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0])

            else:
                raise "Pas implémenté"

            comp = Materials.Elas_Anisot(dim, C_voigt=C_voigt, axis1=axis1, contraintesPlanes=False)
            Gc = 1e3 # J/m2
        else:
            # comp = Elas_IsotTrans(2, El=210e9, Et=20e9, Gl=)
            raise "Pas implémenté pour le moment"

        phaseFieldModel = Materials.PhaseField_Model(comp, split, regularisation, Gc=Gc, l_0=l0, solveur=solveurPhaseField)

        materiau = Materials.Create_Materiau(phaseFieldModel, ro=1)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False)

        def Chargement(dep):
            """Renseignement des conditions limites"""

            simu.Bc_Init()

            if not openCrack:
                simu.add_dirichlet(noeuds_Milieu, [1], ["d"], problemType="damage")
            
            if simulation == "Shear":
                # Conditions en déplacements à Gauche et droite
                simu.add_dirichlet(noeuds_Gauche, [0],["y"])
                simu.add_dirichlet(noeuds_Droite, [0],["y"])
                simu.add_dirichlet(noeuds_Haut, [dep,0], ["x","y"])
                # simu.add_dirichlet(noeuds_Haut, [dep], ["x"])

                # Conditions en déplacements en Bas
                simu.add_dirichlet(noeuds_Bas, [0,0],["x","y"])

            elif simulation == "Tension":
                simu.add_dirichlet(noeuds_Haut, [0,dep], ["x","y"])
                simu.add_dirichlet(noeuds_Bas, [0],["y"])

            else:

                raise "chargement inconnue pour cette simulation"
            
        Chargement(0)

        Affichage.NouvelleSection("Simulations")

        # ----------------------------------------------
        # Paramètres de chargement
        # ----------------------------------------------
        if simulation == "Shear":
            u_inc = 5e-8 if test else 1e-8
            N = 400 if test else 2000 

            chargement = np.linspace(u_inc, u_inc*N, N, endpoint=True)
            
            listInc = [u_inc]
            listThreshold = [chargement[-1]]
            optionTreshold = ["displacement"]

        elif simulation == "Tension":
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
            chargement = ["crack bord"]

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
        dep = 0
        iter = 0
        while Condition():

            Chargement(dep)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter)

            simu.Save_Iteration()
            
            f = np.sum(np.einsum('ij,j->i', Kglob[ddls_Haut, :].toarray(), u, optimize='optimal'))

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
    Affichage.NouvelleSection("Affichage")
    

    if plotResult:

        Affichage.Plot_ResumeIter(simu, folder, None, None)

        Affichage.Plot_BoundaryConditions(simu)

        Affichage.Plot_ForceDep(deplacements*1e6, forces*1e-3, 'ud en µm', 'f en kN', folder)

        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, affichageMaillage=False,deformation=False, folder=folder, filename="damage")
        

        # Affichage.Plot_Result(simu, "dy", folder=folder, deformation=True)
            
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
        # PostTraitement.MakeMovie(filename, "Syy", simu, valeursAuxNoeuds=True, deformation=True)
            
    if plotEnergie:
        # ----------------------------------------------
        # Energie
        # ---------------------------------------------   
        # Affichage.Plot_Energie(simu, forces, deplacements, Niter=400, folder=folder)
        Affichage.Plot_Energie(simu, Niter=400, folder=folder)

    Tic.getResume()

    if solve:
        Tic.getGraphs(folder, True)
    else:
        Tic.getGraphs(details=False)

    if showResult:
        plt.show()

    Tic.Clear()
    plt.close('all')
# %%
