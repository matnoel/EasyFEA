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

simulation = "Shear" # "Shear" , "Tension"
nomDossier = '_'.join([simulation,"Benchmark"])

test = True
solve = True

pltMesh = True
plotResult = True
showResult = True
plotEnergie = True

saveParaview = False; Nparaview=400
makeMovie = False

useNumba = True

# Data --------------------------------------------------------------------------------------------

comportement_str = "Elas_Isot" # "Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"
regularisation = "AT2" # "AT1", "AT2"
solveurs = Simulations.PhaseField_Model.SolveurType
solveur = solveurs.History
openCrack = True
optimMesh = False

for split in ["Amor"]:
# for split in ["Bourdin","Amor","Miehe","Stress"]:
# for split in ["He","AnisotStrain","AnisotStress","Zhang"]:
# for split in ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]:

    maxIter = 1000
    # tolConv = 0.0025
    # tolConv = 0.005
    tolConv = 1e-0

    dim = 2

    # Paramètres géométrie
    L = 1e-3;  #m
    if comportement_str == "Elas_Anisot":
        theta = -30
        l0 = 0.0085e-3
        Gc = 10e-3 * 1e-3 * 1e3
    else:
        theta = 0
        l0 = 1e-5 # taille fissure test femobject ,7.5e-6, 1e-5
        Gc = 2.7e3 # J/m2

    # Paramètres maillage
    if test:
        taille = l0 #taille maille test fem object
        # taille = 0.001  
        taille *= 10      
    else:
        taille = l0/2 #l0/2 2.5e-6
        taille = l0/1.2 #l0/2 2.5e-6
        # taille = 7.5e-6

    if optimMesh:
        zone = L*0.05
        if simulation == "Tension":
            refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L/2+zone), taille=taille)
        elif simulation == "Shear":
            if split == "Bourdin":
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L), taille=taille)
            else:                
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L/2+zone), taille=taille)
        taille *= 3
    else:
        refineDomain = None

    folder = Folder.PhaseField_Folder(dossierSource=nomDossier, comp=comportement_str, split=split, regu=regularisation, simpli2D='DP',tolConv=tolConv, solveur=solveur, test=test, closeCrack= not openCrack, v=0, theta=theta, optimMesh=optimMesh)

    # Construction du modele et du maillage --------------------------------------------------------------------------------

    if solve:

        elemType = "TRI3" # ["TRI3", "TRI6", "QUAD4", "QUAD8"]

        interfaceGmsh = Interface_Gmsh(False)

        domain = Domain(Point(), Point(x=L, y=L), taille=taille)       

        line = Line(Point(y=L/2, isOpen=True), Point(x=L/2, y=L/2), taille=taille, isOpen=openCrack)
        line2 = Line(Point(y=L/4, isOpen=True), Point(x=3*L/4, y=L/4), taille=taille, isOpen=openCrack)

        # cracks = [line, line2]
        cracks = [line]

        mesh = interfaceGmsh.Mesh_Rectangle2D_Avec_Fissures(domain=domain, cracks=cracks, elemType=elemType, refineGeom=refineDomain)
        
        if pltMesh:
            Affichage.Plot_Model(mesh)
            noeudsCracks = list(mesh.Nodes_Line(line2))
            noeudsCracks.extend(mesh.Nodes_Line(line))
            Affichage.Plot_Noeuds(mesh, noeudsCracks, showId=True)
            print(len(noeudsCracks))
            plt.show()

        # Récupère les noeuds qui m'interessent
        noeuds_Milieu = mesh.Nodes_Line(line)
        
        noeuds_Haut = mesh.Nodes_Conditions(conditionY=lambda y: y == L)
        noeuds_Bas = mesh.Nodes_Conditions(conditionY=lambda y: y == 0)
        noeuds_Gauche = mesh.Nodes_Conditions(conditionX=lambda x: x == 0, conditionY=lambda y: y>0 and y <L)
        noeuds_Droite = mesh.Nodes_Conditions(conditionX=lambda x: x == L, conditionY=lambda y: y>0 and y <L)

        NoeudsBord=[]
        for noeuds in [noeuds_Bas,noeuds_Droite,noeuds_Haut]:
            NoeudsBord.extend(noeuds)

        if simulation == "Shear":
            ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["x"])
        else:
            ddls_Haut = BoundaryCondition.Get_ddls_noeuds(2, "displacement", noeuds_Haut, ["y"])

        # Simulation  -------------------------------------------------------------------------------------------
        

        if comportement_str == "Elas_Isot":
            comp = Materials.Elas_Isot(dim, E=210e9, v=0.3, contraintesPlanes=False)
        elif comportement_str == "Elas_Anisot":

            if dim == 2:

                c11 = 65
                c22 = 260
                c33 = 30
                c12 = 20

                C_voigt = np.array([[c11, c12, 0],
                                    [c12, c22, 0],
                                    [0, 0, c33]])

                
                theta_rad = theta * np.pi/180

                axis1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0])
                axis2 = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0])

            else:
                raise "Pas implémenté"

            comp = Materials.Elas_Anisot(dim, C_voigt=C_voigt, axis1=axis1, contraintesPlanes=False)
        else:
            # comp = Elas_IsotTrans(2, El=210e9, Et=20e9, Gl=)
            raise "Pas implémenté pour le moment"

        phaseFieldModel = Materials.PhaseField_Model(comp, split, regularisation, Gc=Gc, l_0=l0, solveur=solveur)

        materiau = Materials.Create_Materiau(phaseFieldModel, ro=1)

        simu = Simulations.Create_Simu(mesh, materiau, verbosity=False, useNumba=useNumba)        

        # Renseignement des conditions limites
        def Chargement(dep):

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
            else:
                simu.add_dirichlet(noeuds_Haut, [0,dep], ["x","y"])
                simu.add_dirichlet(noeuds_Bas, [0],["y"])
            
        Chargement(0)

        Affichage.NouvelleSection("Simulations")

        if simulation == "Shear":
            if test:
                u_inc = 5e-8
                N = 400
                # N = 10
                # u_inc = 5e-7
            else:
                u_inc = 1e-8
                N = 2000

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

            uinc0 = 6e-6; tresh0 = 0
            uinc1 = 2e-7; tresh1 = 0.3

            listInc = [uinc0, uinc1]
            listThreshold = [tresh0, tresh1]
            optionTreshold = ["damage"]*2
            chargement = ["crack bord"]

        simu.Resultats_Set_Resume_Chargement(chargement[-1],listInc, listThreshold, optionTreshold)

        damage_t=[]
        uglob_t=[]

        dep = 0
        iter = 0

        tic = Tic()

        bord = 0        
        
        deplacements=[]
        forces=[]

        def Condition():

            testEndommagementBord = simu.damage[NoeudsBord].max() <= 0.98

            if isinstance(comp, Materials.Elas_Isot):

                return (testEndommagementBord or dep < chargement[-1])
            
            elif isinstance(comp, Materials.Elas_Anisot):

                return testEndommagementBord

            else:

                raise "Pas implémenté"

        N = len(chargement)

        while Condition():

            nombreIter=0
            convergence = False
            damage = simu.damage

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

            if np.any(damage[NoeudsBord] >= 0.98):
                bord +=1
                if bord == 5:
                    break

            iter += 1
                
        # Sauvegarde
        print()
        PostTraitement.Save_Load_Displacement(forces, deplacements, folder)
        simu.Save(folder)        

        forces = np.array(forces)
        deplacements = np.array(deplacements)

    else:   

        simu = Simulations.Load_Simu(folder)

        forces, deplacements = PostTraitement.Load_Load_Displacement(folder)
        
        simu.useNumba = useNumba

        # Affichage.Plot_Maillage(simu.mesh)
        # plt.show()

    # ------------------------------------------------------------------------------------------------------
    Affichage.NouvelleSection("Affichage")

    if makeMovie:
        PostTraitement.Make_Movie(folder, "damage", simu, deformation=True, NiterFin=0)
        # PostTraitement.MakeMovie(folder, "Svm", simu)        
        # PostTraitement.MakeMovie(filename, "Syy", simu, valeursAuxNoeuds=True, deformation=True)

    if plotResult:

        Affichage.Plot_ResumeIter(simu, folder, None, None)

        Affichage.Plot_BoundaryConditions(simu)

        Affichage.Plot_ForceDep(deplacements*1e6, forces*1e-3, 'ud en µm', 'f en kN', folder)

        Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, affichageMaillage=False,deformation=False, folder=folder, filename="damage")
        

        # Affichage.Plot_Result(simu, "dy", folder=folder, deformation=True)
            
    if saveParaview:
        PostTraitement.Make_Paraview(folder, simu, Nparaview)
            
    if plotEnergie:    
        PostTraitement.Plot_Energie(simu, forces, deplacements, Niter=400, folder=folder)

    Tic.getResume()

    if solve:
        Tic.getGraphs(folder, False)
    else:
        Tic.getGraphs(details=False)

    if showResult:
        plt.show()

    Tic.Clear()
    plt.close('all')
# %%
