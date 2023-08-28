from BoundaryCondition import BoundaryCondition
import PostProcessing
import Folder
import Display
import Materials
from Geom import *
from Interface_Gmsh import Interface_Gmsh, ElemType
import Simulations
from TicTac import Tic
from Mesh import Calc_projector

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Display.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------
problem = "Tension" # "Shear" , "Tension"

dim = 2
test = False
solve = False

folderName = '_'.join([problem,"Benchmark"])

if dim == 3:
    folderName += "_3D"
    ep = 0.1/1000
else:
    ep = 0

# ----------------------------------------------
# Post processing
# ----------------------------------------------
plotMesh = False
plotResult = True
plotEnergy = False
getCrack = False
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
updateMesh = False # non-validated implementation

# ----------------------------------------------
# Convergence
# ----------------------------------------------
maxIter = 1000
# tolConv = 0.0025
# tolConv = 0.005
tolConv = 1e-0

# ----------------------------------------------
# Material 
# ----------------------------------------------
materialType = "Elas_Isot" # "Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"
# regularisations = ["AT1", "AT2"]
regularisations = ["AT2"] # "AT1", "AT2"
solveurPhaseField = Simulations.PhaseField_Model.SolverType.History

# splits = ["Bourdin","Amor","Miehe","Stress"] # Splits Isotropes
# splits = ["He","AnisotStrain","AnisotStress","Zhang"] # Splits Anisotropes sans bourdin
# splits = ["Bourdin","Amor","Miehe","Stress","He","AnisotStrain","AnisotStress","Zhang"]
splits = ["Bourdin"]

nSplits = len(splits)
nRegus = len(regularisations)

regularisations = regularisations * nSplits
splits = np.repeat(splits, nRegus)

# splits = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*9 # Splits Anisotropes
# splits = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]*3 # Splits Anisotropes
# # listTheta = [-0, -10, -20, -30, -45, -60]*5
# listTheta = [-70, -80, -90]*5
# listTheta.sort(); listTheta.reverse()
# for split, theta in zip(splits, listTheta):

for split, regu in zip(splits, regularisations):

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------    
    L = 1e-3;  #m
    if materialType == "Elas_Anisot":
        theta = -45
        l0 = 8.5e-6
    else:
        theta = 0
        l0 = 1e-5 # taille fissure test FEMOBJECT ,7.5e-6, 1e-5        

    # meshSize
    if test:
        clC = l0 #taille maille test fem object
    else:
        # On raffine pour avoir au moin 2 element par demie largeur de fissure
        clC = l0/2 #l0/2 2.5e-6 
        # taille = l0/1.2 #l0/2 2.5e-6
        # taille = 7.5e-6
    
    if updateMesh:        
        clD = clC * 2
        refineDomain = None

    elif optimMesh:        
        zone = L*0.05
        if "Tension" in problem:
            # Horizontal mesh refinement
            if materialType == "Elas_Isot":                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L/2+zone, z=ep), meshSize=clC)
            else:                
                refineDomain = Domain(Point(x=L/2-zone, y=L/2-zone), Point(x=L, y=L*0.8, z=ep), meshSize=clC)
        if "Shear" in problem:
            if split == "Bourdin":
                # Upper and lower right mesh refinement
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L, z=ep), meshSize=clC)
            else:
                # Lower right mesh refinement
                refineDomain = Domain(Point(x=L/2-zone, y=0), Point(x=L, y=L/2+zone, z=ep), meshSize=clC)
        clD = clC * 3

    else:        
        clD = clC
        refineDomain = None

    # Builds the path to the folder based on the problem data
    folder = Folder.PhaseField_Folder(folder=folderName, material=materialType, split=split, regu=regu, simpli2D='DP',tolConv=tolConv, solveur=solveurPhaseField, test=test, closeCrack= not openCrack, theta=theta, optimMesh=optimMesh)

    if solve:

        def DoMesh(refineDomain=None):

            elemType = ElemType.TRI3 # ["TRI3", "TRI6", "QUAD4", "QUAD8"]            

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

                meshSize = clD if optimMesh else clC

                l1 = Line(ptC1, ptC2, meshSize, True)
                l2 = Line(ptC2, ptC3, meshSize, False)
                l3 = Line(ptC3, ptC4, meshSize, True)
                l4 = Line(ptC4, ptC1, meshSize, True)
                
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
            Display.Plot_Mesh(mesh)
            Display.Plot_Model(mesh, alpha=0)
            noeudsCracks = mesh.Nodes_Conditions(lambda x,y,z: (x<=L/2)&(y==L/2))            
            Display.Plot_Nodes(mesh, noeudsCracks, showId=True)            
            plt.show()

        # ----------------------------------------------
        # Material
        # ----------------------------------------------
        if materialType == "Elas_Isot":            
            material = Materials.Elas_Isot(dim, E=210e9, v=0.3, planeStress=False)
            Gc = 2.7e3 # J/m2
        elif materialType == "Elas_Anisot":
            if dim == 2:

                c11 = 65
                c22 = 260
                c33 = 30
                c12 = 20

                C_voigt = np.array([[c11, c12, 0],
                                    [c12, c22, 0],
                                    [0, 0, c33]])*1e9

                C_mandel = Materials._Displacement_Model.KelvinMandel_Matrix(dim, C_voigt)

                
                theta_rad = theta * np.pi/180

                axis1 = np.array([np.cos(theta_rad), np.sin(theta_rad), 0])
                axis2 = np.array([-np.sin(theta_rad), np.cos(theta_rad), 0])

            else:
                raise Exception("Not implemented")

            material = Materials.Elas_Anisot(dim, C=C_voigt, axis1=axis1, planeStress=False)
            Gc = 1e3 # J/m2
        else:            
            raise Exception("Pas implémenté pour le moment")

        pfm = Materials.PhaseField_Model(material, split, regu, Gc=Gc, l0=l0, solver=solveurPhaseField)

        # ----------------------------------------------
        # Boundary conditions
        # ----------------------------------------------
        if "Shear" in problem:
            u_inc = 5e-8 if test else 1e-8
            N = 400 if test else 2000 

            loadings = np.linspace(u_inc, u_inc*N, N, endpoint=True)
            
            listInc = [u_inc]
            listThreshold = [loadings[-1]]
            optionTreshold = ["displacement"]

        elif "Tension" in problem:
            if test:
                u0 = 1e-7;  N0 = 40
                u1 = 1e-8;  N1 = 400
            else:
                u0 = 1e-8;  N0 = 400
                u1 = 1e-9;  N1 = 4000

            loadings = np.linspace(u0, u0*N0, N0, endpoint=True)
            loadings = np.append(loadings, np.linspace(u1, u1*N1, N1, endpoint=True)+loadings[-1])
            
            listInc = [u0, u1]
            listThreshold = [loadings[N0], loadings[N1]]
            optionTreshold = ["displacement"]*2

        if isinstance(material, Materials.Elas_Anisot):

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

        def Loading(dep):
            """Boundary conditions"""

            simu.Bc_Init()            

            if not openCrack:
                simu.add_dirichlet(nodes_crack, [1], ["d"], problemType="damage")            
            
            if "Shear" in problem:
                # Left and right travel conditions
                simu.add_dirichlet(nodes_left, [0],["y"])
                simu.add_dirichlet(nodes_right, [0],["y"])
                simu.add_dirichlet(nodes_upper, [dep,0], ["x","y"])
                simu.add_dirichlet(nodes_lower, [0]*dim, simu.Get_directions())

            elif "Tension" in problem:
                if dim == 2:
                    simu.add_dirichlet(nodes_upper, [0,dep], ["x","y"])                    
                elif dim == 3:
                    simu.add_dirichlet(nodes_upper, [0,dep,0], ["x","y","z"])
                simu.add_dirichlet(nodes_lower, [0],["y"])

        # ----------------------------------------------
        # Simulation
        # ----------------------------------------------
        simu = Simulations.Simu_PhaseField(mesh, pfm, verbosity=False)
        simu.Results_Set_Bc_Summary(loadings[-1],listInc, listThreshold, optionTreshold)

        tic = Tic()
        
        def Condition():
            """Loading stop condition."""
            if isinstance(material, Materials.Elas_Isot):
                return dep < loadings[-1]
            else:
                # We're going to charge until we break
                return True
        
        # INIT
        N = len(loadings)
        nDetect = 0 # variable for how many times the damage has touched the edge
        displacements=[]
        forces=[]
        uinc0 = loadings[0]
        dep = uinc0
        iter = 0
        while Condition():

            # Node recovery
            nodes_crack = mesh.Nodes_Conditions(lambda x,y,z: (y==L/2) & (x<=L/2))
            nodes_upper = mesh.Nodes_Conditions(lambda x,y,z: y == L)
            nodes_lower = mesh.Nodes_Conditions(lambda x,y,z: y == 0)
            nodes_left = mesh.Nodes_Conditions(lambda x,y,z: (x == 0) & (y>0) & (y<L))
            nodes_right = mesh.Nodes_Conditions(lambda x,y,z: (x == L) & (y>0) & (y<L))

            # Builds edge nodes
            nodes_Edges=[]
            for nodes in [nodes_lower,nodes_right,nodes_upper]:
                nodes_Edges.extend(nodes)

            # Récupération des ddls pour le calcul de la force
            if problem == "Shear":
                dofs_upper = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_upper, ["x"])
            else:
                dofs_upper = BoundaryCondition.Get_dofs_nodes(2, "displacement", nodes_upper, ["y"])

            Loading(dep)

            u, d, Kglob, convergence = simu.Solve(tolConv=tolConv, maxIter=maxIter, convOption=1)

            simu.Save_Iter()
            
            f = np.sum(Kglob[dofs_upper, :] @ u)

            if isinstance(material, Materials.Elas_Anisot):
                pourcentage = 0
            else:
                pourcentage = iter/N

            simu.Results_Set_Iteration_Summary(iter, dep*1e6, "µm", pourcentage, True)

            # If it doesn't converge, stop the simulation
            if not convergence: break

            # loading
            if isinstance(material, Materials.Elas_Anisot):
                if simu.damage.max() < tresh1:
                    dep += uinc0
                else:
                    dep += uinc1
            else:
                if iter == len(loadings)-1: break
                dep = loadings[iter]    
            displacements.append(dep)
            forces.append(f)

            # check for damaged edges
            if np.any(simu.damage[nodes_Edges] >= 0.98):
                nDetect +=1
                if nDetect == 10:
                    # If the edge has been touched for 10 iter, stop the simulation
                    break
            iter += 1

            if updateMesh:

                meshSize_n = (clC-clD) * d + clD                

                # Display.Plot_Result(simu, meshSize_n)

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
                    
                    # axOld = Display.Plot_Mesh(simu.mesh, alpha=0)
                    # axNew = Display.Plot_Mesh(newMesh, alpha=0)
                    # for n in range(len(newNodes)):
                    #     if n > len(newNodes)//2:
                    #         c="black"
                    #     else:
                    #         c="red"
                    #     Display.Plot_Nodes(simu.mesh, [oldNodes[n]], True, ax=axOld, c=c)
                    #     # Display.Plot_Nodes(newMesh, [newNodes[n]], True, ax=axNew, c=c)
                    #     pass
                    
                    # for n in range(len(newNodes)):
                    #     plt.close("all")
                    #     Display.Plot_Nodes(simu.mesh, [oldNodes[n]], True)
                    #     Display.Plot_Nodes(newMesh, [newNodes[n]], True)
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
                    # Display.Plot_Result(simu.mesh, d, plotMesh=True)
                    # Display.Plot_Result(newMesh, newD, plotMesh=True)                    

                    Display.Plot_Result(simu.mesh, u.reshape(-1,2)[:,0])
                    Display.Plot_Result(newMesh, newU[:,0])

                    plt.pause(1e-12)
                    # Tic.Plot_History()

                    simu.mesh = newMesh
                    mesh = newMesh
                    simu.set_u_n("displacement", newU.reshape(-1))
                    simu.set_u_n("damage", newD.reshape(-1))
                
        # ----------------------------------------------
        # Saving
        # ----------------------------------------------
        print()
        PostProcessing.Save_Load_Displacement(forces, displacements, folder)
        simu.Save(folder)        

        forces = np.array(forces)
        displacements = np.array(displacements)

    else:
        # ----------------------------------------------
        # Loading
        # ---------------------------------------------
        simu = Simulations.Load_Simu(folder)
        forces, displacements = PostProcessing.Load_Load_Displacement(folder)        

    # ----------------------------------------------
    # PostProcessing
    # ---------------------------------------------

    if plotResult:

        Display.Plot_Iter_Summary(simu, folder, None, None)
        Display.Plot_BoundaryConditions(simu)
        Display.Plot_Load_Displacement(displacements*1e6, forces*1e-3, 'ud en µm', 'f en kN', folder)
        Display.Plot_Result(simu, "damage", nodeValues=True, plotMesh=False,deformation=False, folder=folder, filename="damage")
        # Display.Plot_Result(simu, "uy", folder=folder, deformation=True)
            
    if saveParaview:
        PostProcessing.Make_Paraview(folder, simu, Nparaview)

    if makeMovie:
        PostProcessing.Make_Movie(folder, "damage", simu, deformation=True, NiterFin=0)
        # PostProcessing.MakeMovie(folder, "Svm", simu)        
        # PostProcessing.MakeMovie(filename, "Syy", simu, nodeValues=True, deformation=True)
            
    if plotEnergy:        
        # Display.Plot_Energie(simu, forces, displacements, Niter=400, folder=folder)
        Display.Plot_Energy(simu, Niter=400, folder=folder)

    if getCrack:
        # ----------------------------------------------
        # Calculating crack length
        # ----------------------------------------------
        axDamage = Display.Plot_Result(simu, "damage")[1]

        coordoMesh = simu.mesh.coordo[:, :2] # coordonnées du maillage
        connectMesh = simu.mesh.connect

        # récupère la coordonnées des noeuds d'endommagement dans l'ordre
        knownNodes = []
        damagedElements = []
        unwantedElements = []    
        
        p0 = np.array([L/2, L/2])

        diamCercle = 2*l0
        maxDamage = 0.95

        vecteurs = []
        lines = []
        listIter = []

        for iter in range(len(simu.results)):

            simu.Update_Iter(iter)

            # recovers damaged parts for ieration
            nodes = np.where(simu.damage >= 1)[0]        

            if len(nodes) > 0:

                # Creating circles
                nodesInCircles = []
                [nodesInCircles.extend(simu.mesh.Nodes_Circle(Circle(Point(x, y), diamCercle))) for x, y in zip(coordoMesh[nodes, 0], coordoMesh[nodes, 1])]

                # nodesIn circles
                nodesInCircles = list(np.unique(nodesInCircles))            
                
                # in detected nodes removes those already known
                nodesInCircles = list(set(nodesInCircles) - set(knownNodes))

                knownNodes.extend(nodesInCircles) # adds unknown nodes

                if len(nodesInCircles) == 0: continue

                # axDamage.scatter(coordoMesh[nodesInCircles, 0], coordoMesh[nodesInCircles, 1], marker='+', c='white')

                # Recovers nodes with damage >= at tolDamageValide
                idxValables = np.where(simu.damage[nodesInCircles] >= maxDamage)
                nodesInCircles = np.array(nodesInCircles)[idxValables]

                if len(nodesInCircles) == 0: continue
                
                # retrieves elements associated with these nodes
                elements = simu.mesh.groupElem.Get_Elements_Nodes(nodesInCircles, exclusively=False)
                # recovers only unknown elements
                elements = list(set(elements) - set(damagedElements + unwantedElements))

                if len(elements) == 0: continue

                # On each element we will sum the damage values and take the most damaged one.
                localizedDamage = simu.mesh.Locates_sol_e(simu.damage)[elements] # locate damage on all elements
                mostDamagedElement = elements[np.argmax(np.sum(localizedDamage, axis=1))] # most damaged element
                
                # Indicates the iteration where a new element has been detected
                listIter.append(iter)

                # Gravity center of this element
                p1 = np.mean(coordoMesh[connectMesh[mostDamagedElement], :2], axis=0)

                vecteurs.append(p1-p0) # adds the vector from p0 to p1
                lines.append(np.array([[p0],[p1]]).reshape(2,2)) # build the line to draw on the figure
                p0 = p1 # updates the new point 0

                # update elements
                damagedElements.append(mostDamagedElement)
                elements.remove(mostDamagedElement)
                unwantedElements.extend(elements)                
        
        [Display.Plot_Elements(simu.mesh, connectMesh[element], dimElem=2, ax=axDamage, c='white') for element in damagedElements]

        lines = np.array(lines)
        collection = LineCollection(lines, zorder=3, colors='white')
        axDamage.add_collection(collection)
        
        forces = forces[listIter]
        displacements = displacements[listIter]
        stifness = forces/displacements
        lenghts = np.linalg.norm(vecteurs, axis=1)
        crackLength = np.sum(lenghts, axis=0)

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