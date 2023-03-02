import os
import matplotlib.pyplot as plt

import Folder
import PostTraitement
import Affichage
from Geom import *
import Materials
from Mesh import Mesh
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

dim = 2
folder = Folder.New_File(f"OptimMesh{dim}D", results=True)
if not os.path.exists(folder): os.makedirs(folder)
plotResult = True

rapport = 1/2

# Paramètres géométrie
L = 120;  #mm
h = 13
b = 13

P = 800 #N
lineLoad = P/h #N/mm
surfLoad = P/h/b #N/mm2

# Paramètres maillage
# meshSize = h/1
# meshSize = L/2
meshSize = h/10

if dim == 2:
    elemType = "QUAD4" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]
else:
    elemType = "HEXA8" # "TETRA4", "TETRA10", "HEXA8", "PRISM6"

# ----------------------------------------------
# Maillage
# ----------------------------------------------

pt1 = Point(0, 0)
pt2 = Point(L, 0)
pt3 = Point(L, h)
pt4 = Point(0, h)

points = [pt1, pt2, pt3, pt4]

circle = Circle(Point(x=L/2, y=h/2), h*0.3, isCreux=True)

# inclusions = [circle]
inclusions = []

interfaceGmsh = Interface_Gmsh(False)

# Fonction utilisée pour la construction du maillage
def DoMesh(refineGeom=None) -> Mesh:
    if dim == 2:
        return interfaceGmsh.Mesh_From_Points_2D(points, elemType, inclusions, [], refineGeom, meshSize)
    else:
        return interfaceGmsh.Mesh_From_Points_3D(points, [0,0,b], 3, elemType, inclusions, refineGeom, meshSize)

# construit le premier maillage
mesh = DoMesh()

# ----------------------------------------------
# Comportement et Simu
# ----------------------------------------------

comportement = Materials.Elas_Isot(dim, E=210000, v=0.3, epaisseur=b)
simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False)
simu.rho = 8100*1e-9

def DoSimu(i=0):    
    
    noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
    noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == L)
    
    simu.Bc_Init()
    if dim == 2:
        simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
    elif dim == 3:
        simu.add_dirichlet(noeuds_en_0, [0, 0, 0], ["x","y","z"], description="Encastrement")

    simu.add_surfLoad(noeuds_en_L, [-surfLoad], ["y"])

    simu.Solve()

    simu.Save_Iteration()

    # ----------------------------------------------
    # Calcul de l'erreur
    # ----------------------------------------------

    Wdef_e = simu.Get_Resultat("energy", nodeValues=False)
    Wdef = np.sum(Wdef_e)

    WdefLisse_e = simu.Get_Resultat("energy_smoothed", nodeValues=False)
    WdefLisse = np.sum(WdefLisse_e)

    erreur_e = np.abs(WdefLisse_e-Wdef_e).reshape(-1)/Wdef

    erreur = np.abs(Wdef-WdefLisse)/Wdef

    

    # ----------------------------------------------
    # Refine mesh
    # ----------------------------------------------

    groupElem = mesh.groupElem
    coordo = groupElem.coordo
    connect0 = groupElem.connect_e[:, range(groupElem.nbCorners)]
    index = np.append(np.arange(1, groupElem.nbCorners, 1, dtype=int), 0)
    connect1 = groupElem.connect_e[:,index]

    h_e_b = np.linalg.norm(coordo[connect1] - coordo[connect0], axis=2)
    h_e = np.mean(h_e_b, axis=1)

    c_e = (rapport-1)/erreur_e.max() * erreur_e + 1
    # c_e = (rapport-1) * erreur_e + 1

    # Affichage.Plot_Result(simu, h_e, nodeValues=False)
    # Affichage.Plot_Result(simu, c_e, nodeValues=False)

    meshSize_n = simu.Resultats_InterpolationAuxNoeuds(mesh, c_e * h_e)

    Affichage.Plot_Result(simu, erreur_e*100, nodeValues=True, title="erreur %", plotMesh=True)

    path = interfaceGmsh.Create_posFile(groupElem.coordo, meshSize_n, folder, f"simu{i}")

    return path, erreur

path = None

for i in range(5):

    mesh = DoMesh(path)
    simu.mesh = mesh

    path, erreur = DoSimu(i)

    print(f"{i} erreur = {erreur*100:.3} %")

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
Affichage.NouvelleSection("Post traitement")

# folder=""
if plotResult:
    tic = Tic()
    # simu.Resultats_Resume(True)
    # Affichage.Plot_Result(simu, "amplitude")
    # Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Affichage.Plot_Result(simu, "uy", deformation=True, nodeValues=False)        
    Affichage.Plot_Result(simu, "Svm", deformation=False, plotMesh=True, nodeValues=False)
    # Affichage.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)

    

    tic.Tac("Affichage","Affichage des figures", plotResult)


PostTraitement.Make_Paraview(folder, simu)

# Tic.Plot_History(details=True)
plt.show()