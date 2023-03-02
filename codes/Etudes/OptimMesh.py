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

rapport = 1/10
cible = 0.01/2
iterMax = 15

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
meshSize = h/4

# TODO permettre de réutiliser le .geo pour construire la geométrie ?

if dim == 2:
    elemType = "QUAD4" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]
else:
    elemType = "TETRA4" # "TETRA4", "TETRA10", "HEXA8", "PRISM6"

# ----------------------------------------------
# Maillage
# ----------------------------------------------

pt1 = Point(0, 0)
pt2 = Point(L, 0)
pt3 = Point(L, h)
pt4 = Point(0, h)

points = PointsList([pt1, pt2, pt3, pt4], meshSize)

# circle = Circle(Point(x=h, y=h/2), h*0.3, isCreux=True)
# inclusions = [circle]

# xC=h; tC=h/3
# ptC1 = Point(xC-tC/2, h/2-tC/2, r=tC/2) 
# ptC2 = Point(xC+tC/2, h/2-tC/2)
# ptC3 = Point(xC+tC/2, h/2+tC/2, r=tC/2)
# ptC4 = Point(xC-tC/2, h/2+tC/2)
# carre = PointsList([ptC1, ptC2, ptC3, ptC4], meshSize, isCreux=True)
# inclusions = [carre]

inclusions = []
# nL = 10
# nH = 2
# cL = L/(2*nL)
# cH = h/(2*nH)
# for i in range(nL):
#     x = cL + cL*(2*i)
#     for j in range(nH):
#         y = cH + cH*(2*j)

#         ptd1 = Point(x-cL/2, y-cH/2)
#         ptd2 = Point(x+cL/2, y+cH/2)

#         domain = Domain(ptd1, ptd2, meshSize, isCreux=True)

#         inclusions.append(domain)

interfaceGmsh = Interface_Gmsh(False)

# Fonction utilisée pour la construction du maillage
def DoMesh(refineGeom=None) -> Mesh:
    if dim == 2:
        return interfaceGmsh.Mesh_From_Points_2D(points, elemType, inclusions, [], refineGeom)
    else:
        return interfaceGmsh.Mesh_From_Points_3D(points, [0,0,b], 1, elemType, inclusions, refineGeom)

# construit le premier maillage
mesh = DoMesh()

Affichage.Plot_Mesh(mesh)

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

    indexesSegments = groupElem.indexesSegments

    segments_e = groupElem.connect[:, indexesSegments]

    h_e_b = np.linalg.norm(coordo[segments_e[:,:,1]] - coordo[segments_e[:,:,0]], axis=2)
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

erreur = 1
i = -1
while erreur >= cible and i < iterMax:

    i += 1   

    mesh = DoMesh(path)
    simu.mesh = mesh

    if i > 0:
        os.remove(path)

    path, erreur = DoSimu(i)

    print(f"{i} erreur = {erreur*100:.3} %")

if i > 0:
    os.remove(path)

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

# PostTraitement.Make_Movie(folder, "Svm", simu, plotMesh=False)

# Tic.Plot_History(details=True)
plt.show()