import os
import matplotlib.pyplot as plt

import Folder
import PostTraitement
import Affichage
from Geom import *
import Materials
from Mesh import Mesh, Calc_projector, Calc_meshSize_n
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

Affichage.Clear()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

dim = 3
folder = Folder.New_File(f"OptimMesh{dim}D", results=True)
if not os.path.exists(folder): os.makedirs(folder)
plotResult = False
plotErreur = False
plotProj = False

# coef = 1/10
coef = 1/2
cible = 1/100 if dim == 2 else 0.04
iterMax = 20

# Paramètres géométrie
L = 120;  #mm
h = 13
b = 13

P = 800 #N
lineLoad = P/h #N/mm
surfLoad = P/h/b #N/mm2

# Paramètres maillage
meshSize = h/3

# TODO permettre de réutiliser le .geo pour construire la geométrie ?

if dim == 2:
    elemType = "TRI6" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]
else:
    elemType = "PRISM6" # "TETRA4", "TETRA10", "HEXA8", "PRISM6"

# ----------------------------------------------
# Maillage
# ----------------------------------------------

pt1 = Point(0, 0, isOpen=True)
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
nL = 20
nH = 3
cL = L/(2*nL)
cH = h/(2*nH)
for i in range(nL):
    x = cL + cL*(2*i)
    for j in range(nH):
        y = cH + cH*(2*j)

        ptd1 = Point(x-cL/2, y-cH/2)
        ptd2 = Point(x+cL/2, y+cH/2)

        isCreux = True
        
        if i % 2 == 1:
            obj = Domain(ptd1, ptd2, meshSize, isCreux=isCreux)
        else:
            obj = Domain(ptd1, ptd2, meshSize, isCreux=isCreux)
            # obj = Circle(Point(x, y), cH, meshSize, isCreux=isCreux)

        inclusions.append(obj)

# inclusions = []

if dim == 2:

    tC = h/3
    ptC1 = Point(h, h/2-tC/2, isOpen=True)
    ptC2 = Point(h, h/2+tC/2, isOpen=True)

    cracks = [Line(ptC1, ptC2, meshSize, isOpen=True), Line(ptC1+[h], ptC2+[h], meshSize, isOpen=True)]
    # cracks = []
    # cracks = [Line(Point(L,h/2, isOpen=True), Point(L-h,h/2), isOpen=True, meshSize=meshSize)]
    # cracks = [Line(ptC1, ptC2, meshSize, isOpen=True), Line(ptC2, ptC2+[h], meshSize, isOpen=True)]

if dim == 3:
    # ptC1 = Point(h, h/2-tC/2, b/2-tC/2, isOpen=False)
    # cracks = [PointsList([ptC1, ptC1+[0,tC], ptC1+[0,tC,tC], ptC1+[0,0,tC]], isCreux=True)]    
    
    ptC1 = Point(h, h, 0, isOpen=False)
    ptC2 = Point(h, h/2, 0, isOpen=False)
    ptC3 = Point(h, h/2, b, isOpen=False)
    ptC4 = Point(h, h, b, isOpen=False)

    cracks = [PointsList([ptC1, ptC2, ptC3, ptC4], isCreux=True)]

    cracks.append(Line(ptC1, ptC4, meshSize, isOpen=True))
    cracks.append(Line(ptC1, ptC2, meshSize, isOpen=True))
    cracks.append(Line(ptC3, ptC4, meshSize, isOpen=True))
    # cracks.append(Line(ptC3, ptC2, meshSize, isOpen=True))

    cracks = []

# cracks = [Line(ptC1, ptC2, meshSize, isOpen=True)]
cracks = []

interfaceGmsh = Interface_Gmsh(False, False)

# Fonction utilisée pour la construction du maillage
def DoMesh(refineGeom=None) -> Mesh:
    if dim == 2:
        return interfaceGmsh.Mesh_2D(points, inclusions, elemType, cracks, False, refineGeom)
    else:
        return interfaceGmsh.Mesh_3D(points, inclusions, [0,0,b], 5, elemType, cracks, refineGeom)

# construit le premier maillage
mesh = DoMesh()

# tt = mesh.Nodes_Point(ptC1)

# Affichage.Plot_Mesh(mesh)
# ax = Affichage.Plot_Model(mesh, alpha=0)
# Affichage.Plot_Elements(mesh, mesh.nodes, 2, ax=ax, showId=True)
# Affichage.Plot_Nodes(mesh, mesh.Nodes_Tag(["P1"]))

if dim==2:
    nodesCrack = []
    for crack in cracks:
        nodesCrack.extend(mesh.Nodes_Line(crack))    
if dim==3:
    nodesCrack = mesh.Nodes_Conditions(lambda x,y,z: x==h)

if len(nodesCrack) > 0:
    Affichage.Plot_Nodes(mesh, nodesCrack, showId=True)

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
    # simu.add_surfLoad(noeuds_en_L, [surfLoad], ["x"])

    simu.Solve()

    simu.Save_Iteration()

    # ----------------------------------------------
    # Calcul de l'erreur
    # ----------------------------------------------

    erreur, erreur_e = simu._Calc_ZZ1()

    # ----------------------------------------------
    # Refine mesh
    # ----------------------------------------------

    meshSize_n = Calc_meshSize_n(simu.mesh, erreur_e, coef)

    if plotErreur:
        Affichage.Plot_Result(simu, erreur_e*100, nodeValues=True, title="erreur %", plotMesh=True)

    path = interfaceGmsh.Create_posFile(simu.mesh.coordo, meshSize_n, folder, f"simu{i}")

    return path, erreur

path = None

erreur = 1
i = -1
while erreur >= cible and i < iterMax:

    i += 1

    if i > 0:
        oldMesh = simu.mesh
        oldU = simu.displacement

    mesh = DoMesh(path)    

    simu.mesh = mesh

    if i > 0:
        os.remove(path)

        if plotProj:
            
            # Affichage.Plot_Result(simu, "ux", plotMesh=True)
            
            proj = Calc_projector(oldMesh, mesh)

            ddlsNew = Simulations.BoundaryCondition.Get_ddls_noeuds(dim, "displacement", mesh.nodes, ["x"])
            ddlsOld = Simulations.BoundaryCondition.Get_ddls_noeuds(dim, "displacement", oldMesh.nodes, ["x"])
            uproj = np.zeros(mesh.Nn*dim)        
            for d in range(dim):
                uproj[ddlsNew+d] = proj @ oldU[ddlsOld+d]

            simu.set_u_n("displacement", uproj)

            # Affichage.Plot_Result(simu, "ux", plotMesh=True, title="ux proj")

            pass


    path, erreur = DoSimu(i)

    print(f"{i} erreur = {erreur*100:.3} %, Wdef = {simu.Get_Resultat('Wdef'):.3f} mJ")

if i > 0:
    os.remove(path)

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
Affichage.NewSection("Post traitement")

# folder=""
if plotResult:
    tic = Tic()
    # simu.Resultats_Resume(True)
    # Affichage.Plot_Result(simu, "amplitude")
    # Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Affichage.Plot_Result(simu, "ux", deformation=True, nodeValues=False)        
    Affichage.Plot_Result(simu, "Svm", deformation=False, plotMesh=True, nodeValues=False)
    # Affichage.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)   

    tic.Tac("Affichage","Affichage des figures", plotResult)

PostTraitement.Make_Paraview(folder, simu, nodesResult=["ZZ1"])

PostTraitement.Make_Movie(folder, "ZZ1", simu, plotMesh=True, fps=1, nodeValues=True)

Tic.Plot_History(details=True)
plt.show()