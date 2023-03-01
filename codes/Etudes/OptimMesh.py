import os
import matplotlib.pyplot as plt

import Folder
import PostTraitement
import Affichage
from Geom import *
import Materials
from Interface_Gmsh import Interface_Gmsh
import Simulations
from TicTac import Tic

# Affichage.Clear()

# TODO Adaptation de maillage 
# gmsh.view.addListData()
# gmsh.view.s

# import gmsh
# gmsh.model.mesh.refine()

# ----------------------------------------------
# Configuration
# ----------------------------------------------

dim = 2
folder = Folder.New_File(f"OptimMesh{dim}D", results=True)
plotResult = True

# Paramètres géométrie
L = 120;  #mm
h = 13
b = 13

P = 800 #N
lineLoad = P/h #N/mm
surfLoad = P/h/b #N/mm2

# Paramètres maillage
# taille = h/1
# taille = L/2
taille = h/10

# ----------------------------------------------
# Maillage
# ----------------------------------------------

interfaceGmsh = Interface_Gmsh(False)
if dim == 2:
    domain = Domain(Point(y=-h/2), Point(x=L, y=h/2), taille)
    Line0 = Line(Point(y=-h/2), Point(y=h/2))
    LineL = Line(Point(x=L,y=-h/2), Point(x=L, y=h/2))
    LineH = Line(Point(y=h/2),Point(x=L, y=h/2))
    circle = Circle(Point(x=L/2, y=0), h*0.2, isCreux=False)
    
    elemType = "QUAD4" # ["TRI3", "TRI6", "TRI10", "TRI15", "QUAD4", "QUAD8"]

    mesh = interfaceGmsh.Mesh_Rectangle_2D(domain=domain, elemType=elemType, isOrganised=True)
    # mesh = interfaceGmsh.PlaqueAvecCercle(domain=domain, circle=circle, isOrganised=False)
    aire = mesh.aire - L*h
elif dim == 3:
    # # Sans importation
    domain = Domain(Point(y=-h/2,z=-b/2), Point(x=L, y=h/2,z=-b/2), taille=taille)
    # circle = Circle(Point(x=L/2, y=0), h*0.8, taille=taille, isCreux=False)
    # mesh = interfaceGmsh.PlaqueAvecCercle3D(domain,circle ,[0,0,b], elemType="HEXA8", isOrganised=False, nCouches=3)
    
    elemType = "HEXA8" # "TETRA4", "TETRA10", "HEXA8", "PRISM6"
    mesh = interfaceGmsh.Mesh_Poutre3D(domain, [0,0,b], elemType=elemType, isOrganised=False, nCouches=3)

    volume = mesh.volume - L*b*h
    aire = mesh.aire - (L*h*4 + 2*b*h)

Affichage.Plot_Mesh(mesh)
# Affichage.Plot_NoeudsMaillage(mesh,showId=True)
# plt.show()

noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0) # noeuds_en_0 = mesh.Nodes_Line(Line0)
noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == L) # noeuds_en_L = mesh.Nodes_Line(LineL)

# Affichage.Plot_Maillage(mesh)
# plt.show()

noeuds_en_h = mesh.Nodes_Conditions(lambda x,y,z: y == h/2) # noeuds_en_h= mesh.Nodes_Line(LineH)

# ----------------------------------------------
# Comportement et Simu
# ----------------------------------------------

comportement = Materials.Elas_Isot(dim, E=210000, v=0.3, epaisseur=b)
simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False)
simu.rho = 8100*1e-9

simu.Bc_Init()

# Renseigne les condtions limites

if dim == 2:
    simu.add_dirichlet(noeuds_en_0, [0, 0], ["x","y"], description="Encastrement")
elif dim == 3:
    simu.add_dirichlet(noeuds_en_0, [0, 0, 0], ["x","y","z"], description="Encastrement")

simu.add_surfLoad(noeuds_en_L, [-surfLoad], ["y"])

Affichage.Plot_BoundaryConditions(simu)
# plt.show()

simu.Solve()

# ----------------------------------------------
# Calcul de l'erreur
# ----------------------------------------------

matriceType = "rigi"
B_e_pg = simu.mesh.Get_B_dep_e_pg(matriceType)
jacobien_e_pg = simu.mesh.Get_jacobien_e_pg(matriceType)
poid_pg = simu.mesh.Get_poid_pg(matriceType)
N_pg = simu.mesh.Get_N_scalaire_pg(matriceType)

u_e = simu.mesh.Localises_sol_e(simu.displacement)

Epsilon_e_pg = np.einsum("epij,ej->epi", B_e_pg, u_e)
Sigma_e_pg = np.einsum("ij,epj->epi", comportement.C, Epsilon_e_pg)

Wdef_e = 1/2 * b * np.einsum("ep,p,epi,epi->e", jacobien_e_pg, poid_pg, Sigma_e_pg, Epsilon_e_pg)
Wdef = np.sum(Wdef_e)

Sigma_n = simu.Resultats_InterpolationAuxNoeuds(simu.mesh, np.mean(Sigma_e_pg, 1))

Sigma_n_e = simu.mesh.Localises_sol_e(Sigma_n)
sigmaliss_e_pg = np.einsum('eni,pjn->epi',Sigma_n_e, N_pg)

WdefLisse_e = 1/2 * b * np.einsum("ep,p,epi,epi->e", jacobien_e_pg, poid_pg, sigmaliss_e_pg, Epsilon_e_pg)
WdefLisse = np.sum(WdefLisse_e)

erreur_e = np.abs(WdefLisse_e-Wdef_e).reshape(-1)/Wdef    

Affichage.Plot_Result(simu, erreur_e*100, nodeValues=True, title="erreur %", plotMesh=True)

# erreur_e = np.abs(WdefLisse_e-Wdef_e).reshape(-1)
# Affichage.Plot_Result(simu, erreur_e, nodeValues=False, title="erreur_e mJ")

erreur = np.abs(Wdef-WdefLisse)/Wdef

print(f"erreur = {erreur*100:.3} %")

# ----------------------------------------------
# Post traitement
# ----------------------------------------------
Affichage.NouvelleSection("Post traitement")

# folder=""
if plotResult:
    tic = Tic()
    simu.Resultats_Resume(True)
    # Affichage.Plot_Result(simu, "amplitude")
    # Affichage.Plot_Maillage(simu, deformation=True, folder=folder)
    Affichage.Plot_Result(simu, "uy", deformation=True, nodeValues=False)        
    Affichage.Plot_Result(simu, "Svm", deformation=False, plotMesh=False, nodeValues=False)
    # Affichage.Plot_Result(simu, "Svm", deformation=True, nodeValues=False, plotMesh=False, folder=folder)
    
    tic.Tac("Affichage","Affichage des figures", plotResult)

Tic.Plot_History(details=True)
plt.show()