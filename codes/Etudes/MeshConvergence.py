from typing import cast

from Geom import Domain, Point
from Mesh import Mesh
from GroupElem import GroupElem
import Materials
from Interface_Gmsh import Interface_Gmsh
import Simulations
import Display as Display
from TicTac import Tic
import Folder
import PostProcessing

import numpy as np
import matplotlib.pyplot as plt

Display.Clear()

dim = 3

folder = Folder.New_File(f"Convergence {dim}D", results=True)

# Data --------------------------------------------------------------------------------------------

plotResult = True

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13
P = 800 #N


# Paramètres maillage
E=210000
v=0.25
comportement = Materials.Elas_Isot(dim, thickness=b, E=E, v=v, planeStress=True)

WdefRef = 2*P**2*L/E/h/b * (L**2/h/b + (1+v)*3/5)

print()

# Pour chaque type d'element et plusieurs taille d'element on va calculer l'energie de deformation pour verifier la convergence


# Listes pour les graphes
listTemps_e_nb = []
listWdef_e_nb = []
listDdl_e_nb = []

# Listes pour les boucles
if dim == 2:
    listNbElement = np.arange(0.5,10,1)
    # listNbElement = np.arange(1,20,1)
    # listNbElement = list(range(2,20,1))
    # listNbElement = list(range(1,10))
else:
    listNbElement = np.arange(1,4,2)


tic = Tic()

simu = None

# Pour chaque type d'element

elemTypes = GroupElem.get_Types2D() if dim == 2 else GroupElem.get_Types3D()

# elemTypes = ["QUAD4"]

# elemTypes = ["PRISM6","PRISM15"]
# elemTypes = ["PRISM6","PRISM15"]
# elemTypes = ["HEXA20"]
# elemTypes = ["TETRA10"]

interfaceGmsh = Interface_Gmsh(False, False)

for t, elemType in enumerate(elemTypes):
        
    listTemps_nb = []
    listWdef_nb = []
    listDdl_nb = []

    # Et chaque taille de maille
    for nbElem in listNbElement:
        
        taille = b/nbElem

        domain = Domain(Point(), Point(x=L, y=h), meshSize=taille)

        # Construction du modele et du maillage --------------------------------------------------------------------------------
        if dim == 2:            
            mesh = interfaceGmsh.Mesh_2D(domain, [], elemType, isOrganised=True)
        else:            
            mesh = interfaceGmsh.Mesh_3D(domain, [], elemType=elemType, extrude=[0,0,b], nCouches=4)

        mesh = cast(Mesh, mesh)
        # Récupère les noeuds qui m'interessent

        # Display.Plot_Mesh(mesh)

        if mesh.dim == 3:
            volume  = mesh.volume
        else:
            volume  = mesh.area * comportement.thickness

        assert np.abs(volume - (L*h*b))/volume <= 1e-10
        
        noeuds_en_0 = mesh.Nodes_Conditions(lambda x,y,z: x == 0)
        noeuds_en_L = mesh.Nodes_Conditions(lambda x,y,z: x == L)

        # Construit la simulation
        if simu == None:
            simu = Simulations.Simu_Displacement(mesh, comportement, verbosity=False, useNumba=False)
        else:
            simu.Bc_Init()
            simu.mesh = mesh
        
        # Renseigne les condtions limites en deplacement
        if dim == 2:
            simu.add_dirichlet(noeuds_en_0, [0,0], ["x","y"])
        else:
            simu.add_dirichlet(noeuds_en_0, [0,0,0], ["x","y","z"])
        # Renseigne les condtions limites en forces
        simu.add_surfLoad(noeuds_en_L, [-P/h/b], ["y"])

        # Assemblage du système matricielle

        simu.Solve()

        simu.Save_Iter()

        Wdef = simu.Get_Result("Wdef")

        # Stockage des valeurs
        listTemps_nb.append(tic.Tac("Résolutions","Temps total", False))
        listWdef_nb.append(Wdef)
        listDdl_nb.append(mesh.Nn*dim)

        if elemType != mesh.elemType:
            print("erreur lors de la création du maillage")

        print(f"Elem : {mesh.elemType}, nby : {nbElem:2}, Wdef = {np.round(Wdef, 3)}, erreur = {np.abs(WdefRef-Wdef)/WdefRef:.2e} ")
    
    listTemps_e_nb.append(listTemps_nb)
    listWdef_e_nb.append(listWdef_nb)
    listDdl_e_nb.append(listDdl_nb)

# Post traitement --------------------------------------------------------------------------------------

# Affiche la convergence d'energie

fig_Wdef, ax_Wdef = plt.subplots()
fig_Erreur, ax_Temps_Erreur = plt.subplots()
fig_Temps, ax_Temps = plt.subplots()

WdefRefArray = np.ones_like(listDdl_e_nb[0]) * WdefRef
WdefRefArray5 = WdefRefArray * 0.95

print(f"\nWSA = {np.round(WdefRef, 4)} mJ")

# WdefRef = 391.76

for t, elemType in enumerate(elemTypes):

    # Convergence Energie
    ax_Wdef.plot(listDdl_e_nb[t], listWdef_e_nb[t])
    
    # Erreur
    Wdef = np.array(listWdef_e_nb[t])
    erreur = (WdefRef-Wdef)/WdefRef*100
    ax_Temps_Erreur.loglog(listDdl_e_nb[t],erreur)

    # Temps
    # ax_Temps.plot(listDdl_e_nb[elem], listTemps_e_nb[elem])
    ax_Temps.loglog(listDdl_e_nb[t], listTemps_e_nb[t])

# Wdef
ax_Wdef.grid()
ax_Wdef.set_xlim([-10,8000])
ax_Wdef.set_xlabel('ddl')
ax_Wdef.set_ylabel('Wdef [mJ]')
ax_Wdef.legend(elemTypes)
ax_Wdef.fill_between(listDdl_nb, WdefRefArray, WdefRefArray5, alpha=0.5, color='red')

# Erreur
ax_Temps_Erreur.grid()
ax_Temps_Erreur.set_xlabel('ddl')
ax_Temps_Erreur.set_ylabel('Erreur [%]')
ax_Temps_Erreur.legend(elemTypes)


# Temps
ax_Temps.grid()
ax_Temps.set_xlabel('ddl')
ax_Temps.set_ylabel('Temps [s]')
ax_Temps.legend(elemTypes)

Display.Plot_Result(simu, "Svm", nColors=20)

PostProcessing.Make_Paraview(folder, simu, details=True)

Tic.Resume()

# Tic.Plot_History(folder)

plt.show()