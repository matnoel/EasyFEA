# %%
from typing import cast

from Geom import Domain, Point
from Mesh import Mesh
from GroupElem import GroupElem
from Materials import Elas_Isot, Materiau
from Interface_Gmsh import Interface_Gmsh
from Simu import Simu
import Affichage as Affichage
from TicTac import Tic
import Dossier as Dossier

import numpy as np
import matplotlib.pyplot as plt


Affichage.Clear()


folder = Dossier.NewFile("Convergence 2D", results=True)

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13
P = 800 #N


# Paramètres maillage
E=210000
v=0.25
comportement = Elas_Isot(dim, epaisseur=b, E=E, v=v, contraintesPlanes=True)

WdefRef = 2*P**2*L/E/h**2 * (L**2/h**2 + (1+v)*3/5)

# Materiau
materiau = Materiau(comportement)

print()

# Pour chaque type d'element et plusieurs taille d'element on va calculer l'energie de deformation pour verifier la convergence


# Listes pour les graphes
listTemps_e_nb = []
listWdef_e_nb = []
listDdl_e_nb = []

# Listes pour les boucles
listNbElement = np.arange(1,15,3)
# listNbElement = np.arange(1,20,1)
# listNbElement = list(range(2,20,1))
# listNbElement = list(range(1,10))

tic = Tic()

# Pour chaque type d'element
for t, elemType in enumerate(GroupElem.get_Types2D()):
# for t, elemType in enumerate(["TRI3"]):        
        
    listTemps_nb = []
    listWdef_nb = []
    listDdl_nb = []

    # Et chaque taille de maille
    for nbElem in listNbElement:
        
        taille = b/nbElem

        domain = Domain(Point(), Point(x=L, y=h), taille=taille)

        # Construction du modele et du maillage --------------------------------------------------------------------------------
        interfaceGmsh = Interface_Gmsh(verbosity=False)
        mesh = interfaceGmsh.Rectangle_2D(domain, elemType=elemType, isOrganised=False)

        mesh = cast(Mesh, mesh)
        # Récupère les noeuds qui m'interessent
        
        noeuds_en_0 = mesh.Nodes_Conditions(conditionX=lambda x: x == 0)
        noeuds_en_L = mesh.Nodes_Conditions(conditionX=lambda x: x == L)

        # Construit la simulation
        simu = Simu(mesh, materiau, verbosity=False, useNumba=False)

        # Renseigne les condtions limites en deplacement
        simu.add_dirichlet("displacement", noeuds_en_0, [0,0], ["x","y"])
        # Renseigne les condtions limites en forces
        simu.add_surfLoad("displacement", noeuds_en_L, [-P/h**2], ["y"])

        # Assemblage du système matricielle
        simu.Assemblage_u()

        simu.Solve_u()
        Wdef = simu.Get_Resultat("Wdef")

        # Stockage des valeurs
        listTemps_nb.append(tic.Tac("Résolutions","Temps total", False))
        listWdef_nb.append(Wdef)
        listDdl_nb.append(mesh.Nn*dim)

        print(f"Elem : {elemType}, nby : {nbElem:2}, Wdef = {np.round(Wdef, 3)}, erreur = {np.abs(WdefRef-Wdef)/WdefRef:3e} ")
    
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

for t, elemType in enumerate(GroupElem.get_Types2D()):
# for t, elemType in enumerate(["TRI3"]):

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
ax_Wdef.set_ylabel('Wdef [N.mm]')
ax_Wdef.legend(GroupElem.get_Types2D())
ax_Wdef.fill_between(listDdl_nb, WdefRefArray, WdefRefArray5, alpha=0.5, color='red')

# Erreur
ax_Temps_Erreur.grid()
ax_Temps_Erreur.set_xlabel('ddl')
ax_Temps_Erreur.set_ylabel('Erreur [%]')
ax_Temps_Erreur.legend(GroupElem.get_Types2D())


# Temps
ax_Temps.grid()
ax_Temps.set_xlabel('ddl')
ax_Temps.set_ylabel('Temps [s]')
ax_Temps.legend(GroupElem.get_Types2D())


Tic.getResume()

Tic.getGraphs(folder)

plt.show()

# %%
