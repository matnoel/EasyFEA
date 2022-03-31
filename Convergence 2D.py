# %%

from Element import Element
from Materiau import Elas_Isot, Materiau
from ModelGmsh import ModelGmsh
from Mesh import Mesh
from Simu import Simu
from Affichage import Affichage
from TicTac import TicTac

import numpy as np
import matplotlib.pyplot as plt


Affichage.Clear()

# Data --------------------------------------------------------------------------------------------

plotResult = True

dim = 2

# Paramètres géométrie
L = 120;  #mm
h = 13    
b = 13

P = 800 #N

# Paramètres maillage
comportement = Elas_Isot(dim, epaisseur=b)

# Materiau
materiau = Materiau(comportement)

# Pour chaque type d'element et plusieurs taille d'element on va calculer l'energie de deformation pour verifier la convergence


# Listes pour les graphes
listTemps_e_nb = []
listWdef_e_nb = []
listDdl_e_nb = []

# Listes pour les boucles
listNbElement = list(range(1,30,5))
# listNbElement = list(range(1,10))

tic = TicTac()

# Pour chaque type d'element
for elem, type in enumerate(Element.get_Types2D()):
        
        listTemps_nb = []
        listWdef_nb = []
        listDdl_nb = []

        # Et chaque taille de maille
        for nbElem in listNbElement:
                
                taille = b/nbElem

                # Construction du modele et du maillage --------------------------------------------------------------------------------
                modelGmsh = ModelGmsh(dim, organisationMaillage=True, typeElement=elem, tailleElement=taille, verbosity=False)
                (coordo, connect) = modelGmsh.ConstructionRectangle(L, h)
                mesh = Mesh(dim, coordo, connect, verbosity=False)

                # Récupère les noeuds qui m'interessent
                noeuds_en_0 = mesh.Get_Nodes(conditionX=lambda x: x == 0)
                noeuds_en_L = mesh.Get_Nodes(conditionX=lambda x: x == L)

                # Construit la simulation
                simu = Simu(dim, mesh, materiau, verbosity=False)

                # Renseigne les condtions limites en deplacement
                simu.Condition_Dirichlet(noeuds_en_0, valeur=0, directions=["x","y"])
                # Renseigne les condtions limites en forces
                simu.Condition_Neumann(noeuds_en_L, valeur=-P, directions=["y"])

                # Assemblage du système matricielle
                simu.Assemblage_u()

                simu.Solve_u()

                # Stockage des valeurs
                listTemps_nb.append(tic.Tac("Résolution","Temps total", False))
                listWdef_nb.append(simu.GetResultat("Wdef"))
                listDdl_nb.append(mesh.Nn*dim)
        
        listTemps_e_nb.append(listTemps_nb)
        listWdef_e_nb.append(listWdef_nb)
        listDdl_e_nb.append(listDdl_nb)

# Post traitement --------------------------------------------------------------------------------------

# Affiche la convergence d'energie

fig_Wdef, ax_Wdef = plt.subplots()
fig_Erreur, ax_Temps_Erreur = plt.subplots()
fig_Temps, ax_Temps = plt.subplots()

# WdefRef = np.max(listWdef_e_nb)
WdefRef = 371.5
# WdefRef = 391.76

for elem, type in enumerate(Element.get_Types2D()):

        # Convergence Energie
        ax_Wdef.plot(listDdl_e_nb[elem], listWdef_e_nb[elem])
        
        # Erreur
        Wdef = np.array(listWdef_e_nb[elem])
        erreur = (WdefRef-Wdef)/WdefRef*100
        ax_Temps_Erreur.loglog(listDdl_e_nb[elem],erreur)

        # Temps
        # ax_Temps.plot(listDdl_e_nb[elem], listTemps_e_nb[elem])
        ax_Temps.loglog(listDdl_e_nb[elem], listTemps_e_nb[elem])

# Wdef
ax_Wdef.grid()
ax_Wdef.set_xlim([-10,12000])
ax_Wdef.set_xlabel('ddl')
ax_Wdef.set_ylabel('Wdef [N.mm]')
ax_Wdef.legend(Element.get_Types2D())

# Erreur
ax_Temps_Erreur.grid()
ax_Temps_Erreur.set_xlabel('ddl')
ax_Temps_Erreur.set_ylabel('Erreur [%]')
ax_Temps_Erreur.legend(Element.get_Types2D())


# Temps
ax_Temps.grid()
ax_Temps.set_xlabel('ddl')
ax_Temps.set_ylabel('Temps [s]')
ax_Temps.legend(Element.get_Types2D())


TicTac.getResume()

plt.show()



# print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

# # print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm_n"]))) 

# print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx_n"]))) 
# print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx_n"]))) 

# print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy_n"]))) 
# print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy_n"])))

# print(np.max(simu.resultats["Sxx_e"]))

# if plotResult:
        
#         # Affichage.AfficheNoeudsMaillage(simu, showId=False)
#         # Affichage.PlotMesh(mesh, simu.resultats, deformation=False)
#         fig, ax = Affichage.PlotMesh(simu, deformation=True)
#         # Affichage.AfficheNoeudsMaillage(simu, showId=True)
#         Affichage.PlotResult(simu, "amplitude", deformation=True)
#         # Affichage.PlotResult(mesh, simu.resultats, "dx_n", affichageMaillage=False)
#         # Affichage.PlotResult(mesh, simu.resultats, "dx_e", affichageMaillage=True)        
#         # Affichage.PlotResult(mesh, simu.resultats, "Svm_e")
#         Affichage.PlotResult(simu, "Svm_e")
#         # Affichage.PlotResult(simu, "Svm_n")
#         # Affichage.PlotResult(simu, "Svm_n", affichageMaillage=True, deformation=True)

#         # Affichage.PlotResult(mesh, simu.resultats, "dy_n")
#         # Affichage.PlotResult(mesh, simu.resultats, "dy_e", deformation=True, affichageMaillage=True)
        
#         plt.show()

# %%
