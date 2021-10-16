import os
import sys
from typing import cast
import gmsh
from class_Simu import Simu
from class_Noeud import Noeud
import numpy as np
import matplotlib.pyplot as plt

os.system("cls")    #nettoie terminal

# Data --------------------------------------------------------------------------------------------

affichageGMSH = False

plotResult = True

type = "T3"
maillageOrganisé = True

# Paramètres géométrie
L = 1;  #mm

P = -800 #N

# Paramètres maillage
taille = L/2

# Construction GMSH --------------------------------------------------------------------------------

print("==========================================================")
print("Gmsh : \n")

print("Elements {} \n".format(type))

gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 0)
gmsh.model.add("model")

# Créer les points
p1 = gmsh.model.geo.addPoint(0, 0, 0, taille)
p2 = gmsh.model.geo.addPoint(L, 0, 0, taille)
p3 = gmsh.model.geo.addPoint(L, L, 0, taille)
p4 = gmsh.model.geo.addPoint(0, L, 0, taille)

# Créer les lignes reliants les points
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Créer une boucle fermée reliant les lignes     
cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

# Créer une surface
pl = gmsh.model.geo.addPlaneSurface([cl])

# Impose que le maillage soit organisé
if maillageOrganisé:
        gmsh.model.geo.mesh.setTransfiniteSurface(pl)

# Synchronise le modele avec gmsh pour le mailler apres
gmsh.model.geo.synchronize()

if type in ["Q4","Q8"]:
        gmsh.model.mesh.setRecombine(2, pl)

gmsh.model.mesh.generate(2) 

if type in ["Q8"]:
        gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

if type in ["T3","Q4"]:
        gmsh.model.mesh.set_order(1)
elif type in ["T6","Q8"]:
        gmsh.model.mesh.set_order(2)

# Ouvre l'interface de gmsh si necessaire
if '-nopopup' not in sys.argv and affichageGMSH:
    gmsh.fltk.run()

# ------------------------------------------------------------------------------------------------------

print("\n==========================================================")
print("Traitement :")

simu = Simu(2, verbosity=True)

simu.CreationMateriau()

simu.ConstructionMaillageGMSH(gmsh.model.mesh)
gmsh.finalize()

simu.Assemblage(epaisseur=1)



# Récupère la coordonnées des noeuds que je veux
noeuds_B = []
noeuds_H = []
noeuds_F = []
for n in simu.mesh.noeuds:
        n = cast(Noeud, n)                
        if n.y == 0:
                noeuds_B.append(n.id)
        if n.y == L:
                noeuds_H.append(n.id)        
        if np.isclose(n.y, L/2) and (np.isclose(n.x,L/2) or n.x < L/2):
                noeuds_F.append(n.id)



# Calcul H

Nn = simu.mesh.get_Nn()

d = np.zeros((Nn, 1))

u = np.zeros((Nn*2, 1))


# Applique d = 1 sur les noeuds
d[noeuds_F,0] = 1



pass

# simu.PlotMesh()
# plt.show()

# simu.Solve()

# # Post traitement --------------------------------------------------------------------------------------
# print("\n==========================================================")
# print("Post traitement :")

# print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

# print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm"]))) 

# print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx"]))) 
# print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx"]))) 

# print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy"]))) 
# print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy"]))) 

# if plotResult:

#         simu.PlotMesh()
#         simu.PlotResult(resultat="dx",affichageMaillage=True, deformation=True)
#         # simu.PlotResult(resultat="dy")

#         # simu.PlotResult(resultat="Sxx")
#         # simu.PlotResult(resultat="Syy")
#         # simu.PlotResult(resultat="Sxy")
#         simu.PlotResult(resultat="Svm", affichageMaillage=True, deformation=True)
        
#         plt.show()

# print("\n==========================================================")
# print("\n FIN DU PROGRAMME \n")
