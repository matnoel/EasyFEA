import os
import sys
import gmsh
from class_Simu import Simu
import numpy as np
import matplotlib.pyplot as plt

os.system("cls")    #nettoie tèerminal

# Data --------------------------------------------------------------------------------------------

affichageGMSH = False

plotResult = True

type = "T6"
maillageOrganisé = True

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = -800 #N

# Paramètres maillage
taille = h/4


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
p3 = gmsh.model.geo.addPoint(L, h, 0, taille)
p4 = gmsh.model.geo.addPoint(0, h, 0, taille)

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



if '-nopopup' not in sys.argv and affichageGMSH:
    gmsh.fltk.run()

# ------------------------------------------------------------------------------------------------------

print("\n==========================================================")
print("Traitement :")

simu = Simu(2, verbosity=True)

simu.CreationMateriau()

simu.ConstructionMaillageGMSH(gmsh.model.mesh)
gmsh.finalize()

simu.Assemblage(epaisseur=b)

noeuds_en_L = []
noeuds_en_0 = []
for n in simu.mesh.noeuds:        
        if n.x == L:
                noeuds_en_L.append(n)
        if n.x == 0:
                noeuds_en_0.append(n)

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, direction="Y")

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="X")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Y")

simu.Solve()

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Résultats :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"])) 

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm"]))) 

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx"]))) 
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx"]))) 

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy"]))) 
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy"]))) 



if plotResult:

        simu.PlotMesh()
        simu.PlotResult(resultat="dx",affichageMaillage=True)
        # simu.PlotResult(resultat="dy")

        # simu.PlotResult(resultat="Sxx")
        # simu.PlotResult(resultat="Syy")
        # simu.PlotResult(resultat="Sxy")
        # simu.PlotResult(resultat="Svm", affichageMaillage=True, deformation=True, facteurDef=2)
        
        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



