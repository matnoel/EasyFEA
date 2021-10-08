import os
import sys
import gmsh
from class_Simu import Simu
import numpy as np
import matplotlib.pyplot as plt

os.system("cls")    #nettoie terminal

# Data --------------------------------------------------------------------------------------------

affichageGMSH = False

plotResult = True

type = "T4"
maillageOrganisé = False

# Paramètres géométrie
L = 120;  #mm
h = 13;    
b = 13

P = -800 #N

# Paramètres maillage
taille = h/5


# Construction GMSH --------------------------------------------------------------------------------

print("==========================================================")
print("Gmsh : \n")

print("Elements {} \n".format(type))

gmsh.initialize()
gmsh.option.setNumber('General.Verbosity', 0)
gmsh.model.add("model")

gmsh.model.occ.importShapes('part.stp')
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", taille)
gmsh.option.setNumber("Mesh.MeshSizeMax", taille)
gmsh.model.mesh.generate(3)

# if maillageOrganisé:
#         gmsh.model.geo.mesh.setTransfiniteSurface(pl)

# gmsh.model.geo.synchronize()

# if type in ["Q4","Q8"]:
#         gmsh.model.mesh.setRecombine(3, v)

# gmsh.model.mesh.generate(2) 

# if type in ["Q8"]:
#         gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

# if type in ["T3","Q4"]:
#         gmsh.model.mesh.set_order(1)
# elif type in ["T6","Q8"]:
#         gmsh.model.mesh.set_order(2)



if '-nopopup' not in sys.argv and affichageGMSH:
    gmsh.fltk.run()

# ------------------------------------------------------------------------------------------------------
print("==========================================================")
print("Traitement :")

simu = Simu(3, verbosity=True)

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

simu.ConditionEnForce(noeuds=noeuds_en_L, force=P, direction="Z")

simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="X")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Y")
simu.ConditionEnDeplacement(noeuds=noeuds_en_0, deplacement=0, direction="Z")

simu.Solve()

# Post traitement --------------------------------------------------------------------------------------
print("\n==========================================================")
print("Résultats :")

print("\nW def = {:.6f} N.mm".format(simu.resultats["Wdef"]))

print("\nSvm max = {:.6f} MPa".format(np.max(simu.resultats["Svm"])))

print("\nSxx max = {:.6f} MPa".format(np.max(simu.resultats["Sxx"])))
print("Syy max = {:.6f} MPa".format(np.max(simu.resultats["Syy"])))
print("Szz max = {:.6f} MPa".format(np.max(simu.resultats["Szz"])))
print("Sxy max = {:.6f} MPa".format(np.max(simu.resultats["Sxy"])))
print("Syz max = {:.6f} MPa".format(np.max(simu.resultats["Syz"])))
print("Sxz max = {:.6f} MPa".format(np.max(simu.resultats["Sxz"])))

print("\nUx max = {:.6f} mm".format(np.max(simu.resultats["dx"])))
print("Ux min = {:.6f} mm".format(np.min(simu.resultats["dx"])))

print("\nUy max = {:.6f} mm".format(np.max(simu.resultats["dy"])))
print("Uy min = {:.6f} mm".format(np.min(simu.resultats["dy"])))

print("\nUz max = {:.6f} mm".format(np.max(simu.resultats["dz"])))
print("Uz min = {:.6f} mm".format(np.min(simu.resultats["dz"])))


if plotResult:

        simu.PlotMesh(deformation=True)
        simu.PlotResult(resultat="dx", deformation=True)
        # simu.PlotResult(resultat="dy")
        # simu.PlotResult(resultat="dz", affichageMaillage=True)

        # simu.PlotResult(resultat="Sxx")
        # simu.PlotResult(resultat="Syy")
        # simu.PlotResult(resultat="Sxy")
        simu.PlotResult(resultat="Svm", affichageMaillage=True, deformation=True)
        
        plt.show()

print("\n==========================================================")
print("\n FIN DU PROGRAMME \n")



