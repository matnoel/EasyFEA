import pandas as pd

import PostTraitement as PostTraitement
import Affichage as Affichage
import Dossier as Dossier
import numpy as np
import matplotlib.pyplot as plt
import TicTac as TicTac
import PhaseFieldSimulation

# L'objectif du script est de récupérer pour chaque simulation la courbe force déplacement
# Didentifier 3 itérations de déplacement (18.5, 24.6, 30) µm 
# Pour ces 3 itérations tracer endommagement

Affichage.Clear()

test = True
loadSimu = True
plotDamage = False

# "PlateWithHole_Benchmark", "PlateWithHole_CompressionFCBA", "Shear_Benchmark", "Tension_Benchmark"
simulation = "PlateWithHole_Benchmark"

if simulation == "PlateWithHole_Benchmark":
    colorBarIsClose = True
else:
    colorBarIsClose = False

folder = Dossier.NewFile(simulation, results=True)

if test:
    folderSauvegarde = Dossier.Join([folder, "Test", "_Post traitement"])
else:
    folderSauvegarde = Dossier.Join([folder, "_Post traitement"])

# ["Bourdin","Amor","Miehe","He","Stress"]
# ["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]
# ["AnisotStress","AnisotStress_NoCross"]
# ["AnisotMiehe_PM","AnisotMiehe_MP"], ["AnisotMiehe_NoCross","AnisotMiehe"]

# ["Miehe","AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross"]
# ["AnisotMiehe","AnisotMiehe_PM","AnisotMiehe_MP","AnisotMiehe_NoCross","He"]
# ["AnisotMiehe","Miehe"]
# ["AnisotMiehe","He"]
# ["AnisotMiehe", "He", "AnisotStress", "Stress"]

listComp = ["Elas_Isot"] # ["Elas_Isot", "Elas_IsotTrans"]
listRegu = ["AT1"] # ["AT1", "AT2"]
listSimpli2D = ["DP"] # ["CP","DP"]
listSolveur = ["History"]
listSplit = ["AnisotStress"]
listOptimMesh=[True] # [True, False]
listTol = [1e-0, 1e-2] # [1e-0, 1e-1, 1e-2, 1e-3, 1e-4]

# snapshot = [18.5, 24.6, 25, 28, 35]
snapshot = [70]

depMax = 80 # µm 35 ou 80

# Génération des configurations
listConfig = []

for comp in listComp:
    for regu in listRegu:
        for simpli2D in listSimpli2D:
            for solveur in listSolveur:
                for split in listSplit:
                    for tol in listTol:
                        for optimMesh in listOptimMesh:
                            listConfig.append([comp, regu, simpli2D, solveur, split, tol, optimMesh])

Nconfig = len(listConfig)



if simulation == "PlateWithHole_Benchmark":
    v=0.3
else:
    v=0

fig, ax = plt.subplots()

for config in listConfig:

    comp = config[0]
    regu = config[1]
    simpli2D = config[2]
    solveur = config[3]
    split = config[4]
    tolConv = config[5]
    optimMesh = config[6]

    tic = TicTac.Tic()

    foldername = PhaseFieldSimulation.ConstruitDossier(dossierSource=simulation,
    comp=comp,  split=split, regu=regu, simpli2D=simpli2D, tolConv=tolConv,
    solveur=solveur, test=test, optimMesh=optimMesh, openCrack=False, v=v)

    nomSimu = foldername.split(comp+'_')[-1]

    # Charge la force et le déplacement
    load, displacement = PostTraitement.Load_Load_Displacement(foldername, False)

    if loadSimu:
        # Charge la simulations
        simu = PostTraitement.Load_Simu(foldername, False)

        # Affichage.Plot_Maillage(simu.mesh)

    if plotDamage:

        titre = split.replace("AnisotMiehe","Spectral")

        # # Affiche le dernier endommagement
        # Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=colorBarIsClose,
        # folder=folderSauvegarde, filename=f"{split} tol{tolConv} last", 
        # title=f"{titre}")
        

        # Récupère les itérations à 18.5, 24.6, 30 et trace l'endommagement
        for dep in snapshot:
            try:
                i = np.where(np.abs(displacement*1e6-dep)<1e-10)[0][0]
            except:
                # i n'a pas été trouvé on continue les iterations
                continue
            
            simu.Update_iter(i)

            filenameDamage = f"{nomSimu} et ud = {np.round(displacement[i]*1e6,2)}"

            Affichage.Plot_Result(simu, "damage", valeursAuxNoeuds=True, colorbarIsClose=colorBarIsClose,
            folder=folderSauvegarde, filename=filenameDamage, 
            title=filenameDamage)

    # texte = nom.replace(f" pour v={v}", "")
    texte = nomSimu

     
    # texte = texte.replace("AnisotMiehe","Spectral")

    indexLim = np.where(displacement*1e6 <= depMax)[0]
    ax.plot(displacement[indexLim]*1e6, np.abs(load[indexLim]*1e-6), label=texte)

    tic.Tac("Post traitement", split, False)

    if loadSimu:
        try:
            resulats = pd.DataFrame(simu.Get_Results())
            temps = resulats['tempsIter'].sum(axis=0)
            tempsCoef, unite = TicTac.Tic.Get_temps_unite(temps)
            print(f'{np.round(tempsCoef, 2)} {unite}')
        except:
            # Les informations n'ont pas été renseingées
            pass

ax.set_xlabel("displacement [µm]")
ax.set_ylabel("load [kN/mm]")
ax.grid()
ax.legend()
plt.figure(fig)
PostTraitement.Save_fig(folderSauvegarde, "load displacement")


plt.show()
        


    