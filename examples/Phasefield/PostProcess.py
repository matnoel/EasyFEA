# Copyright (C) 2021-2024 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Script to compare damage simulations."""

from EasyFEA import Display, Folder, Tic, plt, np, pd, Simulations

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    # "PlateWithHole_Benchmark", "PlateWithHole_FCBA", "Shear_Benchmark", "Tension_Benchmark" "L_Shape_Benchmark"
    # simulation = "Shear_Benchmark"
    simulation = "Shear_Benchmark"

    meshTest = False
    loadSimu = True
    plotDamage = True
    savefig = True

    folder_results = Folder.New_File(simulation, results=True)

    if savefig:
        if meshTest:
            folder_save = Folder.Join(folder_results, "Test", "_Post processing")
        else:
            folder_save = Folder.Join(folder_results, "_Post processing")
    else:
        folder_save=""

    list_mat = ["Elas_Isot"] # ["Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"]

    list_regu = ["AT2", "AT1"] # ["AT1", "AT2"]
    # list_regu = ["AT2"] # ["AT1", "AT2"]

    list_simpli2D = ["DP"] # ["CP","DP"]
    list_solver = ["History"]

    # list_split = ["Bourdin","Amor","Miehe","He","Zhang"]
    # list_split = ["Bourdin","Amor","Miehe","He","AnisotStrain","AnisotStress","Zhang"]
    # list_split = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]
    list_split = ["AnisotStrain","AnisotStress", "He", "Zhang"]
    # list_split = ["AnisotStress", "He"]
    # list_split = ["Bourdin","Amor","Miehe"]
    # list_split = ["AnisotStress"]

    # listOptimMesh=[False, True] # [True, False]
    listOptimMesh=[False] # [True, False]

    # listTol = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] # [1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
    # listTol = [1e-0, 1e-2]
    listTol = [1e-0]

    # listnL = [100] # [100] [100, 120, 140, 180, 200]
    listnL = [0]

    listTheta = [0]
    # listTheta = [-0, -10, -20, -30, -45, -60, -70, -80, -90]

    # snapshots = [18.5, 24.6, 25, 28, 35]
    # snapshots = [9.05, 10.5, 14.5]
    # snapshots = [18.6,24.6,24.8]
    # snapshots = [24.6, 25]
    snapshots = []

    depMax = 20e-5
    # depMax = 2.5e-5
    # depMax = 3.5e-5

    # Génération des configurations
    listConfig = []

    for theta in listTheta:
        for comp in list_mat:
            for split in list_split:
                for regu in list_regu:
                    for simpli2D in list_simpli2D:
                        for solveur in list_solver:                                    
                            for tol in listTol:
                                for optimMesh in listOptimMesh:
                                    for nL in listnL:
                                        listConfig.append([comp, regu, simpli2D, solveur, split, tol, optimMesh, nL, theta])

    Nconfig = len(listConfig)

    # ----------------------------------------------
    # Loads all simulations
    # ----------------------------------------------
    ax_load = Display.Init_Axes() # superposition axis of force-displacement curves

    missingSimulations = []

    for config in listConfig:

        comp = config[0]
        regu = config[1]
        simpli2D = config[2]
        solveur = config[3]
        split = config[4]
        tolConv = config[5]
        optimMesh = config[6]
        nL = config[7]
        theta = config[8]

        tic = Tic()

        foldername = Folder.PhaseField_Folder(folder_results, material=comp,  split=split, regu=regu, simpli2D=simpli2D, tolConv=tolConv, solver=solveur, test=meshTest, optimMesh=optimMesh, closeCrack=False, nL=nL, theta=theta)

        fileForceDep = Folder.Join(foldername, "force-displacement.pickle")
        fileSimu = Folder.Join(foldername, "simulation.pickle")

        nomSimu = foldername.split(comp+'_')[-1]
        
        # text = nomSimu
        # text = split
        # text = f"{split}_{regu}_tol{tolConv:1.0e}"
        # text = f"{tolConv:1.0e}"
        # text = f"{split}_{regu}"        
        # text = foldername.replace(Folder.Get_Path(foldername), "")[1:]

        text = f"{split} {regu}"
        # if optimMesh:
        #     text += f" optim"

        if (loadSimu or plotDamage) and Folder.Exists(fileSimu):
            # Load simulation
            simu = Simulations.Load_Simu(foldername)
            simu.mesh.groupElem.coord
            results = pd.DataFrame(simu.results)
            temps = results["timeIter"].values.sum()
            temps_str, unite = Tic.Get_time_unity(temps)
            print(len(results),f"-> {temps_str:.3} {unite}")
            
        else:            
            if nomSimu not in missingSimulations: missingSimulations.append(nomSimu)
            Display.MyPrintError("Simu is not available:\n"+fileForceDep)
        

        # Loads force and displacement
        if Folder.Exists(fileForceDep):
            force, displacement = Simulations.Load_Force_Displacement(foldername)
            
            if depMax == 0:
                depMax = displacement[-1]
                
            indexLim = np.where(displacement <= depMax)[0]            
            
            # text += f" ({temps_str:.3} {unite})"

            ls = None
            c = None

            # ls = '--' if optimMesh else None
            # c = ax_load.get_lines()[-1].get_color() if optimMesh else None
            
            ls = '--' if regu == "AT1" else None
            c = ax_load.get_lines()[-1].get_color() if regu == "AT1" else None

            ax_load.plot(displacement[indexLim]*1e3, np.abs(force[indexLim])*1e-6, c=c, label=text, ls=ls)


        else:
            if nomSimu not in missingSimulations: missingSimulations.append(nomSimu)
            Display.MyPrintError("Data are not available:\n"+fileForceDep)
        

        if plotDamage and Folder.Exists(fileSimu):

            # Display.Plot_Mesh(simu.mesh)

            if simulation == "PlateWithHole_Benchmark":
                colorBarIsClose = True
            else:
                colorBarIsClose = False

            # Displays last damage
            filename = foldername.replace(Folder.Get_Path(foldername), "")[1:]

            ax = Display.Plot_Result(simu, "damage", ncolors=21, colorbarIsClose=False)
            ax.axis('off'); ax.set_title("")            
            Display.Save_fig(folder_save, f"{filename} last")

            # Display.Plot_Result(simu, "damage", ncolors=21, nodeValues=True, colorbarIsClose=colorBarIsClose,
            # folder=folder_save, filename=f"{split} tol{tolConv} last", plotMesh=False,
            # title=f"{split}_{regu}_tol{tolConv}")

            # Recover snapshot iterations

            if len(snapshots) > 0:
                # ax_load_2 = Display.Init_Axes()                    
                # ax_load_2.plot(displacement[indexLim]*1e3, np.abs(force[indexLim])*1e-6, label=text)
                # ax_load_2.set_xlabel("Déplacement [mm]")
                # ax_load_2.set_ylabel("Force [kN/mm]")

                for d, dep in enumerate(snapshots):                
                    
                    try:
                        i = np.where(displacement*1e6>=dep)[0][0]
                    except IndexError:
                        continue

                    simu.Set_Iter(i)
                    filenameDamage = f"{nomSimu}, ud = {np.round(displacement[i]*1e6,2)}"
                    # titleDamage = filenameDamage

                    # titleDamage = split
                    # filenameDamage = f"PlateBench {comp}_{split}_{regu}_{simpli2D}"

                    # Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=colorBarIsClose, folder=folder_save, filename=filenameDamage,title=titleDamage)

                    ax = Display.Plot_Result(simu, "damage", ncolors=21, colorbarIsClose=True)
                    ax.axis('off'); ax.set_title("")
                    # Display._Remove_colorbar(ax)
                    filename = foldername.replace(Folder.Get_Path(foldername), "")[1:]
                    Display.Save_fig(folder_save, f"{filename} snap={dep}", True) # , "png", 200
                    
                    # ax_load_2.scatter(displacement[i]*1e3, force[i]*1e-6, c='k', marker='+', zorder=2)

                # plt.figure(ax_load_2.figure)
                # Display.Save_fig(folder_save, f"load displacement {filename}")
                
        
        
        # text = text.replace("AnisotStrain","Spectral")
        tic.Tac("PostProcessing", split, False)

    ax_load.set_xlabel("Déplacement [mm]")
    ax_load.set_ylabel("Force [kN/mm]")
    # ax_load.set_xlabel("displacement [µm]")
    # ax_load.set_ylabel("load [kN/mm]")
    # ax_load.set_xlabel("displacement")
    # ax_load.set_ylabel("load")
    ax_load.grid()
    ax_load.legend()
    plt.figure(ax_load.figure)
    Display.Save_fig(folder_save, "load displacement")

    print('\n Missing Simulations :')
    [print(simul) for simul in missingSimulations]

    plt.show()