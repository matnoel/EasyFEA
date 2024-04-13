"""Script to compare damage simulations."""

from EasyFEA import Display, Folder, Tic, plt, np, pd, Simulations

if __name__ == '__main__':

    Display.Clear()

    # ----------------------------------------------
    # Configuration
    # ----------------------------------------------
    # "PlateWithHole_Benchmark", "PlateWithHole_FCBA", "Shear_Benchmark", "Tension_Benchmark" "L_Shape_Benchmark"
    simulation = "PlateWithHole_Benchmark"

    test = False
    loadSimu = True

    plotDamage = True
    savefig = False

    folder = Folder.New_File(simulation, results=True)

    if savefig:
        if test:
            folder_save = Folder.Join(folder, "Test", "_Post processing")
        else:
            folder_save = Folder.Join(folder, "_Post processing")
    else:
        folder_save=""

    list_mat = ["Elas_Isot"] # ["Elas_Isot", "Elas_IsotTrans", "Elas_Anisot"]

    # list_regu = ["AT1", "AT2"] # ["AT1", "AT2"]
    list_regu = ["AT2"] # ["AT1", "AT2"]

    list_simpli2D = ["DP"] # ["CP","DP"]
    list_solver = ["History"]

    # list_split = ["Bourdin","Amor","Miehe","He","Zhang"]
    # list_split = ["Bourdin","Amor","Miehe","He","Stress","AnisotStrain","AnisotStress","Zhang"]
    # list_split = ["Bourdin","He","AnisotStrain","AnisotStress","Zhang"]
    # list_split = ["He","AnisotStrain","AnisotStress", "Zhang"]
    list_split = ["He"]

    # listOptimMesh=[True, False] # [True, False]
    listOptimMesh=[True] # [True, False]

    listTol = [1e-0, 1e-1, 1e-2] # [1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
    # listTol = [1e-0, 1e-2]
    # listTol = [1e-0]

    # listnL = [100] # [100] [100, 120, 140, 180, 200]
    listnL = [0]

    listTheta = [0]
    # listTheta = [-0, -10, -20, -30, -45, -60, -70, -80, -90]

    # snapshot = [18.5, 24.6, 25, 28, 35]
    # snapshot = [24.6]
    snapshots = []

    # depMax = 20e-5
    depMax = 2.5e-5

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
    ax_load = Display.init_Axes() # superposition axis of force-displacement curves

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

        foldername = Folder.PhaseField_Folder(folder, material=comp,  split=split, regu=regu, simpli2D=simpli2D, tolConv=tolConv, solver=solveur, test=test, optimMesh=optimMesh, closeCrack=False, nL=nL, theta=theta)

        fileForceDep = Folder.Join(foldername, "force-displacement.pickle")
        fileSimu = Folder.Join(foldername, "simulation.pickle")

        nomSimu = foldername.split(comp+'_')[-1]
        
        # text = nomSimu
        text = split
        text = foldername.replace(Folder.Get_Path(foldername), "")[1:]    

        # Loads force and displacement
        if Folder.Exists(fileForceDep):
            load, displacement = Simulations.Load_Force_Displacement(foldername)
            
            if depMax == 0:
                depMax = displacement[-1]
                
            indexLim = np.where(displacement <= depMax)[0]
            
            ax_load.plot(displacement[indexLim], np.abs(load[indexLim]), label=text)

        else:
            if nomSimu not in missingSimulations: missingSimulations.append(nomSimu)
            Display.myPrintError("Data are not available:\n"+fileForceDep)

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
            Display.myPrintError("Simu is not available:\n"+fileForceDep)

        if plotDamage and Folder.Exists(fileSimu):

            # Display.Plot_Mesh(simu.mesh)

            if simulation == "PlateWithHole_Benchmark":
                colorBarIsClose = True
            else:
                colorBarIsClose = False

            # Displays last damage
            Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=colorBarIsClose,
            folder=folder_save, filename=f"{split} tol{tolConv} last", plotMesh=False,
            title=f"{split}_{regu}_tol{tolConv}")

            # Recover snapshot iterations
            for dep in snapshots:
                
                i = np.where(displacement*1e6>=dep)[0][0]
                
                simu.Set_Iter(i)

                filenameDamage = f"{nomSimu}, ud = {np.round(displacement[i]*1e6,2)}"
                # titleDamage = filenameDamage

                titleDamage = split
                # filenameDamage = f"PlateBench {comp}_{split}_{regu}_{simpli2D}"

                Display.Plot_Result(simu, "damage", nodeValues=True, colorbarIsClose=colorBarIsClose, folder=folder_save, filename=filenameDamage,title=titleDamage)
        
        
        # text = text.replace("AnisotStrain","Spectral")
        tic.Tac("PostProcessing", split, False)

    # ax_load.set_xlabel("displacement [µm]")
    # ax_load.set_ylabel("load [kN/mm]")
    ax_load.set_xlabel("displacement")
    ax_load.set_ylabel("load")
    ax_load.grid()
    ax_load.legend()
    plt.figure(ax_load.figure)
    Display.Save_fig(folder_save, "load displacement")

    print('\n Missing Simulations :')
    [print(simul) for simul in missingSimulations]

    plt.show()