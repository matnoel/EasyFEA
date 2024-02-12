"""Post process the experimental data."""

import Display
import Folder

plt = Display.plt
np = Display.np
import pandas as pd
import Functions

Display.Clear()

folder = Folder.Get_Path(__file__)
folder_femu = Folder.New_File(Folder.Join("FCBA", "FEMU"), results=True)
folder_iden = Folder.New_File(Folder.Join("FCBA", "Identification"), results=True)

if __name__ == '__main__':

    # --------------------------------------------------------------------------------------------
    # FEMU Elas
    # --------------------------------------------------------------------------------------------
    Display.Section("FEMU Elas")

    df = Functions.dfParams.copy()

    # print(df)

    params = ["El","Et","Gl","vl"]
    tolError = 0.4

    essais = np.arange(df.shape[0])

    errors = []
    # errors = [2, 5, 6, 9, 11, 15, 17, 19, 20, 29, 30, 32]
    notErrors = [essai for essai in essais if essai not in errors]

    for param in params:

        # get values
        values_mean = df[param].values
        values_std = df["std "+param].values
        values_disp = values_std / values_mean

        # plot
        ax = plt.subplots()[1]
        ax.bar(essais[notErrors], values_mean[notErrors],
            align='center', yerr=values_std[notErrors], capsize=5)
        unite = " [MPa]" if param in ["El","Et","Gl"] else ""    
        param_latex = param.replace("l","_L")
        param_latex = param_latex.replace("t","_T")    
        ax.set_title(f"${param_latex}$" + unite)
        ax.set_xlabel("Essais")
        ax.set_xticks(essais)

        Display.Save_fig(folder_femu, param + " essais")    

        # # check for errors
        # if "v" in param: continue    
        # errors.extend(essais[values_disp>=seuil])

    # 
    errors = np.unique(errors)
    notErrors = [essai for essai in essais if essai not in errors]
    print(f"errors = {errors}")

    # Stats on not error samples
    list_dict = []
    for param in params:
        
        mean = df[param].values[notErrors].mean()
        std = df[param].values[notErrors].std()
        disp = std / mean

        list_dict.append(
            {"param": param,
            "mean": mean,
            "std": std,
            "disp %": disp*100
            }
        )
    df_stats = pd.DataFrame(list_dict)
    df_stats.to_excel(Folder.Join(folder_femu, f"stats rm{tolError}.xlsx"), float_format="%.2f")

    # --------------------------------------------------------------------------------------------
    # FEMU GC
    # --------------------------------------------------------------------------------------------
    Display.Section("FEMU Gc")

    dfGc = Functions.dfGc.copy()
    
    dfGc = dfGc[(dfGc['solveur']==1)&(dfGc['ftol']==1e-5)]
    dfGc = dfGc.sort_values(by=['Essai'])
    dfGc = dfGc.set_index(np.arange(dfGc.shape[0]))
    # dfGc = dfGc[:-1] # enlève essai 17

    # print(dfGc)

    # print(dfGc.describe())
    print(f"mean Gc = {dfGc['Gc'].mean():.3f}")
    print(f"std Gc = {dfGc['Gc'].std():.3f}")
    print(f"disp Gc = {dfGc['Gc'].std()/dfGc['Gc'].mean()*100:.2f} %")

    axFc = plt.subplots()[1]
    axFc.bar(dfGc.index, dfGc["f_crit"].values)
    axFc.set_xticks(dfGc.index)
    axFc.set_xlabel("Essais")
    axFc.set_ylabel("Crack initiation forces")
    Display.Save_fig(folder_iden, 'crack init essais')
    # axFcrit.tick_params(axis='x', labelrotation = 45)
    # plt.xlim([0, None])
    # plt.ylim([0, y_max])

    axGc = plt.subplots()[1]
    axGc.bar(dfGc.index, dfGc["Gc"].values)
    axGc.set_xticks(dfGc.index)
    # axGc.set_xlabel("Essais", fontsize=14)
    axGc.set_xlabel("Essais")
    axGc.set_ylabel("$G_c \ [mJ \ mm^{-2}]$")
    Display.Save_fig(folder_iden, 'Gc essais')

    # errors = [2,6,8,9,11,13,17]
    errors = []

    dfGc.drop(errors, axis=0, inplace=True)

    ax_fit = plt.subplots()[1]
    ax_fit.set_xlabel('$G_c$')
    ax_fit.set_ylabel('Crack initiation forces')

    f_crit = dfGc["f_crit"].values
    Gc = dfGc["Gc"].values
    for i in range(Gc.size):        
        ax_fit.scatter(Gc[i],f_crit[i],c='blue')
        ax_fit.text(Gc[i],f_crit[i],f'Essai{i}')


    from scipy.optimize import minimize
    J = lambda x: np.linalg.norm(f_crit - (x[0]*Gc + x[1]))

    res = minimize(J, [0,0])
    a, b = tuple(res.x)

    Gc_array = np.linspace(Gc.min(), Gc.max(), 100)
    curve: np.ndarray = a*Gc_array + b


    r = np.mean((Gc-Gc.mean())/Gc.std() * (f_crit-f_crit.mean())/f_crit.std())
    # r = np.corrcoef(Gc,f_crit)[0,1]


    ax_fit.plot(Gc_array, curve,c='red')

    ax_fit.text(Gc_array.mean(), curve.mean(), f"{a:.3f} Gc + {b:.3f}, r={r:.3f}", va='top')
    ax_fit.grid()
    # bbox=dict(boxstyle="square,pad=0.3",alpha=1,color='white')

    Display.Save_fig(folder_iden, "corr Gc fcrit", extension='pdf')


    Display.plt.show()

    pass