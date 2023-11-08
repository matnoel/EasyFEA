import Display
import Folder

plt = Display.plt
np = Display.np
import pandas as pd

Display.Clear()


# ----------------------------------------------
# FEMU
# ----------------------------------------------
folder = Folder.Get_Path(__file__)

folder_save = Folder.New_File(Folder.Join(["Essais FCBA", "FEMU"]), results=True)

df = pd.read_excel(Folder.Join([folder, "params_Essais.xlsx"]))
# df = pd.read_excel(Folder.Join([folder, "params_Essais new.xlsx"]))

print(df)

params = ["El","Et","Gl","vl"]
seuil = 0.35

essais = np.arange(df.shape[0])

errors = []

for param in params:

    # get values
    values_mean = df[param].values
    values_std = df["std "+param].values
    values_disp = values_std / values_mean

    # plot
    ax = plt.subplots()[1]
    ax.bar(essais, values_mean, align='center', yerr=values_std, capsize=5)    
    unite = " [MPa]" if param in ["El","Et","Gl"] else ""    
    param_latex = param.replace("l","_L")
    param_latex = param_latex.replace("t","_T")    
    ax.set_title(f"${param_latex}$" + unite)
    ax.set_xlabel("Samples")
    ax.set_xticks(essais)

    Display.Save_fig(folder_save, param + " essais")    

    # check for erroes
    if "v" in param: continue    
    errors.extend(essais[values_disp>=seuil])

errors = np.unique(errors)
notErrors = [essai for essai in essais if essai not in errors]

print(errors)

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

df_stats.to_excel(Folder.Join([folder_save, f"stats rm{seuil}.xlsx"]), float_format="%.2f")


pass






# ----------------------------------------------
# FEMU GC
# ----------------------------------------------
folder_FCBA = Folder.New_File("Essais FCBA",results=True)
folder_Identif = Folder.Join([folder_FCBA, "Identification"])

pathData = Folder.Join([folder_Identif, 'identification.xlsx'])

Display.Clear()

df = pd.read_excel(pathData)

# filtre = df["err"] >= 1e-2
# filtre &= df["regu"] == "AT2"
# filtre = df["Essai"] <= 'Essai10'
filtre = df["test"] == False
# filtre = [True]*df.shape[0]

# filtre = filtre & (df['tolConv'] == 1e-3)
# filtre = filtre & (df['Essai'] == 'Essai00')
dfFiltre = df[filtre]
dfFiltre = dfFiltre.sort_values(by=['Essai'])

essais = np.unique(dfFiltre['Essai'])

print(dfFiltre)

print(essais)

pass