import Display
import Folder

plt = Display.plt
np = Display.np
import pandas as pd

folder_FCBA = Folder.New_File("Essais FCBA",results=True)
folder_Identif = Folder.Join([folder_FCBA, "Identification"])

pathData = Folder.Join([folder_Identif, 'identification.xlsx'])

Display.Clear()

df = pd.read_excel(pathData)

filtre = df["err"] >= 1e-2
# filtre &= df["regu"] == "AT2"
# filtre = df["Essai"] <= 'Essai10'
# filtre = [True]*df.shape[0]

# filtre = filtre & (df['tolConv'] == 1e-3)
# filtre = filtre & (df['Essai'] == 'Essai00')


dfFiltre = df[filtre]
dfFiltre = dfFiltre.sort_values(by=['Essai','err'])

essais = np.unique(dfFiltre['Essai'])


print(dfFiltre)

print(essais)














pass