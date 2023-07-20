import os
from typing import List
from colorama import Fore

def Get_Path(filename="") -> str:
    """Returns the folder containing the file. Otherwise returns the PythonEF folder
    """
    
    if filename == "":
        # Returns the path to PythonEF
        path = os.path.dirname(__file__)
        path = os.path.dirname(path)
    else:
        # Returns the path to the file
        path = os.path.dirname(filename)

    return path

def New_File(filename: str, folder=Get_Path(), results=False) -> str:
    """Returns the path to the file/folder (filename can be a file or a folder). If the path does not exist, the function will create directories.

    Parameters
    ----------
    filename : str
        file or folder name
    folder : str, optional
        folder to use, default Get_Path() -> PythonEF
    results : bool, optional
        saves in folder/results/filename or folder/filename, default False
    """
    
    if results:
        folder = Join([folder, "results"])
    filename = Join([folder, filename])

    if not os.path.exists(folder):
        os.makedirs(folder)
            
    return filename

def Join(list: List[str]) -> str:
    """Builds the path based on a list of str."""

    file = ""
    for f in list:
        file = os.path.join(file, f)
        
    return file

def PhaseField_Folder(dossierSource: str, comp: str, split: str, regu: str, simpli2D: str, tolConv: float, solveur: str, test: bool, optimMesh=False, closeCrack=False, nL=0, theta=0.0):

    import Materials

    nom = ""

    if comp != "":
        nom += f"{comp}"        

    if split != "":
        start = "" if nom == "" else "_"
        nom += f"{start}{split}"

    if regu != "":        
        nom += f"_{regu}"

    if simpli2D != "":
        nom += f"_{simpli2D}"

    if closeCrack: 
        nom += '_closeCrack'

    if optimMesh:
        nom += '_optimMesh'

    if solveur != "History" and solveur != "":
        assert solveur in Materials.PhaseField_Model.get_solveurs()
        nom += '_' + solveur

    if tolConv < 1:
        nom += f'_conv{tolConv}'        
    
    if theta != 0.0:
        nom = f"{nom} theta={theta}"

    if nL != 0:
        assert nL > 0
        nom = f"{nom} nL={nL}"

    folder = New_File(dossierSource, results=True)

    if test:
        folder = Join([folder, "Test", nom])
    else:
        folder = Join([folder, nom])

    texteAvantPythonEF = folder.split('PythonEF')[0]
    folderSansArbre = folder.replace(texteAvantPythonEF, "")

    print(Fore.CYAN + '\nSimulation dans :\n'+folderSansArbre + Fore.WHITE)

    return folder