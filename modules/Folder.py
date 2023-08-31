"""Folder and file creation module."""

import os
from typing import List
from colorama import Fore

def Get_Path(filename="") -> str:
    """Returns the folder containing the file. Otherwise returns the PythonEF folder."""
    
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

def PhaseField_Folder(folder: str, material: str, split: str, regu: str, simpli2D: str, tolConv: float, solver: str, test: bool, optimMesh=False, closeCrack=False, nL=0, theta=0.0) -> str:
    """Create a phase field folder based on the specified arguments."""

    import Materials

    name = ""

    if material != "":
        name += f"{material}"        

    if split != "":
        start = "" if name == "" else "_"
        name += f"{start}{split}"

    if regu != "":        
        name += f"_{regu}"

    if simpli2D != "":
        name += f"_{simpli2D}"

    if closeCrack: 
        name += '_closeCrack'

    if optimMesh:
        name += '_optimMesh'

    if solver != "History" and solver != "":
        assert solver in Materials.PhaseField_Model.get_solvers()
        name += '_' + solver

    if tolConv < 1:
        name += f'_conv{tolConv}'        
    
    if theta != 0.0:
        name = f"{name} theta={theta}"

    if nL != 0:
        assert nL > 0
        name = f"{name} nL={nL}"

    workFolder = New_File(folder, results=True)
    path = workFolder.split(folder)[0]

    if test:
        workFolder = Join([folder, "Test", name])
    else:
        workFolder = Join([folder, name])

    print(Fore.CYAN + '\nWorking in :\n'+ workFolder + Fore.WHITE)    

    return Join([path, workFolder])