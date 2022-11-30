import os
from typing import List
from colorama import Fore

# import 
# export PYTHONPATH=$PYTHONPATH:/home/matthieu/Documents/PythonEF/classes
# export PYTHONPATH=$PYTHONPATH:/home/m/matnoel/Documents/PythonEF/classes

def Get_Path(filename="") -> str:
    """Renvoie le path du fichier ou renvoie le path vers le dossier PythonEF

    Parameters
    ----------
    filename : str, optional
        fichier, by default ""

    Returns
    -------
    str
        filename complet
    """
    
    if filename == "":
        # Renvoie le path vers PythonEF
        path = os.path.dirname(__file__)
        path = os.path.dirname(path)
    else:
        # Renvoie le path vers le fichier
        path = os.path.dirname(filename)    

    return path

def New_File(filename: str, pathname=Get_Path(), results=False) -> str:
    """Renvoie le path vers le fichier avec l'extension ou non\n
    filename peut etre : un fichier ou un dossier\n
    De base le path renvoie vers le path ou est PythonEF

    Parameters
    ----------
    filename : str
        nom du fichier ou du dossier
    pathname : str, optional
        _description_, by default GetPath()
    results : bool, optional
        enregistre dans PythonEF/results/filename ou pathname/filename, by default False

    Returns
    -------
    str
        filename complet
    """
    
    if results:
        pathname = os.path.join(pathname, "results")
    filename = os.path.join(pathname, filename)

    if not os.path.exists(pathname):
        os.mkdir(pathname)

            
    return filename

def Join(list: List[str]) -> str:
    """Construit le path en fonction d'une liste de nom

    Parameters
    ----------
    list : List[str]
        liste de nom

    Returns
    -------
    str
        filename complet
    """

    file = ""
    for f in list:
        file = os.path.join(file, f)
        
    return file

def PhaseField_Folder(dossierSource: str, comp: str, split: str, regu: str, simpli2D: str, tolConv: float, solveur: str, test: bool, optimMesh=False, closeCrack=False, v=0.0, nL=0, theta=0.0):

    import Materials

    nom="_".join([comp, split, regu, simpli2D])

    if closeCrack: 
        nom += '_closeCrack'

    if optimMesh:
        nom += '_optimMesh'

    assert solveur in Materials.PhaseField_Model.get_solveurs()
    if solveur != "History":
        nom += '_' + solveur

    if tolConv < 1:
        nom += f'_conv{tolConv}'
        
    if comp == "Elas_Isot" and v != 0:
        nom = f"{nom} pour v={v}"

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