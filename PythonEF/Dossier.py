import os
from typing import List
# import 
# export PYTHONPATH=$PYTHONPATH:/home/matthieu/Documents/PythonEF/classes
# export PYTHONPATH=$PYTHONPATH:/home/m/matnoel/Documents/PythonEF/classes

def GetPath(filename="") -> str:
    """Renvoie le path du fichier ou renvoie le path vers le dossier Python Ef

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

def NewFile(filename: str, pathname=GetPath(), results=False) -> str:
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
        
    destination = GetPath(filename)    

    if not os.path.isdir(destination):
        # os.mkdir(destination)
        os.makedirs(destination)

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

    if not os.path.isdir(file):
        if '.' in file:
            path = GetPath(file)
            if not os.path.exists(path):
                # os.mkdir(file)
                os.makedirs(path)
        else:
            if not os.path.exists(file):
                # os.mkdir(file)
                os.makedirs(file)
        
    return file





