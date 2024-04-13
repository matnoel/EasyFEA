from ..utilities import Folder, Display
import numpy as np
from ._simu import _Simu

import pickle

# ----------------------------------------------
# Saving / Loading functions
# ----------------------------------------------
def Load_Simu(folder: str):
    """
    Load the simulation from the specified folder.

    Parameters
    ----------
    folder : str
        The name of the folder where the simulation is saved.

    Returns
    -------
    Simu
        The loaded simulation.
    """

    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())
    path_simu = Folder.Join(folder, "simulation.pickle")
    error = "The file simulation.pickle cannot be found."
    assert Folder.Exists(path_simu), error

    try:
        with open(path_simu, 'rb') as file:
            simu = pickle.load(file)
    except EOFError:
        Display.myPrintError(f"The file:\n{path_simu}\nis empty or corrupted.")
        return None

    assert isinstance(simu, _Simu), 'Must be a simu object'
    
    Display.myPrint(f'\nLoaded:\n{path_simu.replace(folder_PythonEF,"")}\n', 'green')

    return simu

def Save_Force_Displacement(force: np.ndarray, displacement: np.ndarray, folder:str):
    """Save the values of load and displacements in the folder"""
    
    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())
    filename = Folder.Join(folder, "force-displacement.pickle")

    if not Folder.os.path.exists(folder):
        Folder.os.mkdir(folder)

    values = {
        'force': force,
        'displacement' : displacement
    }

    with open(filename, "wb") as file:
        pickle.dump(values, file)    
    
    Display.myPrint(f'Saved:\n{filename.replace(folder_PythonEF,"")}\n','green')
    
def Load_Force_Displacement(folder:str):
    """Load forces and displacements

    Parameters
    ----------
    folder : str
        name of the folder in which the "force-displacement.pickle" file is saved

    return force, displacement
    """

    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())

    filename = Folder.Join(folder, "force-displacement.pickle")
    shortName = filename.replace(folder_PythonEF,'') 
    error = f"{shortName} does not exist"
    assert Folder.Exists(filename), error

    with open(filename, 'rb') as file:
        values = pickle.load(file)
    
    force = np.array(values['force'])
    displacement = np.array(values['displacement'])
    
    Display.myPrint(f'Loaded:\n{shortName}\n','green')

    return force, displacement