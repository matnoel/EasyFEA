# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from ..utilities import Folder, Display
import numpy as np

import pickle

# ----------------------------------------------
# Saving / Loading force-displacement.pickle
# ----------------------------------------------

def Save_Force_Displacement(force: np.ndarray, displacement: np.ndarray, folder:str):
    """Saves the values of force and displacement in the folder"""
    
    filename = Folder.Join(folder, "force-displacement.pickle")

    if not Folder.os.path.exists(folder):
        Folder.os.makedirs(folder)

    values = {
        'force': force,
        'displacement' : displacement
    }

    with open(filename, "wb") as file:
        pickle.dump(values, file)    
    
    Display.MyPrint(f'Saved:\n{filename.replace(Folder.EASYFEA_DIR,"")}\n','green')
    
def Load_Force_Displacement(folder:str):
    """Loads force and displacement.

    Parameters
    ----------
    folder : str
        name of the folder in which the "force-displacement.pickle" file is saved

    return force, displacement
    """

    filename = Folder.Join(folder, "force-displacement.pickle")
    shortName = filename.replace(Folder.EASYFEA_DIR,'') 
    error = f"{shortName} does not exist"
    assert Folder.Exists(filename), error

    with open(filename, 'rb') as file:
        values = pickle.load(file)
    
    force = np.array(values['force'])
    displacement = np.array(values['displacement'])
    
    Display.MyPrint(f'Loaded:\n{shortName}\n','green')

    return force, displacement

# ----------------------------------------------
# Save obj in pickle file
# ----------------------------------------------

def Save_pickle(obj, folder: str, filename: str) -> None:
    """Saves the object in folder/filename.pickle."""
    
    file = Folder.Join(folder, f"{filename}.pickle")

    if not Folder.os.path.exists(folder):
        Folder.os.makedirs(folder)

    with open(file, "wb") as f:
        pickle.dump(obj, f)
    
    Display.MyPrint(f'Saved:\n{file.replace(Folder.EASYFEA_DIR,"")}\n','green')

def Load_pickle(folder:str, filename: str):
    """Returns folder/filename.pickle object."""

    file = Folder.Join(folder, f"{filename}.pickle")

    shortName = file.replace(Folder.EASYFEA_DIR,'') 
    error = f"{shortName} does not exist"
    assert Folder.Exists(file), error

    with open(file, 'rb') as f:
        obj = pickle.load(f)
    
    Display.MyPrint(f'Loaded:\n{shortName}\n','green')

    return obj