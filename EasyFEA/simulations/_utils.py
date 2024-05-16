# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from ..utilities import Folder, Display
import numpy as np

import pickle

# ----------------------------------------------
# Saving / Loading force-displacement.pickle
# ----------------------------------------------

def Save_Force_Displacement(force: np.ndarray, displacement: np.ndarray, folder:str):
    """Save the values of load and displacements in the folder"""
    
    folder_PythonEF = Folder.Get_Path(Folder.Get_Path())
    filename = Folder.Join(folder, "force-displacement.pickle")

    if not Folder.os.path.exists(folder):
        Folder.os.makedirs(folder)

    values = {
        'force': force,
        'displacement' : displacement
    }

    with open(filename, "wb") as file:
        pickle.dump(values, file)    
    
    Display.MyPrint(f'Saved:\n{filename.replace(folder_PythonEF,"")}\n','green')
    
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
    
    Display.MyPrint(f'Loaded:\n{shortName}\n','green')

    return force, displacement