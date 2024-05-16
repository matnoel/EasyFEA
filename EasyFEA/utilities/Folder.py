# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to facilitate folder and file creation using (os)."""

import os

def Get_Path(filename="") -> str:
    """Returns the folder containing the file. Otherwise returns the EasyFEA folder."""
    
    if filename == "":
        # Returns the path to EasyFEA
        normPath = os.path.normpath(__file__)
        path = os.path.dirname(normPath) # utilities
        # path = os.getcwd() # utilities
        path = os.path.dirname(path) # EasyFEA (modules)
        path = os.path.dirname(path) # EasyFEA (folder)
    else:
        normPath = os.path.normpath(filename)
        # Returns the path to the file
        path = os.path.dirname(normPath)

    return path

PATH_EASYFEA = Get_Path()

def New_File(filename: str, folder=Get_Path(), results=False) -> str:
    """Returns the path to the file/folder (filename can be a file or a folder). If the path does not exist, the function will create directories.

    Parameters
    ----------
    filename : str
        file or folder name
    folder : str, optional
        folder to use, default Get_Path() -> EasyFEA
    results : bool, optional
        saves in folder/results/filename or folder/filename, default False
    """
    
    if results:
        folder = Join(folder, "results")
    filename = Join(folder, filename)

    if not os.path.exists(filename):
        if "." in filename:
            os.makedirs(folder, exist_ok=True)
        else:
            os.makedirs(filename)
            
    return filename

def Join(*args: str) -> str:
    """Builds the path based on a list of str."""
    return os.path.join(*args)

def Exists(path: str) -> bool:
    """Test whether a path exists. Returns False for broken symbolic links"""
    return os.path.exists(path)

def PhaseField_Folder(folder: str, material: str, split: str, regu: str, simpli2D: str, tolConv: float, solver: str, test: bool, optimMesh=False, closeCrack=False, nL=0, theta=0.0) -> str:
    """Create a phase field folder based on the specified arguments."""

    from EasyFEA import Materials    

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
        assert solver in Materials.PhaseField.Get_solvers()
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
        workFolder = Join(folder, "Test", name)
    else:
        workFolder = Join(folder, name)

    path_PythonEf = Get_Path(Get_Path())
    
    from . import Display
    Display.MyPrint(workFolder.replace(path_PythonEf,""), 'green')

    return Join(path, workFolder)