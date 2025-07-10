# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing functions used to facilitate folder and file creation using (os)."""

import os


def Dir(path="") -> str:
    """Returns the directory of the specified path.\n
    If no path is specified, returns the EasyFEA directory path.
    """
    # TODO Give by default return folder

    assert isinstance(path, str), "filename must be str"

    if path == "":
        dir = EASYFEA_DIR
    else:
        normPath = os.path.normpath(path)
        dir = os.path.dirname(normPath)

    return dir


EASYFEA_DIR = Dir(Dir(Dir(__file__)))
RESULTS_DIR = os.path.join(EASYFEA_DIR, "results")
"""EASYFEA_DIR/results"""


def Join(*args: str, mkdir=False) -> str:
    """Joins two or more pathname components and create (or not) the path."""

    path = os.path.join(*args)

    if not Exists(path) and mkdir:
        if "." in path:
            dir = Dir(path)
            os.makedirs(dir, exist_ok=True)
        else:
            os.makedirs(path)

    return path


def Exists(path: str) -> bool:
    """Test whether a path exists. Returns False for broken symbolic links"""
    return os.path.exists(path)


def PhaseField_Folder(
    folder: str,
    material: str,
    split: str,
    regu: str,
    simpli2D: str,
    tolConv: float,
    solver: str,
    test: bool,
    optimMesh=False,
    closeCrack=False,
    nL=0,
    theta=0.0,
) -> str:
    """Creates a phase field folder based on the specified arguments."""

    from EasyFEA import Models

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
        name += "_closeCrack"

    if optimMesh:
        name += "_optimMesh"

    if solver != "History" and solver != "":
        assert solver in Models.PhaseField.Get_solvers()
        name += "_" + solver

    if tolConv < 1:
        name += f"_conv{tolConv}"

    if theta != 0.0:
        name = f"{name} theta={theta}"

    if nL != 0:
        assert nL > 0
        if isinstance(nL, float):
            name = f"{name} nL={nL:.2f}"
        else:
            name = f"{name} nL={nL}"

    workFolder = Join(RESULTS_DIR, "PhaseField", folder, mkdir=True)
    path = workFolder.split(folder)[0]

    if test:
        workFolder = Join(folder, "Test", name)
    else:
        workFolder = Join(folder, name)

    return Join(path, workFolder)
