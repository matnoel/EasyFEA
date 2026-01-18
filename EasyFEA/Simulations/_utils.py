# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ..Utilities import Folder, Display

import pickle

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

    Display.MyPrint(f"Saved:\n{file.replace(Folder.EASYFEA_DIR, '')}\n", "green")


def Load_pickle(folder: str, filename: str):
    """Returns folder/filename.pickle object."""

    file = Folder.Join(folder, f"{filename}.pickle")

    shortName = file.replace(Folder.EASYFEA_DIR, "")
    error = f"{shortName} does not exist"
    assert Folder.Exists(file), error

    with open(file, "rb") as f:
        obj = pickle.load(f)

    Display.MyPrint(f"Loaded:\n{shortName}\n", "green")

    return obj
