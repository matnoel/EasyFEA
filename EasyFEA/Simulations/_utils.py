# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ..Utilities import Folder

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


def Load_pickle(folder: str, filename: str):
    """Returns folder/filename.pickle object."""

    file = Folder.Join(folder, f"{filename}.pickle")

    shortName = file.replace(Folder.EASYFEA_DIR, "")
    error = f"{shortName} does not exist"
    assert Folder.Exists(file), error

    with open(file, "rb") as f:
        obj = pickle.load(f)

    return obj
