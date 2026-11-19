# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The rows a rank owns or holds, as the tests read them."""

import numpy as np

from EasyFEA.Utilities._mpi import Owned


def owned_elements(groupElem) -> np.ndarray:
    """Rows of the elements this rank owns."""
    return np.arange(groupElem.Ne)[Owned(groupElem._elementOwners, groupElem.rank)]


def owned_nodes(space) -> np.ndarray:
    """Node ids of the DOF rows this rank owns."""
    return space._dofNodes[Owned(space._dofOwners, space.rank)]


def held_nodes(space) -> np.ndarray:
    """Node ids of this rank's main-dimension elements, ghosts included."""
    return np.unique(np.concatenate([g.nodes for g in space.Get_list_groupElem()]))
