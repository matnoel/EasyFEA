# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import numpy as np

from EasyFEA.FEM import ElemType, GroupElemFactory, Mesh, _GroupElem

MeshOf = Callable[[ElemType], Mesh]

MEASURES = {1: 1.0, 2: 2.0, 3: 4.0}
"""of `mesh_of` (conftest): unit line on x, 2 × 1 rectangle, 2 × 1 × 2 box"""

REFERENCE_MEASURES = {
    "SEG": 2.0,
    "TRI": 1 / 2,
    "QUAD": 4.0,
    "TETRA": 1 / 6,
    "HEXA": 8.0,
    "PRISM": 1.0,
}
"""Length, area or volume of each reference element"""


def reference_element(
    elemType: ElemType, A: np.ndarray | None = None, b=(0, 0, 0)
) -> _GroupElem:
    """One `elemType` element whose nodes are `A ξ + b`, ξ its reference coordinates; `A` is (3, dim), identity by default."""
    nPe = GroupElemFactory.DICT_ELEMTYPE[elemType][1]
    connect = np.arange(nPe)[np.newaxis]
    local = GroupElemFactory.Create(elemType, connect, np.zeros((nPe, 3)))
    xi = local.Get_Local_Coords()
    A = np.eye(3)[:, : xi.shape[1]] if A is None else np.asarray(A, dtype=float)
    coord = xi @ A.T + np.asarray(b, dtype=float)
    return GroupElemFactory.Create(elemType, connect, coord)
