# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Small meshes and the element-restriction check of the operator tests."""

from typing import Callable

import numpy as np

from EasyFEA.FEM import ElemType, Mesh
from EasyFEA.Geoms import Domain

from _numerics import rel_error


def element_mesh(dim: int, elemType: ElemType, layers: int = 1) -> Mesh:
    """Organised unit square or cube in `layers`² or `layers`³ cells, few elements to keep finite differences cheap."""
    contour = Domain((0, 0), (1, 1), 1 / layers)
    if dim == 2:
        return contour.Mesh_2D([], elemType, isOrganised=True)
    return contour.Mesh_Extrude([], [0, 0, 1], [layers], elemType, isOrganised=True)


def assert_restricts(operator: Callable, elements: np.ndarray) -> None:
    """`operator(elements)` keeps the rows of `operator(None)` on `elements` and zeros every other."""
    full, restricted = operator(None), operator(elements)
    assert rel_error(restricted[elements], full[elements]) < 1e-14
    assert not np.any(np.delete(restricted, elements, axis=0))
