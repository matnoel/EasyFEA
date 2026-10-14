# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Meshes carrying one or several element groups, and the per-group strain/stress fields the simulations return on them."""

import numpy as np

from EasyFEA import ElemType, Mesh
from EasyFEA.FEM import FeArray, MatrixType
from EasyFEA.Geoms import Domain, Point

L, H = 120.0, 13.0


def bar_mesh(groups: str) -> Mesh:
    """L×H bar: `TRI3` (one group), `QUAD4+TRI3` or `HEXA8+PRISM6` (two)."""
    domain = Domain(Point(), Point(L, H), H / 2)
    if groups == "TRI3":
        return domain.Mesh_2D([], ElemType.TRI3)
    if groups == "QUAD4+TRI3":
        return domain.Mesh_2D([], ElemType.QUAD4)
    return domain.Mesh_Extrude([], [0, 0, H], [2], ElemType.HEXA8)


BAR_MESHES = ["TRI3", "QUAD4+TRI3", "HEXA8+PRISM6"]


def component(
    field, mesh: Mesh, index: int, matrixType: MatrixType = MatrixType.rigi
) -> np.ndarray:
    """Component `index` at every Gauss point, after checking `field` is an FeArray on one group and {groupElem: FeArray} on several."""
    groups = mesh.Get_list_groupElem()
    if len(groups) == 1:
        assert isinstance(field, FeArray)
        field = {groups[0]: field}
    assert isinstance(field, dict) and list(field) == groups
    for groupElem, field_e_pg in field.items():
        assert isinstance(field_e_pg, FeArray)
        nPg = groupElem.Get_gauss(matrixType).nPg
        assert field_e_pg.shape[:2] == (groupElem.Ne, nPg)
    return np.concatenate([np.ravel(f[..., index]) for f in field.values()])
