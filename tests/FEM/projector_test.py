# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""``Calc_projector``: an affine field is interpolated exactly between unrelated meshes."""

import numpy as np

from EasyFEA import ElemType, Mesh
from EasyFEA.FEM import Calc_projector
from EasyFEA.Geoms import Domain, Point

L, H = 120.0, 13.0  # gmsh leaves prisms in a HEXA8 extrusion of this domain


def _box(elemType: ElemType, meshSize: float) -> Mesh:
    domain = Domain(Point(0, 0), Point(L, H), meshSize)
    return domain.Mesh_Extrude([], [0, 0, H], [3], elemType)


def _affine(coord: np.ndarray) -> np.ndarray:
    return coord @ np.array([1.0, -2.0, 3.0]) + 0.5


def test_identity_by_default():
    mesh = _box(ElemType.TETRA4, H / 2)
    proj = Calc_projector(mesh, mesh)
    np.testing.assert_allclose(proj.toarray(), np.eye(mesh.Nn), atol=1e-12)


def test_affine_field_between_unrelated_meshes():
    old, new = _box(ElemType.TETRA4, H / 2), _box(ElemType.TETRA4, H / 3)
    proj = Calc_projector(old, new)
    np.testing.assert_allclose(proj @ _affine(old.coord), _affine(new.coord))


def test_mixed_old_mesh():
    old = _box(ElemType.HEXA8, H / 2)
    assert len(old.Get_list_groupElem()) == 2  # HEXA8 + PRISM6
    new = _box(ElemType.TETRA4, H / 3)
    proj = Calc_projector(old, new)
    np.testing.assert_allclose(proj @ _affine(old.coord), _affine(new.coord))
