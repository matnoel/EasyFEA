# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""``Calc_projector``: an affine field is interpolated exactly between unrelated meshes."""

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh
from EasyFEA.FEM import Calc_projector
from EasyFEA.Geoms import Circle, Domain, Point

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


def test_new_nodes_outside_old_mesh():
    old, new = _box(ElemType.TETRA4, H / 2), _box(ElemType.TETRA4, H / 3)
    new.Translate(dx=0.1)  # nodes at x = L leave the old mesh
    outside = np.isclose(new.coord[:, 0], L + 0.1)

    proj = Calc_projector(old, new, extrapolate=False)
    assert proj[outside].nnz == 0

    proj = Calc_projector(old, new)
    np.testing.assert_allclose(proj @ _affine(old.coord), _affine(new.coord))


def test_extrapolation_without_xi_raises():
    """Around a coarse curved QUAD9 disk, Newton finds no ξ for some outside nodes."""
    oldDisk = Circle(Point(), 2.0, 2.0)
    old = oldDisk.Mesh_2D([], ElemType.QUAD9)
    newDisk = Circle(Point(), 2.2, 0.5)
    new = newDisk.Mesh_2D([], ElemType.TRI3)

    with pytest.raises(ValueError, match="Newton did not converge"):
        Calc_projector(old, new)
