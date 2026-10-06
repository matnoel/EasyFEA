# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import ElemType, MatrixType
from EasyFEA.FEM import Kinematics
from EasyFEA.Geoms import Domain


@pytest.fixture
def kinematics() -> Kinematics:
    mesh = Domain((0, 0), (1, 1), 0.5).Mesh_2D([], ElemType.QUAD4)
    u = np.random.default_rng(0).standard_normal(mesh.Nn * 2) * 1e-2
    return Kinematics(mesh.groupElem, u)


def test_measure_is_computed_once(kinematics: Kinematics, monkeypatch):
    groupElem = kinematics.groupElem
    calls = []
    original = groupElem.Get_Gradient_e_pg

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(groupElem, "Get_Gradient_e_pg", counting)

    eps = kinematics.Compute_Epsilon()
    assert kinematics.Compute_Epsilon() is eps
    kinematics.Compute_F()
    kinematics.Compute_GreenLagrange()
    assert len(calls) == 1


def test_matrixType_is_read_only(kinematics: Kinematics):
    assert kinematics.matrixType == MatrixType.rigi
    with pytest.raises(AttributeError):
        kinematics.matrixType = MatrixType.mass  # type: ignore[misc]


def test_displacement_e_pg_is_not_cached(kinematics: Kinematics):
    groupElem = kinematics.groupElem
    u_e_pg = kinematics.displacement_e_pg
    nPg = groupElem.Get_gauss(kinematics.matrixType).nPg
    assert u_e_pg.shape == (groupElem.Ne, nPg, groupElem.nPe * 2)
    assert np.allclose(
        u_e_pg[:, 0], kinematics.displacement[groupElem.Get_assembly_e(2)]
    )
    assert kinematics.displacement_e_pg is not u_e_pg
