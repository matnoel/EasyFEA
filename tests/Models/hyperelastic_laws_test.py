# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import Models
from EasyFEA.FEM import Kinematics

from _numerics import rel_error
from .conftest import GRAD_U, affine_kinematics, kelvin

TOL = 1e-12

LMBDA, MU = 121153.0, 80769.0

F = np.eye(3) + GRAD_U
E = (F.T @ F - np.eye(3)) / 2
I = kelvin(np.eye(3))

SVK = Models.HyperElastic.SaintVenantKirchhoff(3, LMBDA, MU)


@pytest.fixture(scope="module")
def kinematics() -> Kinematics:
    """3D affine kinematics, whose Green-Lagrange strain is `E` above."""
    return affine_kinematics(3).kinematics


def test_svk_W_is_its_closed_form(kinematics: Kinematics):
    W = LMBDA / 2 * np.trace(E) ** 2 + MU * np.trace(E @ E)
    assert rel_error(SVK.Compute_W(kinematics), W) < TOL


def test_svk_dWde_is_its_closed_form(kinematics: Kinematics):
    dW = LMBDA * np.trace(E) * I + 2 * MU * kelvin(E)
    actual = np.asarray(SVK.Compute_dWde(kinematics))
    assert rel_error(actual, np.broadcast_to(dW, actual.shape)) < TOL


def test_svk_d2Wde_is_its_closed_form(kinematics: Kinematics):
    d2W = LMBDA * np.outer(I, I) + 2 * MU * np.eye(6)
    actual = np.asarray(SVK.Compute_d2Wde(kinematics))
    assert rel_error(actual, np.broadcast_to(d2W, actual.shape)) < TOL


def test_Available_Laws_lists_every_shipped_law():
    """Every concrete law except AutoDiff, which runs a potential the user supplies."""
    from EasyFEA.Models.HyperElastic._laws import _HyperElastic

    shipped = {
        law
        for law in _HyperElastic.__subclasses__()
        if law is not Models.HyperElastic.AutoDiff
    }

    assert set(_HyperElastic.Available_Laws()) == shipped
