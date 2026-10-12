# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Kinematics invariants of C = FᵀF and their derivatives in C, against closed forms under an affine u."""

import numpy as np
import pytest

from EasyFEA.FEM import Kinematics

from .conftest import F_3D, KELVIN_BASIS, affine_kinematics, at_every_point, kelvin

TOL = 1e-12

T4 = np.array([0.0, 1.0, 0.0])
T6 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)

C = F_3D.T @ F_3D


def _hessian_I3(C: np.ndarray) -> np.ndarray:
    """Kelvin Hessian of det(C) by central differences of its gradient det(C)·C⁻¹, exact as it is quadratic."""
    h = 1e-3
    c = kelvin(C)

    def grad(c):
        C = np.einsum("k,kij->ij", c, KELVIN_BASIS)
        return kelvin(np.linalg.det(C) * np.linalg.inv(C))

    return np.column_stack(
        [(grad(c + d) - grad(c - d)) / (2 * h) for d in np.eye(6) * h]
    )


INVARIANTS = {
    "I1": (Kinematics.Compute_I1, np.trace(C)),
    "I2": (Kinematics.Compute_I2, (np.trace(C) ** 2 - np.trace(C @ C)) / 2),
    "I3": (Kinematics.Compute_I3, np.linalg.det(C)),
    "I4": (lambda k: k.Compute_I4(T4), T4 @ C @ T4),
    "I6": (lambda k: k.Compute_I6(T6), T6 @ C @ T6),
    "I8": (lambda k: k.Compute_I8(T6, T4), T6 @ C @ T4),
}

FIRST_DERIVATIVES = {
    "dI1dC": (Kinematics.Compute_dI1dC, kelvin(np.eye(3))),
    "dI2dC": (Kinematics.Compute_dI2dC, kelvin(np.trace(C) * np.eye(3) - C)),
    "dI3dC": (Kinematics.Compute_dI3dC, kelvin(np.linalg.det(C) * np.linalg.inv(C))),
    "dI4dC": (lambda k: k.Compute_dI4dC(T4), kelvin(np.outer(T4, T4))),
    "dI6dC": (lambda k: k.Compute_dI6dC(T6), kelvin(np.outer(T6, T6))),
    "dI8dC": (lambda k: k.Compute_dI8dC(T6, T4), kelvin(np.outer(T6, T4))),
}

SECOND_DERIVATIVES = {
    "d2I1dC": (Kinematics.Compute_d2I1dC, np.zeros((6, 6))),
    "d2I2dC": (
        Kinematics.Compute_d2I2dC,
        np.outer(kelvin(np.eye(3)), kelvin(np.eye(3))) - np.eye(6),
    ),
    "d2I3dC": (Kinematics.Compute_d2I3dC, _hessian_I3(C)),
    "d2I4dC": (Kinematics.Compute_d2I4dC, np.zeros((6, 6))),
    "d2I6dC": (Kinematics.Compute_d2I6dC, np.zeros((6, 6))),
    "d2I8dC": (Kinematics.Compute_d2I8dC, np.zeros((6, 6))),
}


@pytest.fixture(scope="module")
def kinematics() -> Kinematics:
    """3D affine kinematics, whose F is `F_3D`."""
    return affine_kinematics(3).kinematics


def test_C_is_FtF(kinematics: Kinematics):
    assert at_every_point(kinematics.Compute_C(), C) < TOL


def test_green_lagrange_is_half_C_minus_identity(kinematics: Kinematics):
    E = kinematics.Compute_GreenLagrange()
    assert at_every_point(E, (C - np.eye(3)) / 2) < TOL


@pytest.mark.parametrize("compute, expected", INVARIANTS.values(), ids=INVARIANTS)
def test_invariant_is_the_closed_form(kinematics: Kinematics, compute, expected):
    assert at_every_point(compute(kinematics), expected) < TOL


@pytest.mark.parametrize(
    "compute, expected", FIRST_DERIVATIVES.values(), ids=FIRST_DERIVATIVES
)
def test_first_derivative_is_the_closed_form(kinematics: Kinematics, compute, expected):
    assert at_every_point(compute(kinematics), expected) < TOL


@pytest.mark.parametrize(
    "compute, expected", SECOND_DERIVATIVES.values(), ids=SECOND_DERIVATIVES
)
def test_second_derivative_is_the_closed_form(
    kinematics: Kinematics, compute, expected
):
    assert at_every_point(compute(kinematics), expected) < TOL
