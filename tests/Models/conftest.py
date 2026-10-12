# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import NamedTuple

import numpy as np
import pytest

from EasyFEA import ElemType
from EasyFEA.FEM import Kinematics
from EasyFEA.Geoms import Domain
from EasyFEA.Models.InElastic import MaterialPoint
from EasyFEA.Models.InElastic._behavior import PointUpdate

from _numerics import rel_error

# fully populated and non-symmetric, so no component of ε or F is trivially zero
GRAD_U = np.array([[0.02, -0.01, 0.03], [0.015, -0.025, 0.01], [-0.02, 0.005, 0.04]])


TANGENT_RTOL = 1e-6


def assert_consistent_tangent(
    point: MaterialPoint,
    strain: np.ndarray,
    internals: dict[str, np.ndarray] | None = None,
    dt: float = 0.0,
    h: float = 1e-9,
) -> PointUpdate:
    """Asserts the tangent of `point.Update` at `strain` is its stress derivative by central differences; returns that update."""
    update = point.Update(strain, internals, dt)
    C_fd = np.column_stack(
        [
            (
                point.Update(strain + d, internals, dt).stress
                - point.Update(strain - d, internals, dt).stress
            )
            / (2 * h)
            for d in np.eye(strain.size) * h
        ]
    )
    assert rel_error(update.tangent, C_fd) < TANGENT_RTOL
    return update


class AffineKinematics(NamedTuple):
    """Kinematics of an affine `u = ∇u·x`, with the ε (Kelvin-Mandel) and F (3×3) it must yield."""

    kinematics: Kinematics
    epsilon: np.ndarray
    F: np.ndarray


def affine_kinematics(dim: int, Ne: int = 1) -> AffineKinematics:
    """`Ne` QUAD4 / HEXA8 unit elements in a row, displaced by the affine `u = GRAD_U[:dim, :dim]·x`."""
    domain = Domain((0, 0), (Ne, 1), 1.0)
    if dim == 2:
        mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    else:
        mesh = domain.Mesh_Extrude(
            [],
            [0, 0, 1],
            [1],
            ElemType.HEXA8,
            isOrganised=True,
        )
    G = np.zeros((3, 3))
    G[:dim, :dim] = GRAD_U[:dim, :dim]
    u = mesh.coord[:, :dim] @ G[:dim, :dim].T

    eps = (G + G.T) / 2
    r2 = np.sqrt(2)
    if dim == 2:
        epsilon = np.array([eps[0, 0], eps[1, 1], r2 * eps[0, 1]])
    else:
        epsilon = np.array(
            [*np.diag(eps), r2 * eps[1, 2], r2 * eps[0, 2], r2 * eps[0, 1]]
        )
    return AffineKinematics(
        Kinematics(mesh.groupElem, u.ravel()), epsilon, np.eye(3) + G
    )


@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def affine(request) -> AffineKinematics:
    """`affine_kinematics` in 2D and 3D."""
    return affine_kinematics(request.param)
