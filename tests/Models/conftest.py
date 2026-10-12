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
from EasyFEA.Models.InElastic import MaterialPoint, PointUpdate

from _numerics import rel_error

# fully populated and non-symmetric, so no component of ε or F is trivially zero
GRAD_U = np.array([[0.02, -0.01, 0.03], [0.015, -0.025, 0.01], [-0.02, 0.005, 0.04]])
F_3D = np.eye(3) + GRAD_U
"""F of the 3D `affine_kinematics`."""


R2 = np.sqrt(2)
KELVIN_BASIS = np.array(
    [
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 1 / R2], [0, 1 / R2, 0]],
        [[0, 0, 1 / R2], [0, 0, 0], [1 / R2, 0, 0]],
        [[0, 1 / R2, 0], [1 / R2, 0, 0], [0, 0, 0]],
    ]
)
"""Orthonormal basis of symmetric 3×3 tensors, in Kelvin-Mandel order `[xx, yy, zz, yz, xz, xy]`."""


def kelvin(A: np.ndarray) -> np.ndarray:
    """(6,) Kelvin-Mandel vector of the symmetric part of a 3×3 tensor."""
    return np.einsum("kij,ij->k", KELVIN_BASIS, A)


def at_every_point(actual, expected: np.ndarray) -> float:
    """`rel_error` of a (Ne, nPg, ...) field against one point value; absolute where `expected` vanishes."""
    actual = np.asarray(actual)
    scale = None if np.any(expected) else 1.0
    return rel_error(actual, np.broadcast_to(expected, actual.shape), scale)


TANGENT_RTOL = 1e-6
"""Per entry, plus an absolute floor of TANGENT_RTOL·1e-3 of the largest entry for the vanishing ones."""


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
    floor = 1e-3 * TANGENT_RTOL * np.abs(C_fd).max()
    assert np.allclose(update.tangent, C_fd, rtol=TANGENT_RTOL, atol=floor)
    return update


class AffineKinematics(NamedTuple):
    """Kinematics of an affine `u = ∇u·x`, with the ε (Kelvin-Mandel) and F (3×3) it must yield."""

    kinematics: Kinematics
    epsilon: np.ndarray
    F: np.ndarray


def affine_kinematics(dim: int, Ne: int = 1, scale: float = 1.0) -> AffineKinematics:
    """`Ne` QUAD4 / HEXA8 unit elements in a row, displaced by the affine `u = scale·GRAD_U[:dim, :dim]·x`."""
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
    G[:dim, :dim] = scale * GRAD_U[:dim, :dim]
    u = mesh.coord[:, :dim] @ G[:dim, :dim].T

    epsilon = kelvin(G)
    if dim == 2:
        epsilon = epsilon[[0, 1, 5]]
    return AffineKinematics(
        Kinematics(mesh.groupElem, u.ravel()), epsilon, np.eye(3) + G
    )


@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def affine(request) -> AffineKinematics:
    """`affine_kinematics` in 2D and 3D."""
    return affine_kinematics(request.param)
