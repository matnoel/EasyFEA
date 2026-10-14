# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import NamedTuple

import numpy as np

from EasyFEA import ElemType
from EasyFEA.FEM import Kinematics
from EasyFEA.Geoms import Domain
from EasyFEA.Models.InElastic import MaterialPoint, PointUpdate

from _numerics import affine, assert_tangent_matches_fd, mandel, rel_error

# fully populated and non-symmetric, so no component of ε or F is trivially zero
GRAD_U = np.array([[0.02, -0.01, 0.03], [0.015, -0.025, 0.01], [-0.02, 0.005, 0.04]])
F_3D = np.eye(3) + GRAD_U
"""F of the 3D `affine_kinematics`."""


def at_every_point(actual, expected: np.ndarray) -> float:
    """`rel_error` of a (Ne, nPg, ...) field against one point value; absolute where `expected` vanishes."""
    actual = np.asarray(actual)
    scale = None if np.any(expected) else 1.0
    return rel_error(actual, np.broadcast_to(expected, actual.shape), scale)


def assert_point_tangent_matches_fd(
    point: MaterialPoint,
    strain: np.ndarray,
    internals: dict[str, np.ndarray] | None = None,
    dt: float = 0.0,
) -> PointUpdate:
    """Asserts the tangent of `point.Update` at `strain` is its stress derivative by `assert_tangent_matches_fd`; returns that update."""
    update = point.Update(strain, internals, dt)
    assert_tangent_matches_fd(
        lambda e: point.Update(e, internals, dt).stress[None],
        strain,
        update.tangent[None],
        np.arange(strain.size)[None],
    )
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
    u = affine(mesh.coord, G[:dim, :dim])
    return AffineKinematics(
        Kinematics(mesh.groupElem, u), mandel(G, dim), np.eye(3) + G
    )
