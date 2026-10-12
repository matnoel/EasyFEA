# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Exact references for Simulations tests: affine continuum solutions (exact on every element) and Euler-Bernoulli beam closed forms."""

from typing import Callable, NamedTuple

import numpy as np

Field = Callable[[np.ndarray], np.ndarray]


def bar_displacement(
    coord: np.ndarray, L: float, ud: float, v: float, dim: int
) -> np.ndarray:
    """(Nn, dim) displacement of a bar pulled by `ud` at x=L, held by symmetry on x=0, y=0, z=0: uniaxial stress (plane stress in 2D)."""
    strain = ud / L
    return strain * coord[:, :dim] * np.array([1.0, -v, -v])[:dim]


def bar_stress(E: float, L: float, ud: float) -> float:
    """Uniform σxx of that bar; every other stress component is zero."""
    return E * ud / L


SYMMETRY_PLANES = {
    "x": lambda x, y, z: x == 0,
    "y": lambda x, y, z: y == 0,
    "z": lambda x, y, z: z == 0,
}


def add_bar_supports(simu) -> None:
    """Rollers on the x=0, y=0 (and z=0) planes: the bar expands freely, so any homogeneous state is exact."""
    mesh = simu.mesh
    for unknown in ("x", "y", "z")[: mesh.dim]:
        nodesSymmetry = mesh.Nodes_Conditions(SYMMETRY_PLANES[unknown])
        simu.add_dirichlet(nodesSymmetry, [0], [unknown])


def pull_bar(simu, ud: float) -> None:
    """Supports of `add_bar_supports`, and ux = `ud` on the far end x=L."""
    mesh = simu.mesh
    L = mesh.coord[:, 0].max()
    add_bar_supports(simu)
    nodesPulled = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesPulled, [ud], ["x"])


def slab_temperature(x: np.ndarray, L: float, T0: float, TL: float) -> np.ndarray:
    """Steady temperature of a slab held at T0 on x=0 and TL on x=L, insulated elsewhere."""
    return T0 + (TL - T0) * x / L


def rect_Iz(b: float, h: float) -> float:
    """Second moment of a b×h rectangle about its z axis."""
    return b * h**3 / 12


class BeamFields(NamedTuple):
    """Deflection, rotation, bending moment and shear force along x; F, q < 0 point down, Mz > 0 sags."""

    uy: Field
    rz: Field
    Mz: Field
    Ty: Field


def _macaulay(x: np.ndarray, a: float, n: int) -> np.ndarray:
    return np.where(x > a, (x - a) ** n, 0.0)


def cantilever_tip(L: float, F: float, EI: float) -> BeamFields:
    """Clamped at x=0, point force F at the free tip x=L."""
    return BeamFields(
        uy=lambda x: F * (L * x**2 / 2 - x**3 / 6) / EI,
        rz=lambda x: F * (L * x - x**2 / 2) / EI,
        Mz=lambda x: F * (L - x),
        Ty=lambda x: np.full_like(x, F, dtype=float),
    )


def cantilever_distributed(L: float, q: float, EI: float) -> BeamFields:
    """Clamped at x=0, free at x=L, uniform load q."""
    return BeamFields(
        uy=lambda x: q * x**2 * (6 * L**2 - 4 * L * x + x**2) / (24 * EI),
        rz=lambda x: q * (x**3 / 6 - L * x**2 / 2 + L**2 * x / 2) / EI,
        Mz=lambda x: q * (x - L) ** 2 / 2,
        Ty=lambda x: q * (L - x),
    )


def simply_supported(L: float, a: float, F: float, EI: float) -> BeamFields:
    """Pinned at x=0, roller at x=L, point force F at x=a."""
    Ra = -F * (L - a) / L
    Rb = -F * a / L
    C1 = -(Ra * L**2 / 6 + F * (L - a) ** 3 / (6 * L))
    return BeamFields(
        uy=lambda x: (Ra * x**3 / 6 + F * _macaulay(x, a, 3) / 6 + C1 * x) / EI,
        rz=lambda x: (Ra * x**2 / 2 + F * _macaulay(x, a, 2) / 2 + C1) / EI,
        Mz=lambda x: Ra * x + F * _macaulay(x, a, 1),
        Ty=lambda x: np.where(x < a, -Ra, Rb),
    )


def simply_supported_distributed(L: float, q: float, EI: float) -> BeamFields:
    """Pinned at x=0, roller at x=L, uniform load q."""
    return BeamFields(
        uy=lambda x: q * x * (L**3 - 2 * L * x**2 + x**3) / (24 * EI),
        rz=lambda x: q * (L**3 - 6 * L * x**2 + 4 * x**3) / (24 * EI),
        Mz=lambda x: q * x * (x - L) / 2,
        Ty=lambda x: q * (L / 2 - x),
    )


def fixed_fixed(L: float, a: float, F: float, EI: float) -> BeamFields:
    """Clamped at both ends, point force F at x=a."""
    b = L - a
    Ra = -F * b**2 * (3 * a + b) / L**3
    Rb = -F * a**2 * (a + 3 * b) / L**3
    Ma = F * a * b**2 / L**2
    return BeamFields(
        uy=lambda x: (Ma * x**2 / 2 + Ra * x**3 / 6 + F * _macaulay(x, a, 3) / 6) / EI,
        rz=lambda x: (Ma * x + Ra * x**2 / 2 + F * _macaulay(x, a, 2) / 2) / EI,
        Mz=lambda x: Ma + Ra * x + F * _macaulay(x, a, 1),
        Ty=lambda x: np.where(x < a, -Ra, Rb),
    )
