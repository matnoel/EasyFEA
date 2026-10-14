# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import numpy as np


def rel_error(actual, expected, scale: float | None = None) -> float:
    """Largest |actual - expected| over `scale` (default: largest |expected|); pass `scale` where `expected` may vanish. Shapes must match unless `expected` is a scalar."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert (
        expected.ndim == 0 or actual.shape == expected.shape
    ), f"shape {actual.shape} != expected {expected.shape}"
    if scale is None:
        scale = float(np.abs(expected).max())
    return float(np.abs(actual - expected).max() / scale)


FD_TOL = 1e-6
"""Relative tolerance of every finite-difference tangent check."""


def assert_tangent_matches_fd(
    residual: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
    tangent: np.ndarray,
    assembly: np.ndarray,
    eps: float = 1e-6,
) -> None:
    """`tangent` (Ne, ndof, ndof) is ∂residual/∂u by central differences; `residual(u)` gives (Ne, ndof), `assembly` (Ne, ndof) the global dof of each element column."""
    fd = np.zeros_like(tangent, dtype=float)
    for dof in np.unique(assembly):
        du = np.zeros_like(u, dtype=float)
        du[dof] = eps
        diff = (residual(u + du) - residual(u - du)) / (2 * eps)
        for e, j in zip(*np.nonzero(assembly == dof)):
            fd[e, :, j] = diff[e]
    assert rel_error(tangent, fd) < FD_TOL


G3 = np.array([[0.1, 0.2, -0.05], [0.03, -0.1, 0.07], [0.02, 0.05, 0.15]])
"""A full, non-symmetric displacement gradient: no strain component vanishes."""

H3 = np.array([[0.3, -0.1, 0.2], [0.4, 0.1, -0.2], [-0.1, 0.25, 0.05]])
"""A second one, for virtual displacements and velocities."""

R2 = np.sqrt(2)

_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
_KEEP = {1: [0], 2: [0, 1, 5], 3: [0, 1, 2, 3, 4, 5]}


def mandel(A, dim: int | None = None) -> np.ndarray:
    """Kelvin–Mandel `[xx, yy, zz, √2 yz, √2 xz, √2 xy]` of the symmetric part of a (…, n, n) `A`, its `dim` components (`n` by default); written apart from `EasyFEA.FEM` so it can check it."""
    A = np.asarray(A, dtype=float)
    n = A.shape[-1]
    S = np.zeros((*A.shape[:-2], 3, 3))
    S[..., :n, :n] = (A + np.swapaxes(A, -1, -2)) / 2
    vector = np.stack([(1 if i == j else R2) * S[..., i, j] for i, j in _PAIRS], -1)
    return vector[..., _KEEP[dim or n]]


def unmandel(vector) -> np.ndarray:
    """Inverse of `mandel`: a (…, 1), (…, 3) or (…, 6) vector → its symmetric (…, 3, 3), zero out of plane."""
    vector = np.asarray(vector, dtype=float)
    dim = {1: 1, 3: 2, 6: 3}[vector.shape[-1]]
    S = np.zeros((*vector.shape[:-1], 3, 3))
    for k, a in enumerate(_KEEP[dim]):
        i, j = _PAIRS[a]
        S[..., i, j] = S[..., j, i] = vector[..., k] / (1 if i == j else R2)
    return S


def isotropic_stiffness(
    dim: int = 3, lmbda: float = 1.2, mu: float = 0.8
) -> np.ndarray:
    """Kelvin–Mandel λ I⊗I + 2μ 𝕀: (3, 3) in 2D, plane strain, or (6, 6) in 3D."""
    n = 3 if dim == 2 else 6
    C = 2 * mu * np.eye(n)
    C[:dim, :dim] += lmbda
    return C


def affine(coord: np.ndarray, G) -> np.ndarray:
    """Nodal values of x ↦ G x, node by node; a (n,) `G` gives the scalar x ↦ G · x."""
    G = np.asarray(G)
    if G.ndim == 1:
        return coord[:, : G.size] @ G
    return (coord[:, : G.shape[1]] @ G.T).ravel()


def element_dot(groupElem, R_e: np.ndarray, u: np.ndarray) -> float:
    """Σₑ R_e · u_e, `u` gathered on each element: the work of element vectors against a global field."""
    dof_n = R_e.shape[-1] // groupElem.nPe
    return float(np.einsum("ei,ei->", R_e, u[groupElem.Get_assembly_e(dof_n)]))


def quadratic_form(groupElem, K_e: np.ndarray, u: np.ndarray) -> float:
    """Σₑ u_eᵀ K_e u_e, `u` gathered on each element."""
    dof_n = K_e.shape[-1] // groupElem.nPe
    u_e = u[groupElem.Get_assembly_e(dof_n)]
    return float(np.einsum("ei,eij,ej->", u_e, K_e, u_e))


def assert_operator_tangent_matches_fd(
    operator: Callable[[np.ndarray], tuple],
    u: np.ndarray,
    groupElem,
    coefK: float = 1.0,
) -> None:
    """`operator(u)` returns `(K, R, …)`: `coefK · K` is ∂R/∂u at `u`, by `assert_tangent_matches_fd`."""
    K = operator(u)[0]
    assembly = groupElem.Get_assembly_e(K.shape[-1] // groupElem.nPe)
    assert_tangent_matches_fd(lambda u_: operator(u_)[1], u, coefK * K, assembly)
