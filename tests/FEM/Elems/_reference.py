# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The polynomial space each element must reproduce, and the check that its shape functions do.

A monomial is its exponents: `(2, 1)` is ξ²η, `(0, 0, 0)` the constant 1.
"""

from itertools import product
from math import factorial

import numpy as np

from EasyFEA.FEM import ElemType, MatrixType

from .._reference import reference_element


def complete(dim: int, degree: int) -> list[tuple]:
    """Every monomial of total degree ≤ `degree`: triangles, tetrahedra, segments."""
    return [p for p in product(range(degree + 1), repeat=dim) if sum(p) <= degree]


def tensor(dim: int, degree: int) -> list[tuple]:
    """Every monomial of degree ≤ `degree` in each variable: quadrangles, hexahedra."""
    return list(product(range(degree + 1), repeat=dim))


def _monomial(xi: np.ndarray, powers: tuple, d: int = 0, k: int = 0) -> np.ndarray:
    """∂ᵏ/∂ξ_dᵏ of the monomial at the points `xi` (n, dim); `k = 0` is the monomial itself."""
    if powers[d] < k:
        return np.zeros(xi.shape[0])
    coef = factorial(powers[d]) / factorial(powers[d] - k)
    lowered = list(powers)
    lowered[d] -= k
    return coef * np.prod(xi ** np.array(lowered), axis=1)


def _power(base: str, p: int) -> str:
    """`("η", 2)` → `η²`; nothing for `p = 0`, no exponent for `p = 1`."""
    if p <= 1:
        return base * p
    return base + str(p).translate(str.maketrans("23456789", "²³⁴⁵⁶⁷⁸⁹"))


def _label(powers: tuple) -> str:
    """`(2, 1)` → `ξ²η`."""
    return "".join(_power(v, p) for v, p in zip("ξηζ", powers)) or "1"


def _derivative_label(d: int, k: int) -> str:
    """`(1, 2)` → `∂²/∂η²`, `k = 0` → `value`."""
    return f"∂{_power('', k)}/∂{_power('ξηζ'[d], k)}" if k else "value"


def assert_spans(elemType: ElemType, space: list[tuple]) -> None:
    """The shape functions of `elemType` span exactly `space`.

    Interpolating a monomial p from its nodal values, Σᵢ p(ξᵢ) Nᵢ, must give p back, and
    Σᵢ p(ξᵢ) ∂ᵏNᵢ/∂ξ_dᵏ its derivatives, k ≤ 4, at every Gauss point. With as many
    monomials as nodes, no other basis passes: one wrong term of N or dN fails. Partition
    of unity, zero-sum derivatives and Nᵢ(ξⱼ) = δᵢⱼ all follow.
    """
    groupElem = reference_element(elemType)
    assert len(space) == groupElem.nPe, f"{len(space)} monomials, {groupElem.nPe} nodes"

    matrixType = MatrixType.mass
    xi_n = groupElem.Get_Local_Coords()
    xi_pg = groupElem.Get_gauss(matrixType).coord
    # k-th pure derivatives of the shape functions at the Gauss points, (nPg, dim, nPe); k = 0 has dim 1
    derivatives = [
        groupElem.Get_N_pg(matrixType),
        groupElem.Get_dN_pg(matrixType),
        groupElem.Get_ddN_pg(matrixType),
        groupElem.Get_dddN_pg(matrixType),
        groupElem.Get_ddddN_pg(matrixType),
    ]

    for powers in space:
        p_n = _monomial(xi_n, powers)
        for k, D_pg in enumerate(derivatives):
            interpolated = D_pg @ p_n
            for d in range(interpolated.shape[1]):
                error = np.abs(
                    interpolated[:, d] - _monomial(xi_pg, powers, d, k)
                ).max()
                where = f"{_label(powers)}, {_derivative_label(d, k)}"
                assert error < 1e-10, f"{where} off by {error:.0e}"
