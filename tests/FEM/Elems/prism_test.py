# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete


def _wedge(degree: int) -> list[tuple]:
    """Triangle polynomials of total degree ≤ `degree` in (ξ, η) times degree ≤ `degree` in ζ."""
    return [(a, b, c) for a, b in complete(2, degree) for c in range(degree + 1)]


def test_PRISM6_spans_the_linear_wedge_space():
    assert_spans(ElemType.PRISM6, _wedge(1))


def test_PRISM15_spans_the_quadratic_serendipity_wedge_space():
    """Quadratics plus ξ²ζ, ξηζ, η²ζ, ξζ² and ηζ²."""
    extra = [(2, 0, 1), (1, 1, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2)]
    assert_spans(ElemType.PRISM15, complete(3, 2) + extra)


def test_PRISM18_spans_the_quadratic_wedge_space():
    assert_spans(ElemType.PRISM18, _wedge(2))
