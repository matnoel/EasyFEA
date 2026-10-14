# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete, tensor


def test_QUAD4_spans_the_bilinears():
    assert_spans(ElemType.QUAD4, tensor(2, 1))


def test_QUAD8_spans_the_quadratic_serendipity_space():
    """Quadratics plus ξ²η and ξη²."""
    assert_spans(ElemType.QUAD8, complete(2, 2) + [(2, 1), (1, 2)])


def test_QUAD9_spans_the_biquadratics():
    assert_spans(ElemType.QUAD9, tensor(2, 2))
