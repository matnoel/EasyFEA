# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete, tensor


def test_HEXA8_spans_the_trilinears():
    assert_spans(ElemType.HEXA8, tensor(3, 1))


def test_HEXA20_spans_the_quadratic_serendipity_space():
    """Quadratics plus the ten cubic and quartic terms that stay linear in at least two variables."""
    extra = [(2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2)]
    extra += [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]
    assert_spans(ElemType.HEXA20, complete(3, 2) + extra)


def test_HEXA27_spans_the_triquadratics():
    assert_spans(ElemType.HEXA27, tensor(3, 2))
