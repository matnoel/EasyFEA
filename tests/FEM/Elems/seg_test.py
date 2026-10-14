# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete


def test_SEG2_spans_the_linears():
    assert_spans(ElemType.SEG2, complete(1, 1))


def test_SEG3_spans_the_quadratics():
    assert_spans(ElemType.SEG3, complete(1, 2))


def test_SEG4_spans_the_cubics():
    assert_spans(ElemType.SEG4, complete(1, 3))


def test_SEG5_spans_the_quartics():
    assert_spans(ElemType.SEG5, complete(1, 4))
