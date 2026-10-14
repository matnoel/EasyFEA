# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete


def test_TRI3_spans_the_linears():
    assert_spans(ElemType.TRI3, complete(2, 1))


def test_TRI6_spans_the_quadratics():
    assert_spans(ElemType.TRI6, complete(2, 2))


def test_TRI10_spans_the_cubics():
    assert_spans(ElemType.TRI10, complete(2, 3))


def test_TRI15_spans_the_quartics():
    assert_spans(ElemType.TRI15, complete(2, 4))
