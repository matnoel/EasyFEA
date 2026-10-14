# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA.FEM import ElemType

from ._reference import assert_spans, complete


def test_TETRA4_spans_the_linears():
    assert_spans(ElemType.TETRA4, complete(3, 1))


def test_TETRA10_spans_the_quadratics():
    assert_spans(ElemType.TETRA10, complete(3, 2))
