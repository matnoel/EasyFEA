# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.FEM import ElemType, MatrixType


def test_ElemType_prints_its_name():
    assert str(ElemType.HEXA27) == "HEXA27"
    assert ElemType("TRI3") is ElemType.TRI3


def test_ElemType_topology_drops_the_node_count(subtests):
    for elemType, topology in [
        (ElemType.POINT, "POINT"),
        (ElemType.SEG5, "SEG"),
        (ElemType.TRI15, "TRI"),
        (ElemType.QUAD9, "QUAD"),
        (ElemType.TETRA10, "TETRA"),
        (ElemType.HEXA27, "HEXA"),
        (ElemType.PRISM18, "PRISM"),
    ]:
        with subtests.test(elemType=str(elemType)):
            assert elemType.topology == topology


def test_ElemType_Get_1D_lists_the_segments():
    assert {e.topology for e in ElemType.Get_1D()} == {"SEG"}


def test_ElemType_Get_2D_lists_the_triangles_and_quadrangles():
    assert {e.topology for e in ElemType.Get_2D()} == {"TRI", "QUAD"}


def test_ElemType_Get_3D_lists_the_volumes():
    assert {e.topology for e in ElemType.Get_3D()} == {"TETRA", "HEXA", "PRISM"}


def test_ElemType_Get_1D_2D_3D_and_POINT_cover_every_type_once():
    listed = (
        [ElemType.POINT] + ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D()
    )

    assert sorted(listed) == sorted(ElemType)


def test_MatrixType_prints_its_name():
    assert str(MatrixType.beam_shear) == "beam_shear"


def test_MatrixType_Get_types_lists_every_type():
    assert MatrixType.Get_types() == list(MatrixType)


def test_MatrixType_Order_is_explicit_whatever_the_element():
    order = MatrixType.Order(5)

    assert order._Get_order(1, "TRI") == order._Get_order(2, "HEXA") == 5


def test_MatrixType__Get_order_is_the_degree_of_the_integrand(subtests):
    for matrixType, order, topology, expected in [
        (MatrixType.mass, 2, "TRI", 4),
        (MatrixType.mass, 1, "QUAD", 2),
        (MatrixType.rigi, 2, "TRI", 2),
        (MatrixType.rigi, 2, "TETRA", 2),
        (MatrixType.rigi, 2, "SEG", 2),
        (MatrixType.rigi, 2, "QUAD", 4),
        (MatrixType.rigi, 1, "HEXA", 2),
        (MatrixType.rigi, 1, "PRISM", 2),
        (MatrixType.beam, 2, "SEG", 6),
        (MatrixType.beam_shear, 2, "SEG", 2),
    ]:
        with subtests.test(f"{matrixType} P{order} on {topology}"):
            assert matrixType._Get_order(order, topology) == expected


def test_MatrixType__Get_order_of_a_beam_type_needs_a_segment(subtests):
    for matrixType in (MatrixType.beam, MatrixType.beam_shear):
        with subtests.test(str(matrixType)):
            with pytest.raises(ValueError, match="segments"):
                matrixType._Get_order(1, "TRI")
