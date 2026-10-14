# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

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
