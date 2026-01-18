# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from enum import Enum


class ElemType(str, Enum):
    """Implemented Lagrange isoparametric element types."""

    POINT = "POINT"
    SEG2 = "SEG2"
    SEG3 = "SEG3"
    SEG4 = "SEG4"
    SEG5 = "SEG5"
    TRI3 = "TRI3"
    TRI6 = "TRI6"
    TRI10 = "TRI10"
    TRI15 = "TRI15"
    QUAD4 = "QUAD4"
    QUAD8 = "QUAD8"
    QUAD9 = "QUAD9"
    TETRA4 = "TETRA4"
    TETRA10 = "TETRA10"
    HEXA8 = "HEXA8"
    HEXA20 = "HEXA20"
    HEXA27 = "HEXA27"
    PRISM6 = "PRISM6"
    PRISM15 = "PRISM15"
    PRISM18 = "PRISM18"
    # PYRA5 = "PYRA5"
    # PYRA13 = "PYRA13"
    # PYRA14 = "PYRA14"

    def __str__(self) -> str:
        return self.name

    @property
    def topology(self) -> str:
        return "".join([s for s in self.name if not s.isdigit()])

    @staticmethod
    def Get_1D() -> list["ElemType"]:
        """Returns 1D element types."""
        elems_1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4, ElemType.SEG5]
        return elems_1D

    @staticmethod
    def Get_2D() -> list["ElemType"]:
        """Returns 2D element types."""
        elems_2D = [
            ElemType.TRI3,
            ElemType.TRI6,
            ElemType.TRI10,
            ElemType.TRI15,
            ElemType.QUAD4,
            ElemType.QUAD8,
            ElemType.QUAD9,
        ]
        return elems_2D

    @staticmethod
    def Get_3D() -> list["ElemType"]:
        """Returns 3D element types."""
        elems_3D = [
            ElemType.TETRA4,
            ElemType.TETRA10,
            ElemType.HEXA8,
            ElemType.HEXA20,
            ElemType.HEXA27,
            ElemType.PRISM6,
            ElemType.PRISM15,
            ElemType.PRISM18,
        ]
        return elems_3D


class MatrixType(str, Enum):
    """Order used for integration over elements, which determines the number of integration points."""

    rigi = "rigi"
    """int_Ω dN • dN dΩ type"""
    mass = "mass"
    """int_Ω N • N dΩ type"""
    beam = "beam"
    """int_Ω ddNv • ddNv dΩ type"""

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def Get_types() -> list["MatrixType"]:
        return [MatrixType.rigi, MatrixType.mass, MatrixType.beam]
