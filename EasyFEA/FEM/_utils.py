# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from dataclasses import dataclass
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
    """int_Ω dN • dN dΩ type, order 2(k-1), 2k per direction on QUAD, HEXA and PRISM"""
    mass = "mass"
    """int_Ω N • N dΩ type, order 2k"""
    beam = "beam"
    """int_Ω ddNv • ddNv dΩ type, order 2(2k-1) (Hermite of degree 2k+1), segments only"""
    beam_shear = "beam_shear"
    """Reduced integration order for the Timoshenko shear term (n-1 Gauss points for SEGn). Selective reduced integration is the standard cure for shear locking in Lagrange Timoshenko beams (Hughes 1987; Bathe 2006)."""

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def Get_types() -> list["MatrixType"]:
        return [
            MatrixType.rigi,
            MatrixType.mass,
            MatrixType.beam,
            MatrixType.beam_shear,
        ]

    @staticmethod
    def Order(n: int) -> "_Order":
        """Explicit integration order n (per direction on QUAD, HEXA and PRISM), in place of a matrix type."""
        return _Order(n)

    def _Get_order(self, order: int, topology: str) -> int:
        """Integration order on an element of `order` and `topology`."""
        if self == MatrixType.mass:
            return 2 * order
        if self == MatrixType.rigi:
            # a derivative lowers the degree only along its own direction of a tensor basis
            isTensor = topology in ("QUAD", "HEXA", "PRISM")
            return 2 * order if isTensor else 2 * (order - 1)
        if topology != "SEG":
            raise ValueError(f"{self} is only defined on segments.")
        if self == MatrixType.beam:
            return 2 * (2 * order - 1)
        return 2 * (order - 1)


@dataclass(frozen=True)
class _Order:
    """An explicit integration order, from `MatrixType.Order`."""

    order: int

    def _Get_order(self, order: int, topology: str) -> int:
        """The explicit order, whatever the element."""
        return self.order
