# Copyright (C) 2021-2024 Université Gustave Eiffel. All rights reserved.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from enum import Enum

class ElemType(str, Enum):
    """Implemented element types"""

    POINT = "POINT"
    SEG2 = "SEG2"
    SEG3 = "SEG3"
    SEG4 = "SEG4"
    # SEG5 = "SEG5"
    TRI3 = "TRI3"
    TRI6 = "TRI6"
    TRI10 = "TRI10"
    # TRI15 = "TRI15"
    QUAD4 = "QUAD4"
    QUAD8 = "QUAD8"
    # QUAD9 = "QUAD9"
    TETRA4 = "TETRA4"
    TETRA10 = "TETRA10"
    HEXA8 = "HEXA8"
    HEXA20 = "HEXA20"
    PRISM6 = "PRISM6"
    PRISM15 = "PRISM15"
    # PRISM18 = "PRISM18"
    # PYRA5 = "PYRA5"
    # PYRA13 = "PYRA13"
    # PYRA14 = "PYRA14"

    def __str__(self) -> str:
        return self.name
    
    @staticmethod
    def Get_1D() -> list[str]:
        """1D element types."""        
        liste1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4]
        return liste1D
    
    @staticmethod
    def Get_2D() -> list[str]:
        """2D element types."""
        liste2D = [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10, ElemType.QUAD4, ElemType.QUAD8]
        return liste2D
    
    @staticmethod
    def Get_3D() -> list[str]:
        """3D element types."""
        liste3D = [ElemType.TETRA4, ElemType.TETRA10, ElemType.HEXA8, ElemType.HEXA20, ElemType.PRISM6, ElemType.PRISM15]
        return liste3D

class MatrixType(str, Enum):
    """Order used for integration over elements (determines the number of integration points)."""

    rigi = "rigi"
    """dN*dN type"""
    mass = "mass"
    """N*N type"""
    beam = "beam"
    """ddNv*ddNv type"""

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def Get_types() -> list[str]:
        return [MatrixType.rigi, MatrixType.mass, MatrixType.beam]