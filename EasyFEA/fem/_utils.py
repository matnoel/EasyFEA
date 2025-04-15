# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from enum import Enum
import numpy as np

class ElemType(str, Enum):
    """Implemented element types."""

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
    
    @staticmethod
    def Get_1D() -> list[str]:
        """Returns 1D element types."""        
        elems_1D = [ElemType.SEG2, ElemType.SEG3, ElemType.SEG4, ElemType.SEG5]
        return elems_1D
    
    @staticmethod
    def Get_2D() -> list[str]:
        """Returns 2D element types."""
        elems_2D = [ElemType.TRI3, ElemType.TRI6, ElemType.TRI10, ElemType.TRI15,
                    ElemType.QUAD4, ElemType.QUAD8, ElemType.QUAD9]
        return elems_2D
    
    @staticmethod
    def Get_3D() -> list[str]:
        """Returns 3D element types."""
        elems_3D = [ElemType.TETRA4, ElemType.TETRA10,
                    ElemType.HEXA8, ElemType.HEXA20, ElemType.HEXA27,
                    ElemType.PRISM6, ElemType.PRISM15, ElemType.PRISM18]
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
    def Get_types() -> list[str]:
        return [MatrixType.rigi, MatrixType.mass, MatrixType.beam]

class FeArray(np.ndarray):
    """Finite Element array"""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.ndim not in [2, 3, 4, 6]:
            raise ValueError("The input array dimensions must be one of the following: 2, 3, 4, or 6.")
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        # This method is automatically called when new instances are created.
        # It can be used to initialize additional attributes if necessary.
        if obj is None:
            return

    @property        
    def Ne(self) -> int:
        return self.shape[0]

    @property
    def nPg(self) -> int:
        return self.shape[1]
    
    @property
    def _shape(self) -> tuple:
        """finite element shape"""
        return self.shape[2:]
    
    @property
    def _ndim(self) -> int:
        """finite element ndim"""
        return self.ndim - 2

    @property
    def _idx(self) -> str:
        """einsum indicator (e.g "ep", "epi", "epij") used in `np.einsum()` function.\n
        see https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        if self._ndim == 0:
            return "ep"
        elif self._ndim == 1:
            return "epi"
        elif self._ndim == 2:
            return "epij"
        elif self._ndim == 4:
            return "epijkl"
    
    @property
    def _type(self) -> str:
        if self._ndim == 0:
            return "scalar"
        elif self._ndim == 1:
            return "vector"
        elif self._ndim == 2:
            return "matrix"
        elif self._ndim == 4:
            return "tensor"

    def __get_array1_array2(self, other) -> tuple[np.ndarray, np.ndarray]:

        array1 = np.asarray(self)
        ndim1 = self._ndim
        shape1 = self._shape

        array2 = np.asarray(other)
        if isinstance(other, FeArray):
            if self.shape[:2] != other.shape[:2]:
                raise ValueError(f"The FeArray `other` must be defined on {self.Ne} elements and {self.nPg} gauss points.")
            ndim2 = other._ndim
            shape2 = other._shape
        else:
            ndim2 = array2.ndim
            shape2 = array2.shape

        if ndim1 == 0:
            # array1(Ne, nPg)  array2(...) => (Ne, nPg, ...)
            # or
            # array1(Ne, nPg)  array2(Ne, nPg, ...) => (Ne, nPg, ...)
            new_shape = (self.Ne, self.nPg, *[1]*ndim2)
            array1 = array1.reshape(new_shape)
        elif shape1 == shape2:
            pass
        else:
            type_str = "FeArray" if isinstance(other, FeArray) else "array"
            raise ValueError(
                f"The {type_str} `other` with shape {shape2} must be either a {self.shape} array or a {shape1} FeArray."
                )
        
        return array1, array2

    def __add__(self, other):
        # Overload the + operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 + array2
        return FeArray(result)

    def __sub__(self, other):
        # Overload the - operator
        return self.__add__(-other)

    def __mul__(self, other):
        # Overload the * operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 * array2
        return FeArray(result)

    def __truediv__(self, other):
        # Overload the / operator
        return self.__mul__(1/other)