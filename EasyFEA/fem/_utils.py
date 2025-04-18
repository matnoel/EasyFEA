# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from enum import Enum
import numpy as np
from typing import Union

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
    """Finite Element array.\n
    A finite element array has at least two dimensions.
    """

    def __new__(cls, input_array, broadcastFeArrays=False):
        obj = np.asarray(input_array).view(cls)
        if broadcastFeArrays:
            obj = obj[np.newaxis, np.newaxis]
        if obj.ndim < 2:
            raise ValueError("The input array must have at least 2 dimensions.")
        return obj

    def __array_finalize__(self, obj: np.ndarray):
        # This method is automatically called when new instances are created.
        # It can be used to initialize additional attributes if necessary.
        if obj is None:
            return
        
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
        """einsum indicator (e.g "", "i", "ij") used in `np.einsum()` function.\n
        see https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        """
        if self._ndim == 0:
            return ""
        elif self._ndim == 1:
            return "i"
        elif self._ndim == 2:
            return "ij"
        elif self._ndim == 4:
            return "ijkl"
    
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
            ndim2 = other._ndim
            shape2 = other._shape
        else:
            ndim2 = array2.ndim
            shape2 = array2.shape

        if ndim1 == 0:
            # array1(Ne, nPg)  array2(...) => (Ne, nPg, ...)
            # or
            # array1(Ne, nPg)  array2(Ne, nPg, ...) => (Ne, nPg, ...)            
            for _ in range(ndim2):
                array1 = array1[...,np.newaxis]
        elif ndim2 == 0:
            if array2.size == 1:
                # array1(Ne, nPg, ...)  array2() => (Ne, nPg, ...)
                pass
            else:
                # array1(Ne, nPg, ...)  array2(Ne, nPg) => (Ne, nPg, ...)
                for _ in range(ndim1):
                    array2 = array2[...,np.newaxis]
        elif shape1 == shape2:
            pass
        else:
            type_str = "FeArray" if isinstance(other, FeArray) else "np.array"
            raise ValueError(f"The {type_str} `other` with shape {shape2} must be either a {self.shape} np.array or a {shape1} FeArray.")
        
        return array1, array2

    def __add__(self, other) -> Union['FeArray', np.ndarray]:
        # Overload the + operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 + array2
        return FeArray.asfearray(result)

    def __sub__(self, other) -> Union['FeArray', np.ndarray]:
        # Overload the - operator
        return self.__add__(-other)

    def __mul__(self, other) -> Union['FeArray', np.ndarray]:
        # Overload the * operator
        array1, array2 = self.__get_array1_array2(other)
        result = array1 * array2
        return FeArray.asfearray(result)

    def __truediv__(self, other) -> Union['FeArray', np.ndarray]:
        # Overload the / operator
        return self.__mul__(1/other)

    @property 
    def T(self) -> 'FeArray':
        if self._ndim >= 2:
            idx = self._idx
            subscripts = f"...{idx}->...{idx[::-1]}"
            result = np.einsum(subscripts, np.asarray(self), optimize="optimal")
            return FeArray.asfearray(result)
        else:
            return self.copy()
        
    def __matmul__(self, other) -> Union['FeArray', np.ndarray]:

        ndim1 = self._ndim

        if isinstance(other, FeArray):
            ndim2 = other._ndim
        elif isinstance(other, np.ndarray):
            ndim2 = other.ndim
        else:
            raise TypeError("`other` must be either a FeArray or np.ndarray")
        
        if ndim1 == ndim2 == 1:
            result = np.vecdot(self, other)
        elif ndim1 == ndim2 == 2:
            result = super().__matmul__(other)
        elif ndim1 == 1 and ndim2 == 2:
            result = (self[:,:,np.newaxis,:] @ other)[:,:,0,:]
        elif ndim1 == 2 and ndim2 == 1:
            result = (self @ other[:,:,:,np.newaxis])[:,:,:,0]
        else:
            result = self.dot(other)

        return FeArray.asfearray(result)
        
    def dot(self, other) -> Union['FeArray', np.ndarray]:
        
        ndim1 = self._ndim
        if ndim1 == 0:
            raise ValueError("Must be at least a finite element vector (Ne, nPg, i).")
        
        idx1 = self._idx    

        if isinstance(other, FeArray):
            idx2 = other._idx
            ndim2 = other._ndim            
        elif isinstance(other, np.ndarray):
            idx2 = "".join([chr(ord(idx1[0])+i) for i in range(other.ndim)])
            ndim2 = other.ndim
        else:
            raise TypeError("`other` must be either a FeArray or np.ndarray")
        idx2 = "".join([chr(ord(val)+ndim1-1) for val in idx2])

        if ndim2 == 0:
            raise ValueError("`other` must be at least a finite element vector (Ne, nPg, i).") 
        
        end = str(idx1+idx2).replace(idx1[-1],"")
        subscripts = f"...{idx1},...{idx2}->...{end}"

        result = np.einsum(subscripts, self, other)
        
        return FeArray.asfearray(result)

    def ddot(self, other) -> Union['FeArray', np.ndarray]:
        
        ndim1 = self._ndim
        if ndim1 < 2:
            raise ValueError("Must be at least a finite element matrix (Ne, nPg, i, j).")
        
        idx1 = self._idx

        if isinstance(other, FeArray):
            idx2 = other._idx
            ndim2 = other._ndim            
        elif isinstance(other, np.ndarray):
            idx2 = "".join([chr(ord(idx1[0])+i) for i in range(other.ndim)])
            ndim2 = other.ndim
        else:
            raise TypeError("`other` must be either a FeArray or np.ndarray")
        if ndim2 < 2:
            raise ValueError("`other` must be at least a finite element matrix (Ne, nPg, i, j).") 
        idx2 = "".join([chr(ord(val)+ndim1-2) for val in idx2])
        
        end = str(idx1+idx2).replace(idx1[-1],"")
        end = end.replace(idx1[-2],"")
        subscripts = f"...{idx1},...{idx2}->...{end}"

        result = np.einsum(subscripts, self, other)
        
        return FeArray.asfearray(result)
    
    def sum(self, *args, **kwargs) -> Union['FeArray', np.ndarray]:
        """`np.sum()` wrapper."""
        return FeArray.asfearray(super().sum(*args, **kwargs))
    
    def max(self, *args, **kwargs) -> Union['FeArray', np.ndarray]:
        """`np.max()` wrapper."""
        return FeArray.asfearray(super().max(*args, **kwargs))
    
    def min(self, *args, **kwargs) -> Union['FeArray', np.ndarray]:
        """`np.min()` wrapper."""
        return FeArray.asfearray(super().min(*args, **kwargs))
    
    def _get_idx(self, *arrays) -> list[np.ndarray]:

        ndim = len(arrays) + 2

        Ne, nPg = self.shape[:2]

        def get_shape(i: int, array: np.ndarray):
            shape = np.ones(ndim, dtype=int)
            shape[i] = array.size
            return np.reshape(array, shape)
        
        idx = [get_shape(i, val) for i, val in enumerate([np.arange(Ne), np.arange(nPg), *arrays])]

        return idx
    
    def _assemble(self, *arrays, value: Union['FeArray', np.ndarray]):

        idx = self._get_idx(*arrays)

        self[*idx] = value

    @staticmethod
    def asfearray(array, broadcastFeArrays=False) -> Union['FeArray', np.ndarray]:
        array = np.asarray(array)
        if broadcastFeArrays:
            return FeArray(array, broadcastFeArrays=broadcastFeArrays)
        elif array.ndim >= 2:
            return FeArray(array)
        else:
            return array
    
    def _asfearrays(*arrays, broadcastFeArrays=False) -> list[Union['FeArray', np.ndarray]]:
        return [FeArray.asfearray(array, broadcastFeArrays=broadcastFeArrays) for array in arrays]

    @staticmethod
    def zeros(*shape, dtype=None) -> Union['FeArray', np.ndarray]:
        return FeArray.asfearray(np.zeros(shape=shape, dtype=dtype))

    @staticmethod
    def ones(*shape, dtype=None) -> Union['FeArray', np.ndarray]:
        return FeArray.asfearray(np.ones(shape=shape, dtype=dtype))