# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Field class used to construct arbitrary finite element field."""

import copy
import numpy as np

# from fem
from . import FeArray, _GroupElem, MatrixType
from ..utilities import _types


class Field:
    """Field class."""

    def __init__(
        self,
        groupElem: _GroupElem,
        dof_n: int,
        thickness: float = 1.0,
        matrixType: MatrixType = MatrixType.mass,
    ):
        """
        Initialize a new field.

        Parameters
        ----------
        groupElem : _GroupElem
            The group of elements to associate with this instance.
        dof_n : int
            Degree of freedom number, must be between 1 and the dimension of the groupElem.
        thickness : float, optional
            thickness used in the model, by default 1.0
        matrixType : MatrixType, optional
            Determines the number of integration points, by default MatrixType.mass
        """

        assert isinstance(groupElem, _GroupElem)
        self.__groupElem = groupElem

        assert 1 <= dof_n <= groupElem.inDim
        self.__dof_n = dof_n

        self._Set_dofsValues(np.zeros(groupElem.Nn * dof_n))

        self._Set_node(0)
        self._Set_dof(0)

        self.__matrixType = matrixType

    @property
    def groupElem(self) -> _GroupElem:
        """Group of elements."""
        return self.__groupElem

    @property
    def dof_n(self) -> int:
        """degrees of freedom per node."""
        return self.__dof_n

    @property
    def matrixType(self) -> MatrixType:
        return self.__matrixType

    def Get_coords(
        self,
    ) -> tuple[_types.FloatArray, _types.FloatArray, _types.FloatArray]:
        """ "Returns integration point coordinates (x,y,z) for each element."""
        coords = self.groupElem.Get_GaussCoordinates_e_pg(self.__matrixType)
        x = FeArray.asfearray(coords[..., 0])
        y = FeArray.asfearray(coords[..., 1])
        z = FeArray.asfearray(coords[..., 2])
        return x, y, z

    def copy(self) -> "Field":
        return copy.deepcopy(self)

    def _Get_node(self) -> int:
        """Returns current active node."""
        return self.__node

    def _Set_node(self, node: int):
        """Sets current active node."""
        assert 0 <= node < self.groupElem.nPe
        self.__node = node

    def _Get_dof(self) -> int:
        """Returns current active dof."""
        return self.__dof

    def _Set_dof(self, dof: int):
        """Sets current active node."""
        assert 0 <= dof < self.__dof_n
        self.__dof = dof

    def _Get_dofsValues(self) -> _types.FloatArray:
        return self.__dofsValues

    def _Set_dofsValues(self, values: _types.FloatArray):
        Ndof = self.__groupElem.Nn * self.__dof_n
        assert values.ndim == 1 and values.size == Ndof, f"must be a {Ndof} array."
        self.__dofsValues = values

    def __mul__(self, other) -> FeArray.FeArrayALike:
        return self() * other

    def __rmul__(self, other) -> FeArray.FeArrayALike:
        return self.__mul__(other)

    def __matmul__(self, other) -> FeArray.FeArrayALike:
        return self() @ other

    def __rmatmul__(self, other) -> FeArray.FeArrayALike:
        return self.__matmul__(other)

    def __add__(self, other) -> FeArray.FeArrayALike:
        return self() + other

    def __radd__(self, other) -> FeArray.FeArrayALike:
        return self.__add__(other)

    def __sub__(self, other) -> FeArray.FeArrayALike:
        return self() - other

    def __rsub__(self, other) -> FeArray.FeArrayALike:
        return self.__sub__(other)

    def __truediv__(self, other) -> FeArray.FeArrayALike:
        return self() / other

    def __rtruediv__(self, other) -> FeArray.FeArrayALike:
        return self.__truediv__(other)

    def __call__(self) -> FeArray.FeArrayALike:
        """Returns the field as a finite element array."""
        node = self._Get_node()
        N_pg = self.groupElem.Get_N_pg(self.__matrixType)
        nPg, _, _ = N_pg.shape
        array = FeArray.asfearray(N_pg[..., node].reshape(1, nPg, 1))
        return array

    def dot(self, other):
        return self().dot(other)

    def ddot(self, other):
        return self().ddot(other)

    @property
    def grad(self) -> FeArray.FeArrayALike:
        """Returns the gradient of the field."""
        dof_n = self.__dof_n
        node = self._Get_node()
        dof = self._Get_dof()

        # get gem matrices
        dN_e_pg = self.groupElem.Get_dN_e_pg(self.__matrixType)
        Ne, nPg, dim, _ = dN_e_pg.shape

        array = FeArray.asfearray(dN_e_pg[..., node])

        if dof_n == 1:
            return array
        else:
            newArray = FeArray.zeros(Ne, nPg, dim, dof_n, dtype=float)
            newArray[..., :, dof] = array
            return newArray


def Sym_Grad(u: Field) -> FeArray.FeArrayALike:
    return 0.5 * (u.grad.T + u.grad)
