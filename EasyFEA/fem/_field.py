# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Field class used to construct arbitrary finite element field."""

import copy
import numpy as np
from typing import Callable

# from fem
from . import FeArray, _GroupElem, MatrixType, Mesh
from ..Utilities import _types


class Field:
    """Field class."""

    def __init__(
        self,
        groupElem: _GroupElem,
        dof_n: int,
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
        matrixType : MatrixType, optional
            Determines the number of integration points, by default MatrixType.mass
        """

        assert isinstance(groupElem, _GroupElem)
        self.__groupElem = groupElem

        assert 1 <= dof_n <= groupElem.inDim
        self.__dof_n = dof_n

        self._Set_dofsValues(np.zeros(groupElem.Nn * dof_n))

        self._Set_current_active_node(0)
        self._Set_current_active_dof(0)

        self.__matrixType = matrixType
        self.__is_currently_evaluated = False
        """Indicates that the object is currently being evaluated."""

    @property
    def groupElem(self) -> _GroupElem:
        """Group of elements."""
        return self.__groupElem

    def __Get_groupElem_as_mesh(self) -> Mesh:
        try:
            return self.__mesh
        except AttributeError:
            groupElem = self.groupElem
            self.__mesh = Mesh({groupElem.elemType: groupElem})
            return self.__mesh

    @property
    def dof_n(self) -> int:
        """degrees of freedom per node."""
        return self.__dof_n

    @property
    def matrixType(self) -> MatrixType:
        return self.__matrixType

    def Get_coords(self, concatenate=False):
        """Returns integration point coordinates (x,y,z) for each element."""

        coords = self.groupElem.Get_GaussCoordinates_e_pg(self.__matrixType)
        if concatenate:
            return coords
        else:
            x = FeArray.asfearray(coords[..., 0])
            y = FeArray.asfearray(coords[..., 1])
            z = FeArray.asfearray(coords[..., 2])
            return x, y, z

    def copy(self) -> "Field":
        return copy.deepcopy(self)

    def _Get_current_active_node(self) -> int:
        """Returns current active node."""
        return self.__node

    def _Set_current_active_node(self, node: int):
        """Sets current active node."""
        assert 0 <= node < self.groupElem.nPe
        self.__node = node

    def _Get_current_active_dof(self) -> int:
        """Returns current active dof."""
        return self.__dof

    def _Set_current_active_dof(self, dof: int):
        """Sets current active node."""
        assert 0 <= dof < self.__dof_n
        self.__dof = dof

    def _Get_dofsValues(self) -> _types.FloatArray:
        return self.__dofsValues

    def _Set_dofsValues(self, values: _types.FloatArray):
        Ndof = self.__groupElem.Nn * self.__dof_n
        assert values.ndim == 1 and values.size == Ndof, f"must be a ({Ndof},) array."
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
        node = self._Get_current_active_node()
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

        if self.__is_currently_evaluated:
            return self.groupElem.Get_Gradient_e_pg(
                self._Get_dofsValues(), self.matrixType
            )[..., :dof_n, :dof_n]

        node = self._Get_current_active_node()
        dof = self._Get_current_active_dof()

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

    def Evaluate_e(
        self,
        function: Callable[["Field"], FeArray],
        dofsValues: np.ndarray,
        returnMeanValues: bool = True,
    ) -> np.ndarray:
        """Evaluates the given function for the provided field over the elements.

        Parameters
        ----------
        function : Callable[[Field], FeArray]
            A function that takes a `Field` as input and returns a `FeArray`.
        dofsValues : np.ndarray
            Array of shape (Nn * field.dof_n,) containing the degrees of freedom values.
        returnMeanValues : bool, optional
            If True, returns the mean of the values at each element. Default is True.
        """

        # set dofsValues
        self._Set_dofsValues(np.asarray(dofsValues, dtype=float).ravel())

        self.__is_currently_evaluated = True
        values_e_pg = function(self)
        assert isinstance(values_e_pg, FeArray), "must be a FeArray"
        self.__is_currently_evaluated = False

        if returnMeanValues:
            return values_e_pg.mean(1)
        else:
            return values_e_pg

    def Evaluate_n(
        self, function: Callable[["Field"], FeArray], dofsValues: np.ndarray
    ) -> np.ndarray:
        """Evaluates the given function for the provided field over the nodes.

        Parameters
        ----------
        function : Callable[[Field], FeArray]
            A function that takes a `Field` as input and returns a `FeArray`.
        dofsValues : np.ndarray
            Array of shape (Nn * field.dof_n,) containing the degrees of freedom values.
        """

        values_e = self.Evaluate_e(function, dofsValues, returnMeanValues=True)

        mesh = self.__Get_groupElem_as_mesh()

        return mesh.Get_Node_Values(values_e)

    def Interpolate(self, dofsValues: np.ndarray) -> FeArray:
        """Interpolates degrees of freedom values at each integration point for every element.

        Parameters
        ----------
        dofsValues : np.ndarray
            Array of shape (Nn * dof_n,) containing the degrees of freedom values.

        Returns
        -------
        FeArray
            The (Ne, nPg, dof_n) finite element array.
        """

        # get mesh data
        groupElem = self.groupElem
        Ne = groupElem.Ne
        Nn = groupElem.Nn
        nPe = groupElem.nPe

        # get dof_n
        dofsValues = np.asarray(dofsValues).ravel()
        assert dofsValues.size % Nn == 0, "Must be a (Nn * dof_n) array"
        dof_n = dofsValues.size // Nn

        # get (Ne, nPe, dof_n) values
        dofsValues_e = groupElem.Locates_sol_e(dofsValues, dof_n).reshape(
            Ne, nPe, dof_n
        )

        # get (Ne, nPg, dof_n) interpolated values
        N_pg = groupElem.Get_N_pg(self.matrixType)
        values_e_pg = np.einsum("end,pin->epd", dofsValues_e, N_pg, optimize="optimal")

        return FeArray.asfearray(values_e_pg)


def Sym_Grad(u: Field) -> FeArray.FeArrayALike:
    grad = u.grad
    return 0.5 * (grad.T + grad)
