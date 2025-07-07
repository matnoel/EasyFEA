# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Field class used to construct arbitrary finite element field."""

import copy

# from fem
from . import FeArray, _GroupElem, MatrixType


class Field:
    """Field class."""

    def __init__(self, groupElem: _GroupElem, dof_n: int, thickness: float = 1.0):
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
        """

        assert isinstance(groupElem, _GroupElem)
        self.__groupElem = groupElem

        assert 1 <= dof_n <= groupElem.dim
        self.__dof_n = dof_n

        if groupElem.dim == 2:
            assert thickness > 0, "Must be greater than 0"
            self.__thickness = thickness

        self._Set_node(0)

    @property
    def groupElem(self) -> _GroupElem:
        """Group of elements."""
        return self.__groupElem

    @property
    def dof_n(self) -> int:
        """degrees of freedom per node."""
        return self.__dof_n

    def copy(self) -> "Field":
        return copy.deepcopy(self)

    def _Get_node(self) -> int:
        """Returns current active node."""
        return self.__node

    def _Set_node(self, node: int):
        """Sets current active node."""
        assert 0 <= node < self.groupElem.nPe
        self.__node = node

    def __call__(self):
        """Returns the field as a finite element array."""
        node = self._Get_node()
        N_pg = self.groupElem.Get_N_pg(MatrixType.mass)
        nPg, dim, _ = N_pg.shape
        array = FeArray.asfearray(N_pg[..., node].reshape(1, nPg, 1))
        return array

    @property
    def grad(self):
        """Returns the gradient of the field."""
        node = self._Get_node()
        dN_e_pg = self.groupElem.Get_dN_e_pg(MatrixType.rigi)
        array = FeArray.asfearray(dN_e_pg[..., node])
        return array
