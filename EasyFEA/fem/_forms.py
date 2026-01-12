# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the BiLinearForm and LinearForm classes used to construct arbitrary fem matrices."""

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
import numpy as np
from scipy.sparse import csr_matrix

# from fem
from ._linalg import FeArray

if TYPE_CHECKING:
    from ._field import Field


class _Form(ABC):
    """Form class from which BilinearForm and LinearForm are derived."""

    def __init__(self, form: Callable[..., FeArray.FeArrayALike]):
        self._form = form

    def __call__(self, *args, **kwds):
        return self._form(*args, **kwds)

    @abstractmethod
    def Integrate_e(self, field: "Field") -> np.ndarray:
        """Integrates de form with the field on elements.

        Parameters
        ----------
        field : Field
            field

        Returns
        -------
        np.ndarray
            the integrated (Ne, ...) numpy array
        """
        pass

    @abstractmethod
    def Assemble(self, field: "Field") -> csr_matrix:
        pass


class BiLinearForm(_Form):
    """Bilinear form."""

    def Integrate_e(self, field):

        # get field data
        dof_n = field.dof_n
        groupElem = field.groupElem
        nPe = groupElem.nPe

        # init data array
        data = np.zeros((groupElem.Ne, nPe * dof_n, nPe * dof_n), dtype=float)

        # get form function
        form = self._form

        # set u and v field
        u = field
        v = field.copy()

        # get dofs and nodes
        dofs = np.arange(nPe * dof_n)
        nodes = np.arange(nPe).reshape(nPe, 1).repeat(dof_n, axis=1).ravel()

        # get dX to integrate
        dX_e_pg = groupElem.Get_weightedJacobian_e_pg(field.matrixType)

        # loop over u dofs
        for i in dofs:

            # activate node and dof for u
            u._Set_current_active_node(nodes[i])
            u._Set_current_active_dof(i % dof_n)

            # loop over v dofs
            for j in dofs:

                # activate node and dof for v
                v._Set_current_active_node(nodes[j])
                v._Set_current_active_dof(j % dof_n)

                # get (Ne, nPg) array
                values_e_pg = form(u, v)

                # sum on gauss points
                values_e = (values_e_pg * dX_e_pg).sum(axis=1)

                # add data
                data[:, i, j] = values_e

        return data

    def Assemble(self, field) -> csr_matrix:
        """Assemble de form with the field.

        Parameters
        ----------
        field : Field
            field

        Returns
        -------
        csr_matrix
            the assembled (Nn * dof_n, Nn * dof_n) sparse matrix.
        """

        # get field data
        dof_n = field.dof_n
        groupElem = field.groupElem

        # get values
        values = self.Integrate_e(field=field).ravel()
        rows = groupElem.Get_rows_e(dof_n).ravel()
        columns = groupElem.Get_columns_e(dof_n).ravel()

        # get shape
        Ndof = groupElem.Nn * dof_n
        shape = (Ndof, Ndof)

        assert values.size == rows.size, f"Not enough data to fill a {shape} matrix."
        matrix = csr_matrix((values.ravel(), (rows, columns)), shape=shape)

        return matrix


class LinearForm(_Form):
    """Linear form."""

    def Integrate_e(self, field):

        # get field data
        dof_n = field.dof_n
        groupElem = field.groupElem
        nPe = groupElem.nPe

        # init data array
        data = np.zeros((groupElem.Ne, nPe * dof_n, 1), dtype=float)

        # get form function
        form = self._form

        # get v field
        v = field

        # get dofs and nodes
        dofs = np.arange(nPe * dof_n)
        nodes = np.arange(nPe).reshape(nPe, 1).repeat(dof_n, axis=1).ravel()

        # get dX to integrate
        dX_e_pg = groupElem.Get_weightedJacobian_e_pg(field.matrixType)

        # loop over u dofs
        for i in dofs:

            # activate node and dof for v
            v._Set_current_active_node(nodes[i])
            v._Set_current_active_dof(i % dof_n)

            # get (Ne, nPg) array
            values_e_pg = form(v)

            # sum on gauss points
            values_e = (values_e_pg * dX_e_pg).sum(axis=1)

            # add data
            data[:, i] = values_e

        return data

    def Assemble(self, field) -> csr_matrix:
        """Assemble de form with the field.

        Parameters
        ----------
        field : Field
            field

        Returns
        -------
        csr_matrix
            the assembled (Nn * dof_n, 1) sparse vector.
        """

        # get field data
        dof_n = field.dof_n
        groupElem = field.groupElem

        # get values
        values = self.Integrate_e(field=field).ravel()
        rows = groupElem.Get_rows_e(dof_n).ravel()
        columns = np.ones_like(rows)

        # get shape
        Ndof = groupElem.Nn * dof_n
        shape = (Ndof, 1)

        assert values.size == rows.size, f"Not enough data to fill a {shape} vector."
        matrix = csr_matrix((values.ravel(), (rows, columns)), shape=shape)

        return matrix
