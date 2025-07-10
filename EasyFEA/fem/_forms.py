# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the BiLinearForm and LinearForm classes used to construct arbitrary fem matrices."""

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING
from scipy import sparse
import numpy as np

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
    def _assemble(self, field: "Field") -> sparse.csr_matrix:
        """Assemble de form with the field.

        Parameters
        ----------
        field : Field
            field

        Returns
        -------
        sparse.csr_matrix
            assembled sparce matrix
        """
        pass


class BiLinearForm(_Form):

    def _assemble(self, field):

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
            u._Set_node(nodes[i])
            u._Set_dof(i % dof_n)

            # loop over v dofs
            for j in dofs:

                # activate node and dof for v
                v._Set_node(nodes[j])
                v._Set_dof(j % dof_n)

                # get (Ne, nPg) array
                values_e_pg = form(u, v)

                # sum on gauss points
                values_e = (values_e_pg * dX_e_pg).sum(axis=1)

                # add data
                data[:, i, j] = values_e

        # construct sparse matrix
        Ndof = groupElem.Nn * dof_n
        rows = groupElem.Get_rowsVector_e(dof_n)
        columns = groupElem.Get_columnsVector_e(dof_n)
        matrix = sparse.csr_matrix(
            (data.ravel(), (rows.ravel(), columns.ravel())), (Ndof, Ndof), dtype=float
        )

        return matrix


class LinearForm(_Form):

    def _assemble(self, field):

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
            v._Set_node(nodes[i])
            v._Set_dof(i % dof_n)

            # get (Ne, nPg) array
            values_e_pg = form(v)

            # sum on gauss points
            values_e = (values_e_pg * dX_e_pg).sum(axis=1)

            # add data
            data[:, i] = values_e

        # construct sparse matrix
        Ndof = groupElem.Nn * dof_n
        rows = groupElem.Get_assembly_e(dof_n)
        columns = np.zeros_like(rows)
        matrix = sparse.csr_matrix(
            (data.ravel(), (rows.ravel(), columns.ravel())), (Ndof, 1), dtype=float
        )

        return matrix
