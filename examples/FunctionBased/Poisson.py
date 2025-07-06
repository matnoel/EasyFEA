# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
Poisson
=======

Poisson equation with unit load.

Reference: https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-1-poisson-equation-with-unit-load
"""

from EasyFEA import Display, ElemType, MatrixType
from EasyFEA.Geoms import Domain

# ----------------------------------------------
# Class
# ----------------------------------------------

from abc import ABC, abstractmethod
from typing import Callable
import copy
from scipy import sparse
import scipy.sparse.linalg as sla

import numpy as np
from EasyFEA.fem import FeArray, _GroupElem, Gauss


class Field:

    def __init__(self, groupElem: _GroupElem, dof_n: int):

        assert isinstance(groupElem, _GroupElem)
        self.__groupElem = groupElem

        assert 1 <= dof_n <= groupElem.dim
        self.__dof_n = dof_n

        self.__node = 0
        """activated node."""

    @property
    def groupElem(self) -> _GroupElem:
        return self.__groupElem

    @property
    def dof_n(self) -> int:
        return self.__dof_n

    def copy(self) -> "Field":
        return copy.deepcopy(self)

    def _Get_node(self) -> int:
        return self.__node

    def _Set_node(self, node: int):
        assert 0 <= node < self.groupElem.nPe
        self.__node = node

    def __call__(self):
        node = self._Get_node()
        N_pg = self.groupElem.Get_N_pg(MatrixType.mass)
        nPg, dim, _ = N_pg.shape
        array = FeArray.asfearray(N_pg[..., node].reshape(1, nPg, 1))
        return array

    @property
    def grad(self):
        node = self._Get_node()
        dN_e_pg = self.groupElem.Get_dN_e_pg(MatrixType.rigi)
        array = FeArray.asfearray(dN_e_pg[..., node])
        return array


class Form(ABC):

    def __init__(self, form: Callable[..., FeArray.FeArrayALike]):
        self._form = form

    @abstractmethod
    def Assemble(self, field: Field) -> sparse.csr_matrix:
        pass


class BilinearForm(Form):

    def Assemble(self, field):

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

        # loop over u dofs
        for i in dofs:

            # loop over v dofs
            for j in dofs:

                # activate node
                u._Set_node(nodes[i])
                v._Set_node(nodes[j])

                # get (Ne, nPg) array
                values_e_pg = form(u, v)

                # get dX to integrate
                if i == j == 0:
                    elemType = groupElem.elemType
                    nPg = values_e_pg.shape[1]
                    matrixType = Gauss._MatrixType_factory(elemType, nPg)
                    dX_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)

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


class LinearForm(Form):

    def Assemble(self, field):

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

        # loop over u dofs
        for i in dofs:

            # activate node
            v._Set_node(nodes[i])

            # get (Ne, nPg) array
            values_e_pg = form(v())

            # get dX to integrate
            if i == 0:
                elemType = groupElem.elemType
                nPg = values_e_pg.shape[1]
                matrixType = Gauss._MatrixType_factory(elemType, nPg)
                dX_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)

            # sum on gauss points
            values_e = (values_e_pg * dX_e_pg).sum(axis=1)

            # add data
            data[:, i, 0] = values_e[:, 0]

        # construct sparse matrix
        Ndof = groupElem.Nn * dof_n
        rows = groupElem.Get_assembly_e(dof_n)
        columns = np.zeros_like(rows)
        matrix = sparse.csr_matrix(
            (data.ravel(), (rows.ravel(), columns.ravel())), (Ndof, 1), dtype=float
        )

        return matrix


if __name__ == "__main__":
    Display.Clear()

    # ----------------------------------------------
    # Mesh
    # ----------------------------------------------

    contour = Domain((0, 0), (1, 1), 1 / 2**6)

    mesh = contour.Mesh_2D([], ElemType.TRI3, isOrganised=True)

    from EasyFEA import Materials, Simulations

    mat = Materials.Thermal(2, 1, 0)
    simu = Simulations.ThermalSimu(mesh, mat, useIterativeSolvers=False)

    nodes = mesh.Nodes_Tags(["L0", "L1", "L2", "L3"])
    simu.add_dirichlet(nodes, [0], ["t"])
    simu.add_neumann(mesh.nodes, [np.ones(mesh.Nn) / mesh.Nn], ["t"])

    simu.Solve()

    # Display.Plot_Result(simu, "thermal")

    # Display.Plot_Mesh(mesh)

    # ----------------------------------------------
    # Formulations
    # ----------------------------------------------

    field = Field(mesh.groupElem, 1)

    @BilinearForm
    def bilinear_form(u: Field, v: Field):
        return u.grad.dot(v.grad)

    @LinearForm
    def linear_form(v: Field):
        return 1.0 * v

    A = bilinear_form.Assemble(field)

    b = linear_form.Assemble(field)

    dofsKnown = nodes

    x = np.zeros(mesh.Nn)
    x[dofsKnown] = 0

    dofsUnknown = [node for node in mesh.nodes if node not in dofsKnown]

    Ai = A[dofsUnknown, :].tocsc()
    Aii = Ai[:, dofsUnknown].tocsr()
    Aic = Ai[:, dofsKnown].tocsr()
    bi = b.toarray().ravel()[dofsUnknown]
    xc = x[dofsKnown]

    bDirichlet = Aic @ xc

    xi = sla.cg(Aii, bi - bDirichlet)[0]

    # apply result to global vector
    x[dofsUnknown] = xi

    Display.Plot_Result(mesh, x)

    pass
