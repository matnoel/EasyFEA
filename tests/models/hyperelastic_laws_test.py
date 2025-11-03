# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Mesher, ElemType, Models, Simulations, np
from EasyFEA.fem._linalg import Trace, TensorProd
from EasyFEA.models import Project_Kelvin, HyperElastic
from EasyFEA.Geoms import Domain


@pytest.fixture
def simuIsot():

    L = 120
    h = 13
    meshSize = h / 3

    contour = Domain((0, 0), (L, h), h / 3)

    mesh = Mesher().Mesh_Extrude(
        contour, [], [0, 0, h], [h / meshSize], ElemType.TETRA4
    )
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    matIsot = Models.ElasIsot(3)
    simuIsot = Simulations.ElasticSimu(mesh, matIsot)

    simuIsot.add_dirichlet(nodesX0, [0, 0, 0], simuIsot.Get_unknowns())
    simuIsot.add_dirichlet(nodesXL, [1], ["x"])

    simuIsot.Solve()

    return simuIsot


class TestSaintVenantKirchhoff:

    def test_chain_rule(self, simuIsot: Simulations.ElasticSimu):

        matIsot: Models.ElasIsot = simuIsot.material
        mesh = simuIsot.mesh
        u = simuIsot.displacement

        matrixType = "rigi"

        mat = Models.SaintVenantKirchhoff(3, matIsot.get_lambda(), matIsot.get_mu())

        # test W
        E = HyperElastic(mesh, u, matrixType).Compute_GreenLagrange()
        # W_hyper = 1/2 * mat.lmbda * Trace(E)**2 + mat.mu * E.ddot(E)
        W_hyper = 1 / 2 * mat.lmbda * Trace(E) ** 2 + mat.mu * Trace(E @ E)
        W_e_pg = mat.Compute_W(mesh, u, matrixType)
        diff_w = W_hyper - W_e_pg
        assert np.linalg.norm(diff_w) / np.linalg.norm(W_hyper) < 1e-11

        # test dW
        I = np.array([1, 1, 1, 0, 0, 0])
        dW_hyper = mat.lmbda * Trace(E) * I + 2 * mat.mu * Project_Kelvin(E, 2)
        dW_e_pg = mat.Compute_dWde(mesh, u, matrixType)
        diff_dW = dW_hyper - dW_e_pg
        assert np.linalg.norm(diff_dW) / np.linalg.norm(dW_hyper) < 1e-11

        # test d2W
        d2W_hyper = mat.lmbda * TensorProd(I, I) + 2 * mat.mu * Project_Kelvin(
            TensorProd(np.eye(3), np.eye(3), symmetric=True)
        )
        d2W_e_pg = mat.Compute_d2Wde(mesh, u, matrixType)
        diff_d2W = d2W_hyper - d2W_e_pg
        assert np.linalg.norm(diff_d2W) / np.linalg.norm(d2W_hyper) < 1e-11
