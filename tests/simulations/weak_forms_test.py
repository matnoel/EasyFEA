# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import ElemType, Models, Simulations, SolverType
from EasyFEA.fem import FeArray, Field, BiLinearForm, Sym_Grad, Trace
from EasyFEA.Geoms import Domain


class TestWeakForms:

    def test_thermal(self):

        contour = Domain((0, 0), (1, 1))
        mesh = contour.Mesh_2D([], ElemType.TRI6)

        nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodesX1 = mesh.Nodes_Conditions(lambda x, y, z: x == 1)

        # ----------------------------------------------
        # thermal simu
        # ----------------------------------------------
        material = Models.Thermal(k=1)

        thermalSimu = Simulations.Thermal(mesh, material)
        thermalSimu.solver = SolverType.scipy

        thermalSimu.add_dirichlet(nodesX0, [0], ["t"])
        thermalSimu.add_dirichlet(nodesX1, [1], ["t"])

        thermalSimu.Solve()

        # ----------------------------------------------
        # weak form simu
        # ----------------------------------------------

        field = Field(mesh.groupElem, 1)

        @BiLinearForm
        def bilinear_form(u: Field, v: Field):
            return u.grad.dot(v.grad)

        weakForms = Models.WeakForms(field, bilinear_form)

        simu = Simulations.WeakForms(mesh, weakForms)
        simu.solver = SolverType.scipy

        simu.add_dirichlet(nodesX0, [0], ["u"])
        simu.add_dirichlet(nodesX1, [1], ["u"])

        simu.Solve()

        # ----------------------------------------------
        # Test
        # ----------------------------------------------

        norm_diff = np.linalg.norm(simu.u - thermalSimu.thermal)

        assert norm_diff < 1e-12

    def test_linear_elastic(self):

        contour = Domain((0, 0), (1, 1))
        mesh = contour.Mesh_2D([], ElemType.TRI6)

        nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodesX1 = mesh.Nodes_Conditions(lambda x, y, z: x == 1)

        # ----------------------------------------------
        # thermal simu
        # ----------------------------------------------
        material = Models.Elastic.Isotropic(dim=2, E=210000, v=0.3, planeStress=False)
        lmbda = material.get_lambda()
        mu = material.get_mu()

        elasticSimu = Simulations.Elastic(mesh, material)
        elasticSimu.solver = SolverType.scipy

        elasticSimu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
        elasticSimu.add_dirichlet(nodesX1, [0.1], ["x"])

        elasticSimu.Solve()

        # ----------------------------------------------
        # weak form simu
        # ----------------------------------------------

        field = Field(mesh.groupElem, 2)

        def S(u: Field) -> FeArray:
            Eps = Sym_Grad(u)
            return 2 * mu * Eps + lmbda * Trace(Eps) * np.eye(2)

        @BiLinearForm
        def ComputeK(u: Field, v: Field):
            Sig = S(u)
            Eps = Sym_Grad(v)
            return Sig.ddot(Eps)

        weakForms = Models.WeakForms(field, ComputeK)

        simu = Simulations.WeakForms(mesh, weakForms)
        simu.solver = SolverType.scipy

        simu.add_dirichlet(nodesX0, [0, 0], ["x", "y"])
        simu.add_dirichlet(nodesX1, [0.1], ["x"])

        simu.Solve()

        # ----------------------------------------------
        # Test
        # ----------------------------------------------

        norm_diff = np.linalg.norm(simu.u - elasticSimu.displacement)
        assert norm_diff < 1e-12
