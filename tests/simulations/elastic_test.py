# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, Models, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Simulations


class TestElastic:

    def test_Elastic(self):
        # For each type of mesh one simulates

        dim = 2

        # Load to apply
        P = -800  # N

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a / 10)
        inclusions = [Circle(Point(a / 2, a / 2), a / 3, a / 10)]

        doMesh2D = lambda elemType: Mesher().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Mesher().Mesh_Extrude(
            domain, inclusions, [0, 0, -a], [3], elemType
        )

        listMesh = [doMesh2D(elemType) for elemType in ElemType.Get_2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in ElemType.Get_3D()]

        # For each mesh
        for mesh in listMesh:

            dim = mesh.dim

            comportement = Models.ElasIsot(dim, thickness=a)

            simu = Simulations.ElasticSimu(mesh, comportement, verbosity=False)

            noeuds_en_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            noeuds_en_L = mesh.Nodes_Conditions(lambda x, y, z: x == a)

            simu.add_dirichlet(noeuds_en_0, [0, 0], ["x", "y"])
            simu.add_surfLoad(noeuds_en_L, [P / a / a], ["y"])

            simu.Solve()
            simu.Save_Iter()

            # static
            ax = Display.Plot_Result(simu, "ux", plotMesh=True, nodeValues=True)
            # plt.pause(1e-12)
            plt.close(ax.figure)

            # dynamic
            simu.Solver_Set_Hyperbolic_Algorithm(dt=0.1)
            simu.Solve()
            # don't plot because result is not relevant

    def test_Update_Elastic(self):
        """Function use to check that modifications on elastic material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate == True  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1, 1)))

        matIsot = Models.ElasIsot(2)
        # E, v, planeStress

        simu = Simulations.ElasticSimu(mesh, matIsot)
        simu.Get_K_C_M_F()
        assert (
            simu.needUpdate == False
        )  # check that need update is now set to false once Get_K_C_M_F() get called
        matIsot.E *= 2
        DoTest(simu)
        matIsot.v = 0.2
        DoTest(simu)
        matIsot.planeStress = not matIsot.planeStress
        DoTest(simu)
        try:
            # must return an error
            matIsot.E = -10
        except AssertionError:
            assert simu.needUpdate == False
        try:
            # must return an error
            matIsot.v = 10
        except AssertionError:
            assert simu.needUpdate == False
        try:
            matIsot.planeStress = 10
        except AssertionError:
            assert simu.needUpdate == False

        matElasIsotTrans = Models.ElasIsotTrans(2, 10, 10, 10, 0.1, 0.1)
        # El, Et, Gl, vl, vt, planeStress
        simu = Simulations.ElasticSimu(mesh, matElasIsotTrans)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False
        matElasIsotTrans.El *= 2
        DoTest(simu)
        matElasIsotTrans.Et *= 2
        DoTest(simu)
        matElasIsotTrans.Gl *= 2
        DoTest(simu)
        matElasIsotTrans.vl = 0.2
        DoTest(simu)
        matElasIsotTrans.vt = 0.4
        DoTest(simu)
        matElasIsotTrans.planeStress = not matElasIsotTrans.planeStress
        DoTest(simu)

        matAnisot = Models.ElasAnisot(2, matElasIsotTrans.C, False, (0, 1), (-1, 0))
        # Set_C,
        simu = Simulations.ElasticSimu(mesh, matAnisot)
        simu.Get_K_C_M_F()
        assert simu.needUpdate == False
        matAnisot.Set_C(matIsot.C, False)
        DoTest(simu)
