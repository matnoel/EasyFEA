# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, Models, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Simulations


class TestThermal:

    def test_Thermal(self):

        a = 1

        domain = Domain(Point(0, 0), Point(a, a), a / 10)
        inclusions = [Circle(Point(a / 2, a / 2), a / 3, a / 10)]

        doMesh2D = lambda elemType: Mesher().Mesh_2D(domain, inclusions, elemType)
        doMesh3D = lambda elemType: Mesher().Mesh_Extrude(
            domain, inclusions, [0, 0, -a], [3], elemType
        )

        listMesh = [doMesh2D(elemType) for elemType in ElemType.Get_2D()]
        [listMesh.append(doMesh3D(elemType)) for elemType in ElemType.Get_3D()]

        self.thermalSimulation = []

        for mesh in listMesh:

            dim = mesh.dim

            thermalModel = Models.Thermal(dim=dim, k=1, c=1, thickness=a)

            simu = Simulations.ThermalSimu(mesh, thermalModel, False)

            noeuds0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
            noeudsL = mesh.Nodes_Conditions(lambda x, y, z: x == a)

            simu.add_dirichlet(noeuds0, [0], ["t"])
            simu.add_dirichlet(noeudsL, [40], ["t"])
            simu.Solve()
            simu.Save_Iter()

            ax = Display.Plot_Result(simu, "thermal", nodeValues=True, plotMesh=True)
            # plt.pause(1e-12)
            plt.close(ax.figure)

    def test_Update_Thermal(self):
        """Function use to check that modifications on thermal material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate == True  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1, 1)))

        thermal = Models.Thermal(2, 1, 1)
        # k, c

        simu = Simulations.ThermalSimu(mesh, thermal)
        simu.Get_K_C_M_F()
        assert (
            simu.needUpdate == False
        )  # check that need update is now set to false once Get_K_C_M_F() get called
        thermal.k *= 2
        DoTest(simu)
        thermal.c *= 0.2
        DoTest(simu)
