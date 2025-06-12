# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Materials, Simulations


class TestPhaseField:

    def test_PhaseField(self):

        a = 1
        l0 = a / 10
        meshSize = l0 / 2
        mesh = Mesher._Construct_2D_meshes(L=a, h=a, meshSize=meshSize)[
            5
        ]  # take the first mesh

        nodes_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_a = mesh.Nodes_Conditions(lambda x, y, z: x == a)

        material = Materials.Elas_Isot(
            2, E=210000, v=0.3, planeStress=True, thickness=1
        )

        splits = list(Materials.PhaseField.SplitType)
        regularizations = list(Materials.PhaseField.ReguType)

        for split in splits:
            for regu in regularizations:

                pfm = Materials.PhaseField(material, split, regu, 2700, l0)

                print(f"{split} {regu}")

                simu = Simulations.PhaseFieldSimu(mesh, pfm)

                for ud in np.linspace(0, 5e-8 * 400, 3):

                    simu.Bc_Init()
                    simu.add_dirichlet(nodes_0, [0, 0], ["x", "y"])
                    simu.add_dirichlet(nodes_a, [ud], ["x"])

                    simu.Solve()
                    simu.Save_Iter()

    def test_Update_PhaseField(self):
        """Function use to check that modifications on phase field material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate == True  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1, 1)))

        matIsot = Materials.Elas_Isot(2)
        # E, v, planeStress

        pfm = Materials.PhaseField(matIsot, "He", "AT1", 1, 0.01)
        # split, regu, split, Gc, l0, solver, A

        simu = Simulations.PhaseFieldSimu(mesh, pfm)

        simu.Get_K_C_M_F("elastic")
        assert simu.needUpdate == True
        simu.Get_K_C_M_F("damage")
        assert simu.needUpdate == False
        # matrices are updated once damage and displacement matrices are build

        matIsot.E *= 2
        DoTest(simu)
        matIsot.v = 0.1
        DoTest(simu)

        pfm.split = "Miehe"
        DoTest(simu)
        pfm.regularization = "AT2"
        DoTest(simu)
        pfm.Gc = 10
        DoTest(simu)
        pfm.l0 = 1
        DoTest(simu)
        pfm.solver = pfm.SolverType.BoundConstrain
        DoTest(simu)
        pfm.A = np.eye(2) * 3
        DoTest(simu)
