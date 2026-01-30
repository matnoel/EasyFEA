# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA import Mesher, Models, Simulations, SolverType
from EasyFEA.Geoms import Domain


class TestPhaseField:

    def test_PhaseField(self):

        a = 1
        l0 = a / 10
        meshSize = l0 / 2
        mesh = Mesher().Mesh_2D(Domain((0, 0), (a, a), meshSize))

        nodes_0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
        nodes_a = mesh.Nodes_Conditions(lambda x, y, z: x == a)

        material = Models.Elastic.Isotropic(
            2, E=210000, v=0.3, planeStress=True, thickness=1
        )

        splits = list(Models.PhaseField.SplitType)
        regularizations = list(Models.PhaseField.ReguType)

        for split in splits:
            for regu in regularizations:

                pfm = Models.PhaseField(material, split, regu, 2700, l0)

                print(f"{split} {regu}")

                simu = Simulations.PhaseField(mesh, pfm)
                simu.solver = SolverType.scipy

                for ud in np.linspace(0, 5e-8 * 400, 3):

                    simu.Bc_Init()
                    simu.add_dirichlet(nodes_0, [0, 0], ["x", "y"])
                    simu.add_dirichlet(nodes_a, [ud], ["x"])

                    simu.Solve()
                    simu.Save_Iter()

    def test_Update_PhaseField(self):
        """Function use to check that modifications on phase field material activate the update of the simulation"""

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Mesher().Mesh_2D(Domain((0, 0), (1, 1)))

        matIsot = Models.Elastic.Isotropic(2)
        # E, v, planeStress

        pfm = Models.PhaseField(matIsot, "He", "AT1", 1, 0.01)
        # split, regu, split, Gc, l0, solver, A

        simu = Simulations.PhaseField(mesh, pfm)

        simu.Get_K_C_M_F("elastic")
        assert simu.needUpdate
        simu.Get_K_C_M_F("damage")
        assert not simu.needUpdate
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
