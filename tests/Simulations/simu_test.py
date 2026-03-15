# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Models, Simulations
from EasyFEA.Geoms import Domain, Point


class TestSimu:

    def test_Update_Mesh(self):

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Domain(Point(), Point(1, 1)).Mesh_2D()

        thermal = Models.Thermal(1, 1)
        # k, c

        simu = Simulations.Thermal(mesh, thermal)
        simu.Get_K_C_M_F()
        assert (
            not simu.needUpdate
        )  # check that need update is now set to false once Get_K_C_M_F() get called

        mesh.Rotate(45, mesh.center)
        DoTest(simu)

        mesh.Translate(dy=-10)
        DoTest(simu)

        mesh.Symmetry(mesh.center, (1, 0))
        DoTest(simu)

        try:
            # must return an error
            mesh.Rotate(45, mesh.center, direction=(1, 0))
        except AssertionError:
            assert not simu.needUpdate

        try:
            # must return an error
            mesh.Translate(dz=20)
        except AssertionError:
            assert not simu.needUpdate

        simu.mesh = mesh.copy()
        DoTest(simu)
