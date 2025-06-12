# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Display, plt, np
from EasyFEA.Geoms import Domain, Circle, Point, Line
from EasyFEA import Mesher, ElemType
from EasyFEA import Materials, Simulations


class TestSimu:

    def test_Update_Mesh(self):

        def DoTest(simu: Simulations._Simu) -> None:
            assert simu.needUpdate == True  # should trigger the event
            simu.Need_Update(False)  # init

        mesh = Mesher().Mesh_2D(Domain(Point(), Point(1, 1)))

        thermal = Materials.Thermal(2, 1, 1)
        # k, c

        simu = Simulations.ThermalSimu(mesh, thermal)
        simu.Get_K_C_M_F()
        assert (
            simu.needUpdate == False
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
            assert simu.needUpdate == False

        try:
            # must return an error
            mesh.Translate(dz=20)
        except AssertionError:
            assert simu.needUpdate == False

        simu.mesh = mesh.copy()
        DoTest(simu)
