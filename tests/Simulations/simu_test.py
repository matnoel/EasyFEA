# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

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

    @staticmethod
    def _mini(folder=""):
        mesh = Domain(Point(), Point(1, 1), meshSize=0.5).Mesh_2D()
        mat = Models.Elastic.Isotropic(
            dim=2, E=210e3, v=0.3, planeStress=True, thickness=1.0
        )
        return Simulations.Elastic(mesh, mat, folder=folder)

    def test_history_pins_mode_per_entry(self, tmp_path):
        """Each Save_Iter pins its entry as a dict (folder=='') or pickle path (folder set) at write time. Get_result dispatches uniformly."""
        simu = self._mini()
        for _ in range(2):
            simu.Save_Iter()  # in-memory
        simu.folder = str(tmp_path / "run")
        for _ in range(2):
            simu.Save_Iter()  # on-disk

        assert simu.Niter == 4
        for i in range(simu.Niter):  # all resolve to dicts
            assert "indexMesh" in simu.Get_results(i)
        # last two pickles really do exist on disk
        results_dir = os.path.join(simu.folder, "Results")
        assert sorted(os.listdir(results_dir)) == [f"results{i}.pickle" for i in (2, 3)]

    def test_folder_at_construction(self, tmp_path):
        """Passing `folder=...` at construction must not crash — setter runs before __list_results exists."""
        simu = self._mini(folder=str(tmp_path / "run"))
        simu.Save_Iter()
        assert os.path.isfile(os.path.join(simu.folder, "Results", "results0.pickle"))
        assert "indexMesh" in simu.Get_results(0)

    def test_folder_can_be_unset(self, tmp_path):
        """`simu.folder = ''` after some on-disk Save_Iter calls is fine — old entries keep their paths; new ones go to memory."""
        simu = self._mini(folder=str(tmp_path / "run"))
        simu.Save_Iter()
        simu.folder = ""
        simu.Save_Iter()
        assert "indexMesh" in simu.Get_results(0)  # disk entry
        assert "indexMesh" in simu.Get_results(1)  # in-memory entry
