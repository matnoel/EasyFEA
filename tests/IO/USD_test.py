# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Folder, Mesh, IO

from .GLTF_test import list_mesh, get_frames, get_simu

folder_results = Folder.Results_Dir()


def validate(usdaFile: str):
    # no checks for now
    pass


class TestUSD:

    def test_save_mesh(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh", mkdir=True)

        for mesh in list_mesh:

            usdaFile = IO.USD.Save_mesh(mesh, folder, mesh.elemType)

            validate(usdaFile)

    def test_save_mesh_frames(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            usdaFile = IO.USD.Save_mesh(
                mesh, folder, mesh.elemType, list_displacementMatrix=frames
            )

            validate(usdaFile)

    def test_save_mesh_frames_sols(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames_sols", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            # norm
            usdaFile = IO.USD.Save_mesh(
                mesh,
                folder,
                mesh.elemType,
                list_displacementMatrix=frames,
                list_nodesValues_n=frames,
            )
            validate(usdaFile)

            # x
            usdaFile = IO.USD.Save_mesh(
                mesh,
                folder,
                mesh.elemType,
                list_displacementMatrix=frames,
                list_nodesValues_n=[frame[:, 0] for frame in frames],
            )
            validate(usdaFile)

    def test_save_simu(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "simu", mkdir=True)

        for mesh in list_mesh:

            simu = get_simu(mesh)

            saved = IO.USD.Save_simu(simu, folder, results=["uy"], fps=1)

            assert saved == folder

    def test_save_simu_needs_a_simulation(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "simu", mkdir=True)

        assert IO.USD.Save_simu(list_mesh[0], folder, results=["uy"]) is None  # type: ignore [arg-type]

    def test_save_mesh_is_binary_only(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh", mkdir=True)

        with pytest.raises(ValueError):
            IO.USD.Save_mesh(list_mesh[0], folder, "text", useBinary=False)
