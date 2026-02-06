# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import Folder, Mesh, USD

from .GLTF_test import list_mesh, get_frames, get_simu

folder_results = Folder.Results_Dir()


def validate(usdaFile: str):
    # no checks for now
    pass


class TestUSD:

    def test_save_mesh(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh", mkdir=True)

        for mesh in list_mesh:

            usdaFile = USD.Save_mesh(mesh, folder, mesh.elemType)

            validate(usdaFile)

    def test_save_mesh_frames(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            usdaFile = USD.Save_mesh(
                mesh, folder, mesh.elemType, list_displacementMatrix=frames
            )

            validate(usdaFile)

    def test_save_mesh_frames_sols(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames_sols", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            # norm
            usdaFile = USD.Save_mesh(
                mesh,
                folder,
                mesh.elemType,
                list_displacementMatrix=frames,
                list_nodesValues_n=frames,
            )
            validate(usdaFile)

            # x
            usdaFile = USD.Save_mesh(
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

            usdaFile = USD.Save_simu(simu, "uy", folder, mesh.elemType, fps=1)

            validate(usdaFile)
