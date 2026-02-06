# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

import numpy as np

from pygltflib import GLTF2
from pygltflib.validator import validate

# https://samaust.github.io/pygltflib/actions/validate.html

from EasyFEA import Folder, ElemType, Mesh, GLTF, Models, Simulations
from EasyFEA.Geoms import Domain, Circle, Rotate, Translate

folder_results = Folder.Results_Dir()


@pytest.fixture
def list_mesh() -> list[Mesh]:

    meshSize = 1 / 3
    contour = Domain((0, 0), (1, 1), meshSize)
    circle = Circle((0.5, 0.5), 0.3, meshSize)

    list_mesh = [
        contour.Mesh_2D([circle], elemType=elemType) for elemType in ElemType.Get_2D()
    ]

    list_mesh.extend(
        [
            contour.Mesh_Extrude([circle], layers=[3], elemType=elemType)
            for elemType in ElemType.Get_3D()
        ]
    )

    return list_mesh


def get_frames(mesh: Mesh):
    coord = mesh.coord

    thetas = np.linspace(0, 360, 4)
    dys = np.linspace(0, 1, thetas.size // 2)
    dys = [*dys, *dys[::-1]]
    frames = [
        Translate(Rotate(coord, theta, mesh.center, direction=(0, 1, 0)), dy=dy) - coord
        for theta, dy in zip(thetas, dys)
    ]
    return frames


def get_simu(mesh: Mesh):

    ymin, ymax = mesh.coord[:, 1].max(), mesh.coord[:, 1].max()

    mat = Models.Elastic.Isotropic(mesh.dim)
    simu = Simulations.Elastic(mesh, mat)

    for i in range(3):
        simu.Bc_Init()
        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: y == ymin),
            [0] * mesh.dim,
            simu.Get_unknowns(),
        )
        simu.add_dirichlet(
            mesh.Nodes_Conditions(lambda x, y, z: y == ymax), [0.1 * i], ["y"]
        )
        simu.Solve()
        simu.Save_Iter()

    return simu


class TestGLTF:

    def test_save_mesh(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh", mkdir=True)

        for mesh in list_mesh:

            glbFile = GLTF.Save_mesh(mesh, folder, mesh.elemType)

            gltf = GLTF2().load(glbFile)

            validate(gltf)

    def test_save_mesh_frames(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            glbFile = GLTF.Save_mesh(
                mesh, folder, mesh.elemType, list_displacementMatrix=frames
            )

            gltf = GLTF2().load(glbFile)

            validate(gltf)

    def test_save_mesh_frames_sols(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "mesh_frames_sols", mkdir=True)

        for mesh in list_mesh:

            frames = get_frames(mesh)

            # norm
            glbFile = GLTF.Save_mesh(
                mesh,
                folder,
                mesh.elemType,
                list_displacementMatrix=frames,
                list_nodesValues_n=frames,
            )
            gltf = GLTF2().load(glbFile)
            validate(gltf)

            # x
            glbFile = GLTF.Save_mesh(
                mesh,
                folder,
                mesh.elemType,
                list_displacementMatrix=frames,
                list_nodesValues_n=[frame[:, 0] for frame in frames],
            )
            gltf = GLTF2().load(glbFile)
            validate(gltf)

    def test_save_simu(self, list_mesh: list[Mesh]):

        folder = Folder.Join(folder_results, "simu", mkdir=True)

        for mesh in list_mesh:

            simu = get_simu(mesh)

            glbFile = GLTF.Save_simu(simu, "uy", folder, mesh.elemType, fps=1)

            gltf = GLTF2().load(glbFile)

            validate(gltf)
