# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Mesher, ElemType, Mesh, MeshIO, np
from EasyFEA.Geoms import Points

L = 2
H = 1


@pytest.fixture
def meshes() -> list[Mesh]:

    meshSize = H / 3

    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)
    meshes: list[Mesh] = []

    # 1d meshes
    for elemType in ElemType.Get_1D():

        mesher = Mesher()
        factory = mesher._factory

        p1 = factory.addPoint(0, 0, 0)
        p2 = factory.addPoint(1, 0, 0)
        factory.addLine(p1, p2)

        mesher._Mesh_Generate(1, elemType)

        meshes.append(mesher._Mesh_Get_Mesh())

    # 2d meshes
    for elemType in ElemType.Get_2D():
        mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)
        meshes.append(mesh)

    # 3d meshes
    for elemType in ElemType.Get_3D():
        mesh = Mesher().Mesh_Extrude(
            contour, [], [0, 0, L], [3], elemType, isOrganised=True
        )
        meshes.append(mesh)

    return meshes


class TestMeshIO:

    def test_surface_reconstruction(self, meshes: list[Mesh]):

        for mesh in meshes:

            newMesh = MeshIO.Surface_reconstruction(mesh)

            expectedNe = np.sum(
                [groupElem.Ne for groupElem in mesh.Get_list_groupElem(2)]
            )
            computedNe = np.sum(
                [groupElem.Ne for groupElem in newMesh.Get_list_groupElem(2)]
            )

            diff_Ne = expectedNe - computedNe
            assert diff_Ne == 0

            if mesh.dim >= 1:
                diff_length = mesh.length - newMesh.length
                assert diff_length < 1e-12

            if mesh.dim >= 2:
                diff_area = mesh.area - newMesh.area
                assert diff_area < 1e-12

            if mesh.dim >= 3:
                diff_volume = mesh.volume - newMesh.volume
                assert diff_volume < 1e-12
