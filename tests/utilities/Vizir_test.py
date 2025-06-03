# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Mesher, ElemType, Mesh, np, Vizir
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


class TestVizir:

    def test_barycentric_coordinates(self, meshes: list[Mesh]):

        for mesh in meshes:

            groupElem = mesh.groupElem
            elemType = groupElem.elemType
            local_coords = groupElem.Get_Local_Coords()
            vertices = local_coords[: groupElem.nbCorners]

            if elemType.startswith(("SEG", "TETRA", "TRI")):

                barycentric_coords = Vizir._Get_BaryCentric_Coordinates(groupElem)

                verif_coords = np.zeros_like(groupElem.Get_Local_Coords())
                for i, coefs in enumerate(barycentric_coords):
                    weighted_coords = coefs[:, None] * vertices
                    verif_coords[i] = np.sum(weighted_coords, axis=0)

            else:
                continue

            test = np.linalg.norm(verif_coords - local_coords) / np.linalg.norm(
                local_coords
            )
            assert test < 1e-12
