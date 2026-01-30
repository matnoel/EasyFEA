# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA import Mesher, ElemType, Mesh
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


class TestGroupElem:

    def test_nPe(self, meshes: list[Mesh]):

        for mesh in meshes:

            groupElem = mesh.groupElem

            assert groupElem.nPe == (
                groupElem.Nvertex
                + groupElem.Nedge
                + groupElem.Nface
                + groupElem.Nvolume
            )

    def test_shape_fucntion(self, meshes: list[Mesh]):

        for mesh in meshes:

            groupElem = mesh.groupElem

            shape_functions = groupElem._N()

            local_coords = groupElem.Get_Local_Coords()

            for shape_function, coords in zip(shape_functions, local_coords):

                eval = shape_function[0](*coords)
                assert np.abs(1 - eval) < 1e-12
