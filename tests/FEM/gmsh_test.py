# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import ElemType, Mesh
from EasyFEA.Geoms import Domain, Point
from EasyFEA.Geoms import Rotate
from EasyFEA import Matplotlib as Matplotlib

from _numerics import rel_error


class TestGmsh:

    def test_MoveMesh(
        self, cracked_meshes_2D: list[Mesh], cracked_meshes_3D: list[Mesh]
    ):
        """Check that if you move the mesh the properties remain the same."""

        meshes = cracked_meshes_2D + cracked_meshes_3D

        def testSameMesh(mesh1: Mesh, mesh2: Mesh) -> None:

            assert mesh1.dim == mesh2.dim
            assert mesh1.Nn == mesh2.Nn
            assert mesh1.Ne == mesh2.Ne

            assert rel_error(mesh1.length, mesh2.length) <= 1e-12
            assert rel_error(mesh1.area, mesh2.area) <= 1e-12
            if mesh1.dim == 2:
                assert mesh1.volume is None
                assert mesh2.volume is None
            else:
                assert rel_error(mesh1.volume, mesh2.volume) <= 1e-12

        for mesh in meshes:

            # Translate
            meshTranslate = mesh.copy()
            dec = np.random.rand(3)
            meshTranslate.Translate(*dec)
            assert rel_error(meshTranslate.center, mesh.center + dec) <= 1e-12
            testSameMesh(meshTranslate, mesh)

            # Rotate
            rot = np.random.rand() * 360
            meshRotate = mesh.copy()
            newCenter = Rotate(mesh.center, rot, mesh.center, (1, 0))[0]
            meshRotate.Rotate(rot, meshRotate.center, (1, 0))
            assert rel_error(meshRotate.center, newCenter) <= 1e-12
            testSameMesh(meshRotate, mesh)
            axis = (1, 3, -1)
            newCenter = Rotate(newCenter, rot, newCenter, axis)[0]
            meshRotate.Rotate(rot, meshRotate.center, axis)
            assert rel_error(meshRotate.center, newCenter) <= 1e-12
            testSameMesh(meshRotate, mesh)

    def test_mesh_isOrganised(self):

        contour = Domain(Point(), Point(1, 1), 1 / 2)

        elemTypes = ElemType.Get_2D()
        elemTypes.extend(ElemType.Get_3D())
        elemTypes.remove("TETRA4")
        elemTypes.remove("TETRA10")

        for elemType in elemTypes:

            print(elemType)

            dim = 2 if elemType in ElemType.Get_2D() else 3

            if dim == 2:
                mesh = contour.Mesh_2D([], elemType, isOrganised=True)
            else:
                mesh = contour.Mesh_Extrude(
                    [], [0, 0, 1], 2, elemType, isOrganised=True
                )

            Ne = mesh.Ne

            coef_recombine = 1 if elemType.startswith(("QUAD", "HEXA")) else 2
            coef_dim = 1 if dim == 2 else 2
            assert Ne == 4 * coef_recombine * coef_dim

    def test_mesh_error_carries_geoms(self):
        from EasyFEA.FEM._mesher import MeshError
        from EasyFEA.Geoms import Points

        contour = Points([(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)])
        with pytest.raises(MeshError) as error:
            contour.Mesh_2D()
        assert error.value.geoms == [contour]
