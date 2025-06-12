# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Mesher, ElemType, Mesh, plt, np
from EasyFEA.Geoms import Domain, Point
from EasyFEA.geoms import Rotate
from EasyFEA import Display as Display


class TestGmsh:

    def test_MoveMesh(self):
        """Check that if you move the mesh the properties remain the same."""

        meshes = Mesher._Construct_2D_meshes()
        meshes.extend(Mesher._Construct_3D_meshes())

        def testSameMesh(mesh1: Mesh, mesh2: Mesh) -> None:

            testSame(mesh1.dim, mesh2.dim)
            testSame(mesh1.Nn, mesh2.Nn)
            testSame(mesh1.Ne, mesh2.Ne)

            testSame(mesh1.length, mesh2.length)  # same length
            testSame(mesh1.area, mesh2.area)  # same area
            if mesh1.dim == 2:
                assert mesh1.volume is None
                assert mesh2.volume is None
            else:
                testSame(mesh1.volume, mesh2.volume)  # same volume

        def testSame(val1, val2) -> None:
            """Test if the values are the same"""
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                test = np.linalg.norm(val1 - val2) / np.linalg.norm(val2)
            else:
                test = np.abs((val1 - val2) / val2)
            assert test <= 1e-12

        for mesh in meshes:

            # Translate
            meshTranslate = mesh.copy()
            dec = np.random.rand(3)
            meshTranslate.Translate(*dec)
            testSame(meshTranslate.center, mesh.center + dec)  # same center
            testSameMesh(meshTranslate, mesh)

            # Rotate
            rot = np.random.rand() * 360
            meshRotate = mesh.copy()
            newCenter = Rotate(mesh.center, rot, mesh.center, (1, 0))
            meshRotate.Rotate(rot, meshRotate.center, (1, 0))
            testSame(meshRotate.center, newCenter)  # same center
            testSameMesh(meshRotate, mesh)
            axis = (1, 3, -1)
            newCenter = Rotate(newCenter, rot, newCenter, axis)
            meshRotate.Rotate(rot, meshRotate.center, axis)
            testSame(meshRotate.center, newCenter)  # same center
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
                mesh = Mesher().Mesh_2D(contour, [], elemType, isOrganised=True)
            else:
                mesh = Mesher().Mesh_Extrude(
                    contour, [], [0, 0, 1], 2, elemType, isOrganised=True
                )

            Ne = mesh.Ne

            coef_recombine = 1 if elemType.startswith(("QUAD", "HEXA")) else 2
            coef_dim = 1 if dim == 2 else 2
            assert Ne == 4 * coef_recombine * coef_dim
