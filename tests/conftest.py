# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable, Optional

import pytest
import numpy as np

from EasyFEA import ElemType, Mesh
from EasyFEA.Geoms import Points, Point, Circle, Line

L = 2
H = 1
B = 1

MakeMesh2D = Callable[..., Mesh]
MakeMesh3D = Callable[..., Mesh]


def _make_mesh_2D(
    L: float = L,
    H: float = H,
    elemType: ElemType = ElemType.TRI3,
    meshSize: Optional[float] = None,
    isOrganised: bool = False,
    inclusions: list[Circle] = [],
    cracks: list[Line] = [],
) -> Mesh:
    meshSize = meshSize or H / 3
    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)
    return contour.Mesh_2D(inclusions, elemType, cracks, isOrganised=isOrganised)


def _make_mesh_3D(
    L: float = L,
    H: float = H,
    B: float = B,
    elemType: ElemType = ElemType.TETRA4,
    meshSize: Optional[float] = None,
    isOrganised: bool = False,
    inclusions: list[Circle] = [],
) -> Mesh:
    meshSize = meshSize or H / 3
    contour = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize)
    return contour.Mesh_Extrude(
        inclusions, [0, 0, -B], [3], elemType, isOrganised=isOrganised
    )


@pytest.fixture
def make_mesh_2D() -> MakeMesh2D:
    """Factory: build a rectangular 2D mesh. `contour = Points(...)` under the hood, per element type / geometry variant needed."""
    return _make_mesh_2D


@pytest.fixture
def make_mesh_3D() -> MakeMesh3D:
    """Factory: build an extruded 3D mesh. `contour = Points(...)` under the hood, per element type / geometry variant needed."""
    return _make_mesh_3D


@pytest.fixture(scope="session")
def cracked_meshes_2D() -> list[Mesh]:
    """One mesh per 2D element type for each geometry variant: plain, organised, hollow-circle inclusion, filled-circle inclusion, closed crack, open crack."""

    meshSize = H / 3
    area = L * H

    def check_area(mesh: Mesh):
        assert np.abs(area - mesh.area) / area <= 1e-10, "Incorrect surface"

    circle = Circle(Point(L / 2, H / 2), L / 3, meshSize=meshSize)
    circleFilled = Circle(Point(L / 2, H / 2), L / 3, meshSize=meshSize, isFilled=True)
    crack = Line(
        Point(x=0, y=H / 2, isOpen=True),
        Point(x=L / 2, y=H / 2),
        meshSize=meshSize,
        isOpen=False,
    )
    crackOpen = Line(
        Point(x=0, y=H / 2, isOpen=True),
        Point(x=L / 2, y=H / 2),
        meshSize=meshSize,
        isOpen=True,
    )

    meshes: list[Mesh] = []

    for elemType in ElemType.Get_2D():
        mesh1 = _make_mesh_2D(elemType=elemType, isOrganised=False)
        check_area(mesh1)

        mesh2 = _make_mesh_2D(elemType=elemType, isOrganised=True)
        check_area(mesh2)

        # too few elements to properly represent the hole, area not checked
        mesh3 = _make_mesh_2D(elemType=elemType, inclusions=[circle])

        mesh4 = _make_mesh_2D(elemType=elemType, inclusions=[circleFilled])
        check_area(mesh4)

        mesh5 = _make_mesh_2D(elemType=elemType, cracks=[crack])
        check_area(mesh5)

        mesh6 = _make_mesh_2D(elemType=elemType, cracks=[crackOpen])
        check_area(mesh6)

        meshes.extend([mesh1, mesh2, mesh3, mesh4, mesh5, mesh6])

    return meshes


@pytest.fixture(scope="session")
def cracked_meshes_3D() -> list[Mesh]:
    """One mesh per 3D element type for each geometry variant: plain, hollow-circle inclusion, filled-circle inclusion."""

    meshSize = H / 3
    volume = L * H * B

    def check_volume(mesh: Mesh):
        assert np.abs(volume - mesh.volume) / volume <= 1e-10, "Incorrect volume"

    circle = Circle(Point(L / 2, H / 2), H * 0.7, meshSize=meshSize)
    circleFilled = Circle(
        Point(L / 2, H / 2), H * 0.7, meshSize=meshSize, isFilled=True
    )

    meshes: list[Mesh] = []

    for elemType in ElemType.Get_3D():
        mesh1 = _make_mesh_3D(elemType=elemType)
        check_volume(mesh1)

        # too few elements to properly represent the hole, volume not checked
        mesh2 = _make_mesh_3D(elemType=elemType, inclusions=[circle])

        mesh3 = _make_mesh_3D(elemType=elemType, inclusions=[circleFilled])
        check_volume(mesh3)

        meshes.extend([mesh1, mesh2, mesh3])

    return meshes
