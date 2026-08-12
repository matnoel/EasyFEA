# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA import Mesher, ElemType, Mesh
from EasyFEA.FEM._group_elem import GroupElemFactory
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
        mesh = contour.Mesh_2D([], elemType, isOrganised=True)
        meshes.append(mesh)

    # 3d meshes
    for elemType in ElemType.Get_3D():
        mesh = contour.Mesh_Extrude([], [0, 0, L], [3], elemType, isOrganised=True)
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


def test_globalElements_is_sorted():
    """`_globalElements` pairs a whole-mesh array with `connect`, and both consumers do it through
    `np.searchsorted`, which needs sorted input and returns out-of-range indices rather than raising
    when it does not get it. Under MPI `Mesher` builds the partition arrays from python sets, so they
    arrive in hash order — the case that broke `Mesh._Gather` on a rank owning elements but no ghosts.
    """

    mesh = Points([(0, 0), (L, 0), (L, H), (0, H)], H / 3).Mesh_2D()
    groupElem = mesh.groupElem
    Ne = groupElem.Ne

    shuffled = np.random.default_rng(0).permutation(Ne)

    # a rank owning every element of the group and no ghost of it
    groupElem._Set_partitioned_data(shuffled, groupElem.nodes)
    globalElements = groupElem._globalElements

    assert np.array_equal(globalElements, np.arange(Ne))
    assert globalElements.size == Ne

    # owned + ghost, the case that already worked
    owned, ghosts = shuffled[: Ne // 2], shuffled[Ne // 2 :]
    groupElem._Set_partitioned_data(owned, groupElem.nodes, 0, ghosts)
    globalElements = groupElem._globalElements

    assert np.array_equal(globalElements, np.arange(Ne))
    # what `_Gather` does: map an owned global index back to its row in connect
    rows = np.searchsorted(globalElements, np.sort(owned))
    assert np.array_equal(globalElements[rows], np.sort(owned))


def test_globalElements_of_an_empty_group():
    """A rank owning no element of a type still has a group; the index array must stay `int` so it
    remains usable as an index rather than becoming an empty float array."""

    groupElem = GroupElemFactory._Create(
        2, np.empty((0, 3), dtype=int), np.empty((0, 3), dtype=float)
    )

    globalElements = groupElem._globalElements

    assert globalElements.size == 0 == groupElem.Ne
    assert globalElements.dtype.kind == "i"
