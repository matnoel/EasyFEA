# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import meshio
import numpy as np

from EasyFEA import ElemType, IO

from ._meshes import check_mesh, get_tags

# serendipity elements have no medit keyword
SERENDIPITY = (ElemType.QUAD8, ElemType.HEXA20, ElemType.PRISM15)


def test_Save_mesh_round_trips_through_Load_mesh(mesh_of, tmp_path, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        if elemType in SERENDIPITY:
            continue
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)

            path = IO.Medit.Save_mesh(mesh, str(tmp_path), "mesh")

            assert path.endswith(".mesh")
            check_mesh(mesh, IO.Medit.Load_mesh(path))


def test_Save_mesh_binary_writes_a_meshb_that_round_trips(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    path = IO.Medit.Save_mesh(mesh, str(tmp_path), "mesh", useBinary=True)

    assert path.endswith(".meshb")
    check_mesh(mesh, IO.Medit.Load_mesh(path))


def test_Save_mesh_writes_no_reference_without_a_converter(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    assert get_tags(mesh) != []

    path = IO.Medit.Save_mesh(mesh, str(tmp_path), "mesh")

    assert get_tags(IO.Medit.Load_mesh(path)) == []


def test_Save_mesh_writes_the_converted_tags_as_references(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    path = IO.Medit.Save_mesh(
        mesh, str(tmp_path), "mesh", dict_tags_converter={"S0": 7}
    )
    newMesh = IO.Medit.Load_mesh(path)

    assert get_tags(newMesh) == ["S7"]
    assert np.array_equal(np.sort(newMesh.Elements_Tags("S7")), np.arange(mesh.Ne))


def test_Load_mesh_keeps_each_reference_on_its_own_elements(tmp_path):
    """The middle triangle's nodes all belong to its neighbours' reference, which must not take it."""

    #  0---2---4
    #   \ / \ /      refs: (0,1,2) -> 1, (1,3,2) -> 2, (2,3,4) -> 1
    #    1---3
    points = np.array([[0, 0], [1, -1], [2, 0], [3, -1], [4, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    refs = np.array([1, 2, 1])
    meshioMesh = meshio.Mesh(
        points, [("triangle", triangles)], cell_data={"medit:ref": [refs]}
    )
    path = str(tmp_path / "refs.mesh")
    meshio.medit.write(path, meshioMesh)

    mesh = IO.Medit.Load_mesh(path)

    assert np.array_equal(np.sort(mesh.Elements_Tags("S1")), [0, 2])
    assert np.array_equal(mesh.Elements_Tags("S2"), [1])


def test_Load_mesh_returns_None_for_a_file_without_elements(tmp_path):
    path = tmp_path / "empty.mesh"
    path.write_text(
        "MeshVersionFormatted 2\nDimension 3\nVertices\n0\nEnd\n", encoding="utf-8"
    )

    assert IO.Medit.Load_mesh(str(path)) is None
