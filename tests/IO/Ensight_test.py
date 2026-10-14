# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import os

import numpy as np
import pytest
import pyvista as pv

from EasyFEA import ElemType, IO

from ._meshes import check_mesh, get_tags, main_group_only


def test_Save_mesh_round_trips_through_Load_mesh(mesh_of, tmp_path, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        if elemType not in IO.Ensight.DICT_ELEMTYPE_TO_ENSIGHT:
            continue
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)

            path = IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh")

            assert path.endswith(".geo")
            check_mesh(mesh, IO.Ensight.Load_mesh(path))


def test_Save_mesh_is_text_only(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()

    with pytest.raises(ValueError):
        IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh", useBinary=True)


def test_Load_mesh_names_each_tag_after_its_part_number(make_mesh_2D, tmp_path):
    mesh = make_mesh_2D()
    path = IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh")

    newMesh = IO.Ensight.Load_mesh(path)

    # the part number is all Load_mesh reads of a part description: S0, L0, P0 all give 0
    assert get_tags(newMesh) == ["0", "1", "2", "3"]
    assert np.array_equal(np.sort(newMesh.Elements_Tags("0")), np.arange(mesh.Ne))


def test_Ensight_to_PyVista_reads_the_mesh_and_removes_its_case_file(
    make_mesh_2D, tmp_path
):
    mesh = main_group_only(make_mesh_2D())
    path = IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh")

    multiBlock = IO.Ensight._Ensight_to_PyVista(path)

    assert isinstance(multiBlock, pv.MultiBlock)
    assert multiBlock.n_blocks == 1
    assert multiBlock.get_block(0).n_cells == mesh.Ne
    assert sorted(os.listdir(tmp_path)) == ["mesh.geo"]


@pytest.mark.xfail(
    strict=True,
    reason="Save_mesh repeats a part header per element type; VTK keeps the last block",
)
def test_Ensight_to_PyVista_reads_every_element_of_a_tagged_mesh(
    make_mesh_2D, tmp_path
):
    mesh = make_mesh_2D()
    path = IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh")

    multiBlock = IO.Ensight._Ensight_to_PyVista(path)

    nCells = sum(multiBlock.get_block(i).n_cells for i in range(multiBlock.n_blocks))
    assert nCells == sum(groupElem.Ne for groupElem in mesh.dict_groupElem.values())


def test_Ensight_to_Meshio_keeps_the_points_and_cells(make_mesh_2D, tmp_path):
    mesh = main_group_only(make_mesh_2D())
    path = IO.Ensight.Save_mesh(mesh, str(tmp_path), "mesh")

    meshioMesh = IO.Ensight._Ensight_to_Meshio(path)

    # .geo coordinates carry 6 significant digits
    np.testing.assert_allclose(meshioMesh.points, mesh.coord, atol=1e-5)
    assert np.array_equal(meshioMesh.cells_dict["triangle"], mesh.connect)
