# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest
import pyvista as pv

from EasyFEA import ElemType, Mesh, IO

from _numerics import rel_error

from ._meshes import check_mesh, get_tags, main_group_only

# pyvista has no Lagrange triangle: TRI10 and TRI15 go out as their TRI3 sub-triangles
SUBDIVIDED = (ElemType.TRI10, ElemType.TRI15)


def boundary_area(mesh: Mesh) -> float:
    return sum(groupElem.area for groupElem in mesh.Get_list_groupElem(2))


def test_Surface_reconstruction_rebuilds_the_boundary_faces(mesh_of, subtests):
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)

            newMesh = IO.PyVista.Surface_reconstruction(mesh)

            check_mesh(mesh, newMesh)
            for groupElem1, groupElem2 in zip(
                mesh.Get_list_groupElem(2), newMesh.Get_list_groupElem(2)
            ):
                assert groupElem1.Ne == groupElem2.Ne
            assert rel_error(boundary_area(newMesh), boundary_area(mesh)) < 1e-12


def test_Surface_reconstruction_returns_a_2D_mesh_unchanged(make_mesh_2D):
    mesh = make_mesh_2D()

    assert IO.PyVista.Surface_reconstruction(mesh) is mesh


def test_Surface_reconstruction_rebuilds_the_skin_of_a_volume_only_mesh(make_mesh_3D):
    """Faces shared by two elements are inside, so they must not come back."""
    mesh = make_mesh_3D(elemType=ElemType.TETRA4)

    newMesh = IO.PyVista.Surface_reconstruction(main_group_only(mesh))

    assert rel_error(boundary_area(newMesh), boundary_area(mesh)) < 1e-12


def test_Get_pyvista_cell_gives_one_cell_per_element(mesh_of, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem

            cellType, connect = IO.PyVista._Get_pyvista_cell(groupElem)

            if elemType in SUBDIVIDED:
                nTriangles = groupElem.Ne * np.size(groupElem.triangles) // 3
                assert cellType == pv.CellType.TRIANGLE
                assert connect.shape == (nTriangles, 3)
            else:
                pvMesh = IO.PyVista.EasyFEA_to_PyVista(Mesh({elemType: groupElem}))
                assert cellType == pv.CellType(pvMesh.celltypes[0])
                assert connect.shape == groupElem.connect.shape
                assert np.array_equal(
                    np.sort(connect, 1), np.sort(groupElem.connect, 1)
                )


def test_EasyFEA_to_PyVista_round_trips_through_PyVista_to_EasyFEA(mesh_of, subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)

            pvMesh = IO.PyVista.EasyFEA_to_PyVista(mesh)
            newMesh = IO.PyVista.PyVista_to_EasyFEA(pvMesh)

            assert isinstance(pvMesh, pv.UnstructuredGrid)
            if elemType in SUBDIVIDED:
                assert rel_error(newMesh.area, mesh.area) < 1e-12
            else:
                check_mesh(mesh, newMesh)


def test_EasyFEA_to_PyVista_main_group_only_aligns_cells_with_elements(make_mesh_2D):
    mesh = make_mesh_2D()

    pvMesh = IO.PyVista.EasyFEA_to_PyVista(mesh, useAllElements=False)

    assert pvMesh.n_cells == mesh.Ne
    assert np.array_equal(pvMesh.cells_dict[pv.CellType.TRIANGLE], mesh.connect)


def test_EasyFEA_to_PyVista_uses_the_given_coordinates(make_mesh_2D):
    mesh = make_mesh_2D()
    coord = mesh.coord * 2

    pvMesh = IO.PyVista.EasyFEA_to_PyVista(mesh, coord)

    np.testing.assert_array_equal(pvMesh.points, coord)


def test_EasyFEA_to_PyVista_rejects_coordinates_of_the_wrong_shape(make_mesh_2D):
    mesh = make_mesh_2D()

    with pytest.raises(AssertionError, match="coord must be"):
        IO.PyVista.EasyFEA_to_PyVista(mesh, mesh.coord[:, :2])


def test_GroupElem_to_PyVista_numbers_the_group_nodes_locally(make_mesh_2D):
    mesh = make_mesh_2D()
    groupElem = mesh.Get_list_groupElem(1)[0]
    elements = groupElem.Get_Elements_Tag("L1")

    pvMesh = IO.PyVista._GroupElem_to_PyVista(groupElem, elements)

    assert pvMesh.n_cells == elements.size
    np.testing.assert_array_equal(
        pvMesh.points[pvMesh.cells_dict[pv.CellType.LINE]],
        mesh.coord[groupElem.connect[elements]],
    )


def test_PyVista_to_EasyFEA_merges_blocks_and_tags_the_nodes_of_each(make_mesh_2D):
    mesh = make_mesh_2D()
    pvMesh = IO.PyVista.EasyFEA_to_PyVista(mesh, useAllElements=False)

    newMesh = IO.PyVista.PyVista_to_EasyFEA(pv.MultiBlock([pvMesh, pvMesh]))

    assert newMesh.Ne == 2 * mesh.Ne
    assert get_tags(newMesh) == ["0", "1"]


def test_PyVista_to_EasyFEA_rejects_anything_else():
    with pytest.raises(TypeError):
        IO.PyVista.PyVista_to_EasyFEA(pv.PolyData())  # type: ignore [arg-type]
