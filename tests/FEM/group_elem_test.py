# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Callable

import pytest
import numpy as np

from EasyFEA.FEM import (
    ElemType,
    MatrixType,
    Mesh,
    FeArray,
    Gauss,
    GroupElemFactory,
    _GroupElem,
)
from EasyFEA.Geoms import Line, Domain, Circle, Points

from _numerics import affine, mandel, rel_error

from ._reference import MEASURES, REFERENCE_MEASURES, MeshOf, reference_element

ALL = ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D()

# `mesh_of` (conftest): its box spans z in [-2, 0]
CENTERS = {1: (0.5, 0, 0), 2: (1, 0.5, 0), 3: (1, 0.5, -1)}


def _segments(coord: list, elemType: ElemType = ElemType.SEG2) -> _GroupElem:
    """A chain of segments through `coord`."""
    coord = np.asarray(coord, dtype=float)
    connect = np.array([[i, i + 1] for i in range(len(coord) - 1)])
    return GroupElemFactory.Create(elemType, connect, coord)


def _main(mesh: Mesh) -> _GroupElem:
    return mesh.Get_list_groupElem(mesh.dim)[0]


# ----------------------------------------------------------------------------
# construction and properties
# ----------------------------------------------------------------------------


def test_GroupElem_rejects_a_connect_of_the_wrong_width():
    with pytest.raises(AssertionError):
        GroupElemFactory.Create(ElemType.TRI3, np.array([[0, 1]]), np.zeros((3, 3)))


def test_GroupElem_rejects_coordinates_without_three_columns():
    with pytest.raises(AssertionError):
        GroupElemFactory.Create(ElemType.SEG2, np.array([[0, 1]]), np.zeros((2, 2)))


def test_GroupElem_rejects_a_node_without_coordinates():
    with pytest.raises(AssertionError):
        GroupElemFactory.Create(ElemType.SEG2, np.array([[0, 2]]), np.zeros((2, 3)))


def test_GroupElem__InitMatrix_drops_the_cached_matrices():
    groupElem = reference_element(ElemType.QUAD4)
    jacobian = groupElem.Get_jacobian_e_pg(MatrixType.rigi)
    assert groupElem.Get_jacobian_e_pg(MatrixType.rigi) is jacobian

    groupElem._InitMatrix()

    assert groupElem.Get_jacobian_e_pg(MatrixType.rigi) is not jacobian


def test_GroupElem_gmshId_is_the_gmsh_element_number(subtests):
    for elemType, gmshId in [
        (ElemType.POINT, 15),
        (ElemType.SEG2, 1),
        (ElemType.TRI3, 2),
        (ElemType.HEXA27, 12),
    ]:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)

            assert groupElem.gmshId == gmshId


def test_GroupElem_elemType_is_the_created_type():
    assert reference_element(ElemType.QUAD9).elemType is ElemType.QUAD9


def test_GroupElem_topology_is_the_type_without_its_node_count():
    assert reference_element(ElemType.QUAD9).topology == "QUAD"


def test_GroupElem_nPe_counts_every_entity_node():
    groupElem = reference_element(ElemType.QUAD9)

    assert groupElem.nPe == groupElem.connect.shape[1] == 9
    assert groupElem.nPe == (
        groupElem.Nvertex + groupElem.Nedge + groupElem.Nface + groupElem.Nvolume
    )


def test_GroupElem_dim_is_the_reference_dimension():
    groupElem = reference_element(ElemType.QUAD9)

    assert groupElem.dim == groupElem.Get_Local_Coords().shape[1] == 2


def test_GroupElem_order_is_the_polynomial_degree(subtests):
    for elemType, order in [
        (ElemType.SEG5, 4),
        (ElemType.TRI10, 3),
        (ElemType.QUAD8, 2),
        (ElemType.HEXA8, 1),
    ]:
        with subtests.test(elemType=str(elemType)):
            assert reference_element(elemType).order == order


def test_GroupElem_inDim_reads_the_coordinates(subtests):
    for coord, inDim in [
        ([(0, 0, 0), (1, 0, 0)], 1),
        ([(0, 0, 0), (1, 1, 0)], 2),
        ([(0, 0, 0), (1, 0, 1)], 3),
    ]:
        with subtests.test(inDim=inDim):
            assert _segments(coord).inDim == inDim


def test_GroupElem_inDim_of_a_volume_or_an_empty_group():
    empty = GroupElemFactory.Create(
        ElemType.TRI3, np.empty((0, 3), dtype=int), np.empty((0, 3))
    )

    assert reference_element(ElemType.TETRA4, b=(0, 0, 0)).inDim == 3
    assert empty.inDim == 2


def test_GroupElem_Ne_counts_the_connect_rows():
    assert _segments([(0, 0, 0), (1, 0, 0), (2, 0, 0)]).Ne == 2


def test_GroupElem_nodes_are_the_used_ones_sorted():
    connect = np.array([[4, 2]])
    groupElem = GroupElemFactory.Create(ElemType.SEG2, connect, np.zeros((5, 3)))

    assert np.array_equal(groupElem.nodes, [2, 4])
    assert groupElem.Nn == 2 and groupElem.Ncoords == 5


def test_GroupElem_elements_are_the_connect_rows():
    groupElem = _segments([(0, 0, 0), (1, 0, 0), (2, 0, 0)])

    assert np.array_equal(groupElem.elements, [0, 1])


def test_GroupElem_Nn_counts_the_used_nodes():
    connect = np.array([[3, 1]])

    assert GroupElemFactory.Create(ElemType.SEG2, connect, np.zeros((6, 3))).Nn == 2


def test_GroupElem_Ncoords_counts_the_mesh_coordinates():
    connect = np.array([[3, 1]])

    assert (
        GroupElemFactory.Create(ElemType.SEG2, connect, np.zeros((6, 3))).Ncoords == 6
    )


def test_GroupElem_coord_holds_the_used_nodes_and_is_set_from_the_mesh():
    coord = np.arange(12.0).reshape(4, 3)
    groupElem = GroupElemFactory.Create(ElemType.SEG2, np.array([[3, 1]]), coord)
    assert np.array_equal(groupElem.coord, coord[[1, 3]])
    length = groupElem.length

    groupElem.coord = 2 * coord

    assert np.array_equal(groupElem.coord, 2 * coord[[1, 3]])
    assert rel_error(groupElem.length, 2 * length) < 1e-12
    with pytest.raises(AssertionError):
        groupElem.coord = coord[:2]


def test_GroupElem_Nvertex_counts_the_corners():
    assert reference_element(ElemType.QUAD9).Nvertex == 4


def test_GroupElem_Nedge_counts_the_edge_nodes(subtests):
    for elemType, Nedge in [
        (ElemType.SEG4, 2),
        (ElemType.TRI6, 3),
        (ElemType.TRI15, 9),
        (ElemType.HEXA20, 12),
    ]:
        with subtests.test(elemType=str(elemType)):
            assert reference_element(elemType).Nedge == Nedge


def test_GroupElem_Nface_counts_the_face_nodes(subtests):
    for elemType, Nface in [
        (ElemType.TRI6, 0),
        (ElemType.TRI10, 1),
        (ElemType.QUAD9, 1),
        (ElemType.HEXA27, 6),
    ]:
        with subtests.test(elemType=str(elemType)):
            assert reference_element(elemType).Nface == Nface


def test_GroupElem_Nvolume_counts_the_volume_nodes(subtests):
    for elemType, Nvolume in [
        (ElemType.HEXA20, 0),
        (ElemType.HEXA27, 1),
        (ElemType.PRISM18, 0),
    ]:
        with subtests.test(elemType=str(elemType)):
            assert reference_element(elemType).Nvolume == Nvolume


def test_GroupElem_connect_is_a_copy():
    groupElem = _segments([(0, 0, 0), (1, 0, 0)])

    groupElem.connect[:] = -1

    assert np.array_equal(groupElem.connect, [[0, 1]])


# ----------------------------------------------------------------------------
# partition
# ----------------------------------------------------------------------------


def test_GroupElem__Set_partitioned_data_sorts_and_derives_the_ghost_nodes():
    groupElem = _segments([(i, 0, 0) for i in range(5)])

    groupElem._Set_partitioned_data([3, 1], [2, 0, 1], 1, [2])

    rank, elements, ghostElements, nodes, ghostNodes = groupElem._Get_partitioned_data()
    assert rank == 1
    assert np.array_equal(elements, [1, 3])
    assert np.array_equal(ghostElements, [2])
    assert np.array_equal(nodes, [0, 1, 2])
    assert np.array_equal(ghostNodes, [3, 4])


def test_GroupElem__Get_partitioned_data_of_a_serial_group_owns_everything():
    groupElem = _segments([(i, 0, 0) for i in range(4)])

    rank, elements, ghostElements, nodes, ghostNodes = groupElem._Get_partitioned_data()

    assert rank == 0
    assert np.array_equal(elements, groupElem.elements)
    assert np.array_equal(nodes, groupElem.nodes)
    assert ghostElements.size == ghostNodes.size == 0


def test_GroupElem__globalElements_is_sorted(make_mesh_2D: Callable[..., Mesh]):
    """Mesher builds partition arrays from python sets (hash order); consumers pair them with `connect` through `np.searchsorted`, which needs them sorted."""
    mesh = make_mesh_2D()
    groupElem = mesh.groupElem
    Ne = groupElem.Ne

    shuffled = np.random.default_rng(0).permutation(Ne)

    # a rank owning every element of the group and no ghost of it
    groupElem._Set_partitioned_data(shuffled, groupElem.nodes)
    assert np.array_equal(groupElem._globalElements, np.arange(Ne))

    # owned + ghost
    owned, ghosts = shuffled[: Ne // 2], shuffled[Ne // 2 :]
    groupElem._Set_partitioned_data(owned, groupElem.nodes, 0, ghosts)
    globalElements = groupElem._globalElements

    assert np.array_equal(globalElements, np.arange(Ne))
    rows = np.searchsorted(globalElements, np.sort(owned))
    assert np.array_equal(globalElements[rows], np.sort(owned))


def test_GroupElem__globalElements_of_an_empty_group_stays_int():
    groupElem = GroupElemFactory.Create(
        ElemType.TRI3, np.empty((0, 3), dtype=int), np.empty((0, 3))
    )

    globalElements = groupElem._globalElements

    assert globalElements.size == 0 == groupElem.Ne
    assert globalElements.dtype.kind == "i"


def test_GroupElem__rankPartition_is_set_from_an_array():
    groupElem = _segments([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert groupElem._rankPartition is None

    groupElem._rankPartition = np.array([0, 1])

    assert np.array_equal(groupElem._rankPartition, [0, 1])
    with pytest.raises(AssertionError):
        groupElem._rankPartition = [0, 1]  # type: ignore [assignment]


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------


def test_GroupElem_Get_connect_n_e_counts_the_elements_of_each_node():
    groupElem = _segments([(i, 0, 0) for i in range(4)])

    connect_n_e = groupElem.Get_connect_n_e()

    assert connect_n_e.shape == (4, 3)
    assert np.array_equal(connect_n_e @ np.ones(3), [1, 2, 2, 1])


def test_GroupElem_Get_assembly_e_interleaves_the_dofs_of_each_node(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))
            connect = groupElem.connect

            assembly_e = groupElem.Get_assembly_e(3)

            assert np.array_equal(assembly_e[:, 0::3], 3 * connect)
            assert np.array_equal(assembly_e[:, 2::3], 3 * connect + 2)


def test_GroupElem__Get_assembly_e_takes_any_connect():
    assembly_e = _GroupElem._Get_assembly_e(np.array([[2, 0]]), 2)

    assert np.array_equal(assembly_e, [[4, 5, 0, 1]])


def test_GroupElem_Get_rows_e_repeats_each_dof_along_the_row(mesh_of: MeshOf, subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))
            assembly_e = groupElem.Get_assembly_e(2)
            ndof = assembly_e.shape[1]

            rows_e = groupElem.Get_rows_e(2).reshape(-1, ndof, ndof)

            expected = np.repeat(assembly_e[:, :, None], ndof, axis=2)
            assert np.array_equal(rows_e, expected)


def test_GroupElem_Get_columns_e_repeats_the_dofs_down_the_column(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))
            assembly_e = groupElem.Get_assembly_e(2)
            ndof = assembly_e.shape[1]

            columns_e = groupElem.Get_columns_e(2).reshape(-1, ndof, ndof)

            expected = np.repeat(assembly_e[:, None, :], ndof, axis=1)
            assert np.array_equal(columns_e, expected)


def test_GroupElem_Get_rows_e_and_Get_columns_e_of_an_empty_group():
    groupElem = GroupElemFactory.Create(
        ElemType.TRI3, np.empty((0, 3), dtype=int), np.empty((0, 3))
    )

    assert groupElem.Get_rows_e(2).shape == groupElem.Get_columns_e(2).shape == (0, 36)


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


def test_GroupElem__Get_sysCoord_e_of_a_segment_is_orthonormal_along_it(subtests):
    for name, coord in {
        "x axis": [(0, 0, 0), (1, 0, 0)],
        "plane": [(0, 0, 0), (1, 2, 0)],
        "space": [(0, 0, 0), (1, 2, 3)],
        "space along x": [(0, 0, 0), (2, 0, 0), (2, 0, 3)],
    }.items():
        with subtests.test(coord=name):
            groupElem = _segments(coord)

            P_e = groupElem._Get_sysCoord_e()

            identity = np.broadcast_to(np.eye(3), P_e.shape)
            assert rel_error(np.swapaxes(P_e, 1, 2) @ P_e, identity) < 1e-12
            tangents = np.diff(np.asarray(coord, dtype=float), axis=0)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
            assert rel_error(P_e[:, :, 0], tangents) < 1e-12


def test_GroupElem__Get_sysCoord_e_of_a_face_has_k_normal_to_it(subtests):
    """A face in the plane x + y + z = cst: k = ±(1, 1, 1)/√3."""
    A = np.array([[1, 0], [0, 1], [-1, -1]])
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType, A)

            P_e = groupElem._Get_sysCoord_e(np.ones((groupElem.Ncoords, 3)))

            assert rel_error(np.swapaxes(P_e, 1, 2) @ P_e, np.eye(3)[None]) < 1e-12
            assert rel_error(np.abs(P_e[0, :, 2]), np.ones(3) / np.sqrt(3)) < 1e-12


def test_GroupElem__Get_sysCoord_e_of_a_point_or_volume_is_the_global_frame(subtests):
    for elemType in [ElemType.POINT, ElemType.HEXA8]:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)

            assert np.array_equal(groupElem._Get_sysCoord_e(), np.eye(3)[None])


def _enclosed_measure(mesh: Mesh, displacement: np.ndarray | None = None) -> float:
    """Measure enclosed by the inDim - 1 groups, from the divergence theorem: ∫ x·n dΓ / inDim."""
    measure = 0.0
    for groupElem in mesh.Get_list_groupElem(mesh.inDim - 1):
        x_e_pg = groupElem.Get_GaussCoordinates_e_pg(
            MatrixType.mass, displacementMatrix=displacement
        )
        normal_e_pg = groupElem.Get_normals_e_pg(
            MatrixType.mass, displacement, normalize=False
        )
        weight_e_pg = FeArray.from_pg(
            groupElem.Get_weight_pg(MatrixType.mass), groupElem.Ne
        )
        measure += float((x_e_pg.dot(normal_e_pg) * weight_e_pg).integrate().sum())
    return measure / mesh.inDim


# a flat-sided contour and a curved one, built per case: meshing mutates a geometry
CONTOURS = {
    "Points": lambda: Points([(0, 0), (2, 0), (2, 1), (0, 1)], 1 / 3),
    "Circle": lambda: Circle((1, 0.5), 2, 0.5),
}


def test_GroupElem_Get_normals_e_pg_enclose_the_area(subtests):
    """1D normals point inward on a counterclockwise contour."""
    for elemType in ElemType.Get_2D():
        for isOrganised in [True, False]:
            for name, make_contour in CONTOURS.items():
                with subtests.test(
                    elemType=str(elemType), isOrganised=isOrganised, contour=name
                ):
                    contour = make_contour()
                    mesh = contour.Mesh_2D([], elemType, isOrganised=isOrganised)
                    mesh.Rotate(30, direction=(0, 0, 1))
                    mesh.Translate(-1, 2)

                    assert rel_error(-_enclosed_measure(mesh), mesh.area) < 1e-13


def test_GroupElem_Get_normals_e_pg_enclose_the_volume(subtests):
    for elemType in ElemType.Get_3D():
        for extrude in [(0, 0, 1), (0, 0, -1)]:
            for name, make_contour in CONTOURS.items():
                with subtests.test(
                    elemType=str(elemType), extrude=extrude, contour=name
                ):
                    contour = make_contour()
                    mesh = contour.Mesh_Extrude(
                        [], extrude, [3], elemType, isOrganised=True
                    )
                    mesh.Rotate(45, direction=(1, 0, 0))
                    mesh.Rotate(45, direction=(0, 1, 0))
                    mesh.Translate(-1, 2, 3)

                    assert rel_error(_enclosed_measure(mesh), mesh.volume) < 1e-13


def test_GroupElem_Get_normals_e_pg_follow_the_displacement(subtests):
    """Stretching x by 2 doubles the enclosed volume, read on the deformed configuration."""
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            contour = Domain((0, 0), (1, 1), 0.5)
            mesh = contour.Mesh_Extrude([], (0, 0, 1), [2], elemType, isOrganised=True)
            displacement = np.zeros_like(mesh.coord)
            displacement[:, 0] = mesh.coord[:, 0]

            assert rel_error(_enclosed_measure(mesh, displacement), 2.0) < 1e-12
            unit_e_pg = mesh.Get_list_groupElem(2)[0].Get_normals_e_pg(MatrixType.mass)
            norms = np.linalg.norm(np.asarray(unit_e_pg), axis=-1)
            assert rel_error(norms, 1.0) < 1e-12


def test_GroupElem_Get_normals_e_pg_rejects_a_volume():
    with pytest.raises(AssertionError):
        reference_element(ElemType.HEXA8).Get_normals_e_pg(MatrixType.mass)


def test_GroupElem__Get_gap_and_normal_measures_along_the_outward_normal(subtests):
    """Points pushed off the circle by t along the outward normal sit at gap t."""
    for elemType in ElemType.Get_1D():
        with subtests.test(elemType=str(elemType)):
            circle = Circle((1, 2), 2, 0.4)
            mesh = circle.Mesh_2D(
                [], ElemType.TRI3 if elemType == ElemType.SEG2 else ElemType.TRI6
            )
            surface = mesh.Get_list_groupElem(1)[0]
            samples_e_pg = surface.Get_GaussCoordinates_e_pg(MatrixType.mass)
            outward_e_pg = np.asarray(samples_e_pg) - [1, 2, 0]
            outward_e_pg /= np.linalg.norm(outward_e_pg, axis=-1, keepdims=True)

            for t in [0.1, -0.05]:
                normal_e_pg = surface._Get_gap_and_normal(samples_e_pg)[1]
                x_e_pg = samples_e_pg + t * normal_e_pg
                gap_e_pg, _ = surface._Get_gap_and_normal(
                    x_e_pg, coord=np.array([1, 2, 0])
                )

                assert rel_error(np.asarray(gap_e_pg), t) < 1e-12
            # the normals point away from the center
            assert np.all(np.einsum("epi,epi->ep", normal_e_pg, outward_e_pg) > 0.9)


def test_GroupElem__Get_gap_and_normal_on_a_patch():
    domain = Domain((0, 0), (2, 1), 0.5)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    surface = mesh.Get_list_groupElem(1)[0]
    bottom = surface.Get_Elements_Tag("L0")
    x_e_pg = FeArray.asfearray(np.array([[[0.6, -0.3, 0.0]]]))

    gap_e_pg, normal_e_pg = surface._Get_gap_and_normal(x_e_pg, bottom)

    assert rel_error(np.asarray(gap_e_pg), 0.3) < 1e-12
    assert (
        rel_error(np.asarray(normal_e_pg), np.array([[[0, -1, 0]]]), scale=1.0) < 1e-12
    )


def test_GroupElem_Integrate_e_integrates_x_over_the_mesh(mesh_of: MeshOf, subtests):
    """∫ x dΩ = measure · x̄."""
    for elemType in ALL:
        for matrixType in [MatrixType.rigi, MatrixType.mass]:
            with subtests.test(elemType=str(elemType), matrixType=matrixType):
                groupElem = _main(mesh_of(elemType))
                dim = groupElem.dim

                values_e = groupElem.Integrate_e(lambda x, y, z: x, matrixType)

                assert values_e.shape == (groupElem.Ne,)
                assert (
                    rel_error(values_e.sum(), MEASURES[dim] * CENTERS[dim][0]) < 1e-12
                )
                assert rel_error(groupElem.Integrate_e().sum(), MEASURES[dim]) < 1e-12


def test_GroupElem_length_e_sums_to_length(mesh_of: MeshOf, subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.Get_list_groupElem(1)[0]
            if mesh.dim == 1:
                assert rel_error(groupElem.length, 1.0) < 1e-12
            assert rel_error(groupElem.length_e.sum(), groupElem.length) < 1e-12
            assert _main(mesh).length_e is None or mesh.dim == 1


def test_GroupElem_length_is_None_off_lines():
    assert reference_element(ElemType.QUAD9).length is None


def test_GroupElem_area_e_sums_to_area(mesh_of: MeshOf, subtests):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.Get_list_groupElem(2)[0]
            if mesh.dim == 2:
                assert rel_error(groupElem.area, 2.0) < 1e-12
            assert rel_error(groupElem.area_e.sum(), groupElem.area) < 1e-12


def test_GroupElem_area_is_None_off_surfaces():
    groupElem = reference_element(ElemType.SEG3)

    assert groupElem.area is None
    assert groupElem.area_e is None


def test_GroupElem_volume_e_sums_to_volume(mesh_of: MeshOf, subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))
            if groupElem.dim != 3:
                assert groupElem.volume_e is None
            else:
                assert rel_error(groupElem.volume_e.sum(), groupElem.volume) < 1e-12


def test_GroupElem_volume_is_the_box_volume(mesh_of: MeshOf, subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))

            if groupElem.dim == 3:
                assert rel_error(groupElem.volume, 4.0) < 1e-12
            else:
                assert groupElem.volume is None


def test_GroupElem_center_is_the_barycenter(mesh_of: MeshOf, subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))

            center = CENTERS[groupElem.dim]
            assert rel_error(groupElem.center, center, scale=1.0) < 1e-12


def _point() -> _GroupElem:
    return GroupElemFactory.Create(ElemType.POINT, np.array([[0]]), np.ones((1, 3)))


def test_GroupElem_origin_is_the_first_node(subtests):
    """POINT overrides it."""
    for groupElem in [reference_element(ElemType.QUAD9), _point()]:
        with subtests.test(elemType=str(groupElem.elemType)):
            first = groupElem.Get_Local_Coords()[0]

            assert np.array_equal(np.broadcast_to(groupElem.origin, first.shape), first)


def test_GroupElem_triangles_tile_the_reference_element(subtests):
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            local = groupElem.Get_Local_Coords()

            a, b, c = (
                local[np.asarray(groupElem.triangles).reshape(-1, 3)[:, i]]
                for i in range(3)
            )
            u, v = b - a, c - a
            areas = (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2

            assert np.all(areas > 0)
            reference = REFERENCE_MEASURES[elemType.topology]
            assert rel_error(areas.sum(), reference) < 1e-12


def test_GroupElem_segments_join_two_vertices_through_the_edge_nodes(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            local = groupElem.Get_Local_Coords()
            segments = np.asarray(groupElem.segments)

            assert np.all(segments[:, [0, -1]] < groupElem.Nvertex)
            for segment in segments:
                start, end = local[segment[0]], local[segment[-1]]
                t = np.linspace(0, 1, segment.size)[:, None]
                expected = start + t * (end - start)
                assert rel_error(local[segment], expected, scale=1.0) < 1e-12


def test_GroupElem_segments_of_a_point_is_none():
    groupElem = GroupElemFactory.Create(
        ElemType.POINT, np.array([[0]]), np.zeros((1, 3))
    )

    assert groupElem.segments is None


def test_GroupElem_edges_put_the_vertices_first(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            segments = np.asarray(groupElem.segments)

            edges = groupElem.edges

            assert np.array_equal(edges[:, :2], segments[:, [0, -1]])
            assert np.array_equal(edges[:, 2:], segments[:, 1:-1])


def _face_normal(local: np.ndarray, face) -> np.ndarray:
    """Normal of the face through its first three nodes, its vertices."""
    a, b, c = local[list(face[:3])]
    return np.cross(b - a, c - a)


def _polygon_normal(local: np.ndarray, contour) -> np.ndarray:
    """Newell normal of the closed polygon through `contour`, whatever its edge nodes."""
    points = local[list(contour)]
    return np.cross(points, np.roll(points, -1, axis=0)).sum(0) / 2


def test_GroupElem_surfaces_of_a_volume_point_outward(subtests):
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            local = groupElem.Get_Local_Coords()
            centroid = local.mean(0)

            for surface in groupElem.surfaces:
                normal = _polygon_normal(local, surface)
                assert np.dot(normal, centroid - local[surface[0]]) < 0


def test_GroupElem_surfaces_of_a_line_or_face_go_around_its_contour(subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)

            surface = np.asarray(groupElem.surfaces).ravel()

            if groupElem.dim == 1:
                assert np.array_equal(surface, groupElem.segments[0])
            else:
                assert surface[0] == surface[-1]
                edges = sorted(groupElem.segments[:, :-1].ravel())
                assert sorted(surface[:-1]) == edges


def test_GroupElem_faces_of_a_volume_point_outward(subtests):
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            local = groupElem.Get_Local_Coords()
            centroid = local.mean(0)

            for face in groupElem.faces:
                normal = _face_normal(local, face)
                assert np.dot(normal, centroid - local[face[0]]) < 0


def test_GroupElem_faces_of_a_face_start_counterclockwise(subtests):
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            local = np.pad(groupElem.Get_Local_Coords(), ((0, 0), (0, 1)))

            assert sorted(groupElem.faces) == list(range(groupElem.nPe))
            assert _face_normal(local, groupElem.faces)[2] > 0


def test_GroupElem_faces_of_a_line_is_empty():
    assert reference_element(ElemType.SEG3).faces.size == 0


def test_GroupElem_Get_Local_Coords_lie_in_the_reference_element(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            local = reference_element(elemType).Get_Local_Coords()

            gauss = Gauss(elemType, MatrixType.mass).coord
            low = np.minimum(local.min(0), gauss.min(0))
            high = np.maximum(local.max(0), gauss.max(0))
            assert np.array_equal(low, local.min(0))
            assert np.array_equal(high, local.max(0))
            assert len({tuple(xi) for xi in local}) == local.shape[0]


# ----------------------------------------------------------------------------
# Gauss
# ----------------------------------------------------------------------------


def test_GroupElem_Get_gauss_is_the_rule_of_the_element(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            gauss = reference_element(elemType).Get_gauss(MatrixType.rigi)

            assert np.array_equal(gauss.coord, Gauss(elemType, MatrixType.rigi).coord)


def test_GroupElem_Get_weight_pg_sum_to_the_reference_measure(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            weights = reference_element(elemType).Get_weight_pg(MatrixType.mass)

            reference = REFERENCE_MEASURES[elemType.topology]
            assert rel_error(weights.sum(), reference) < 1e-12


def test_GroupElem_Get_GaussCoordinates_e_pg_maps_the_rule_affinely(subtests):
    """x = A ξ + b at every Gauss point, displaced by `displacementMatrix` if given."""
    b = np.array([1.0, -2.0, 3.0])
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            dim = reference_element(elemType).dim
            A = np.random.default_rng(dim).random((3, dim)) + np.eye(3)[:, :dim]
            groupElem = reference_element(elemType, A, b)
            xi_pg = groupElem.Get_gauss(MatrixType.mass).coord

            x_e_pg = groupElem.Get_GaussCoordinates_e_pg(MatrixType.mass)
            moved_e_pg = groupElem.Get_GaussCoordinates_e_pg(
                MatrixType.mass, displacementMatrix=np.ones((groupElem.Ncoords, 3))
            )
            first_e_pg = groupElem.Get_GaussCoordinates_e_pg(
                MatrixType.mass, np.array([0])
            )

            assert rel_error(np.asarray(x_e_pg)[0], xi_pg @ A.T + b) < 1e-12
            assert rel_error(np.asarray(moved_e_pg - x_e_pg), 1.0) < 1e-12
            assert np.array_equal(first_e_pg, x_e_pg)


def test_GroupElem_Interpolate_e_pg_is_exact_on_an_affine_field(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            coord = mesh.coord
            x_e_pg = np.asarray(groupElem.Get_GaussCoordinates_e_pg(MatrixType.rigi))

            scalar_e_pg = groupElem.Interpolate_e_pg(1 + coord @ [1, 2, 3])
            vector_e_pg = groupElem.Interpolate_e_pg(coord[:, :2].ravel())

            assert rel_error(np.asarray(scalar_e_pg), 1 + x_e_pg @ [1, 2, 3]) < 1e-12
            vector = np.asarray(vector_e_pg)
            assert rel_error(vector, x_e_pg[..., :2], scale=1.0) < 1e-12


# ----------------------------------------------------------------------------
# isoparametric mapping
# ----------------------------------------------------------------------------

AFFINE = {
    1: np.array([[2.0], [0], [0]]),
    2: np.array([[2.0, 1], [0.5, 3], [0, 0]]),
    3: np.array([[2.0, 1, 0], [0.5, 3, 0], [0, 1, -4]]),
}


def _affine(elemType: ElemType) -> tuple[_GroupElem, np.ndarray]:
    dim = reference_element(elemType).dim
    A = AFFINE[dim]
    return reference_element(elemType, A, b=(1, 2, 0)), A[:dim, :dim]


def test_GroupElem__Get_J_e_pg_is_the_transposed_affine_map(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem, A = _affine(elemType)

            J_e_pg = groupElem._Get_J_e_pg(MatrixType.rigi)

            expected = np.broadcast_to(A.T, J_e_pg.shape)
            assert rel_error(np.asarray(J_e_pg), expected) < 1e-12


def test_GroupElem__Get_J_e_pg_of_a_face_in_space_keeps_its_area(subtests):
    """A unit-jacobian face rotated out of the plane."""
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[1, 0], [0, c], [0, s]])
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType, R)

            jacobian_e_pg = groupElem.Get_jacobian_e_pg(MatrixType.rigi)

            assert groupElem.inDim == 3
            assert rel_error(np.asarray(jacobian_e_pg), 1.0) < 1e-12


def test_GroupElem__Get_J_e_pg_of_a_point_is_none():
    groupElem = GroupElemFactory.Create(
        ElemType.POINT, np.array([[0]]), np.zeros((1, 3))
    )

    assert groupElem._Get_J_e_pg(MatrixType.rigi) is None
    assert groupElem.Get_jacobian_e_pg(MatrixType.rigi) is None
    assert groupElem.Get_weightedJacobian_e_pg(MatrixType.rigi) is None
    assert groupElem._Get_invJ_e_pg(MatrixType.rigi) is None


def test_GroupElem_Get_jacobian_e_pg_is_the_determinant_of_the_map(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem, A = _affine(elemType)
            det = np.linalg.det(A)

            jacobian_e_pg = groupElem.Get_jacobian_e_pg(MatrixType.rigi)
            signed_e_pg = groupElem.Get_jacobian_e_pg(
                MatrixType.rigi, absoluteValues=False
            )

            assert rel_error(np.asarray(jacobian_e_pg), abs(det)) < 1e-12
            assert rel_error(np.asarray(signed_e_pg), det) < 1e-12


def test_GroupElem_Get_weightedJacobian_e_pg_sums_to_the_measure(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = _main(mesh_of(elemType))

            wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(MatrixType.mass)

            assert rel_error(wJ_e_pg.sum(), MEASURES[groupElem.dim]) < 1e-12


def test_GroupElem__Get_invJ_e_pg_inverts_J(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem, A = _affine(elemType)

            invJ_e_pg = groupElem._Get_invJ_e_pg(MatrixType.rigi)

            expected = np.broadcast_to(np.linalg.inv(A.T), invJ_e_pg.shape)
            assert rel_error(np.asarray(invJ_e_pg), expected) < 1e-12


# ----------------------------------------------------------------------------
# shape functions
# ----------------------------------------------------------------------------


def test_GroupElem__Eval_Functions_evaluates_each_function_at_each_point():
    functions = np.array(
        [[lambda x, y: x, lambda x, y: y], [lambda x, y: x * y, lambda x, y: 1]]
    )
    points = np.array([[1.0, 2.0], [3.0, 4.0]])

    values = _GroupElem._Eval_Functions(functions, points)

    assert values.shape == (2, 2, 2)
    assert np.array_equal(values[1], [[3, 12], [4, 1]])


def test_GroupElem__Init_Functions_gives_zeros_above_the_element_order(subtests):
    for elemType in [ElemType.SEG2, ElemType.TRI3, ElemType.TETRA4]:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            point = groupElem.Get_Local_Coords()[:1]

            functions = groupElem._Init_Functions(2)

            assert functions.shape == (groupElem.nPe, groupElem.dim)
            assert not np.any(_GroupElem._Eval_Functions(functions, point))
            with pytest.raises(TypeError):
                groupElem._Init_Functions(1)


def test_GroupElem__N_is_the_kronecker_delta_at_the_nodes():
    """N_i(ξ_j) = δ_ij on the reference element."""
    groupElem = reference_element(ElemType.QUAD9)

    values = _GroupElem._Eval_Functions(groupElem._N(), groupElem.Get_Local_Coords())

    assert rel_error(values[:, 0], np.eye(groupElem.nPe)) < 1e-12


def test_GroupElem_Get_N_pg_is_a_partition_of_unity():
    N_pg = reference_element(ElemType.QUAD9).Get_N_pg(MatrixType.mass)

    assert rel_error(N_pg.sum(-1), 1.0) < 1e-12


def test_GroupElem_Get_N_pg_rep_is_block_diagonal(subtests):
    groupElem = reference_element(ElemType.TRI6)
    N_pg = groupElem.Get_N_pg(MatrixType.mass)
    for repeat in [1, 2, 3]:
        with subtests.test(repeat=repeat):
            N_rep = groupElem.Get_N_pg_rep(MatrixType.mass, repeat)

            assert N_rep.shape == (N_pg.shape[0], repeat, repeat * groupElem.nPe)
            for r in range(repeat):
                assert np.array_equal(N_rep[:, r, r::repeat], N_pg[:, 0])
                others = np.delete(
                    N_rep[:, r], np.arange(r, N_rep.shape[2], repeat), axis=1
                )
                assert not np.any(others)


def test_GroupElem_Get_N_pg_rep_of_a_point_is_none():
    groupElem = GroupElemFactory.Create(
        ElemType.POINT, np.array([[0]]), np.zeros((1, 3))
    )

    assert groupElem.Get_N_pg_rep(MatrixType.mass, 2) is None


def test_GroupElem__dN_sums_to_zero_at_the_nodes():
    groupElem = reference_element(ElemType.QUAD9)

    dN_n = _GroupElem._Eval_Functions(groupElem._dN(), groupElem.Get_Local_Coords())

    assert np.abs(dN_n.sum(-1)).max() < 1e-12


def test_GroupElem_Get_dN_pg_sums_to_zero():
    dN_pg = reference_element(ElemType.QUAD9).Get_dN_pg(MatrixType.mass)

    assert np.abs(dN_pg.sum(-1)).max() < 1e-12


def test_GroupElem_Get_dN_e_pg_differentiates_the_coordinates_to_the_identity(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            dim = groupElem.dim
            coord_e = mesh.coord[groupElem.connect][..., :dim]

            dN_e_pg = groupElem.Get_dN_e_pg(MatrixType.rigi)

            gradient = np.einsum("epdn,enj->epdj", dN_e_pg, coord_e)
            identity = np.broadcast_to(np.eye(dim), gradient.shape)
            assert rel_error(gradient, identity, scale=1.0) < 1e-10


def test_GroupElem__ddN_sums_to_zero_at_the_nodes():
    groupElem = reference_element(ElemType.QUAD9)

    ddN_n = _GroupElem._Eval_Functions(groupElem._ddN(), groupElem.Get_Local_Coords())

    assert np.abs(ddN_n.sum(-1)).max() < 1e-10


def test_GroupElem_Get_ddN_pg_sums_to_zero():
    ddN_pg = reference_element(ElemType.QUAD9).Get_ddN_pg(MatrixType.mass)

    assert np.abs(ddN_pg.sum(-1)).max() < 1e-10


def test_GroupElem_Get_ddN_e_pg_differentiates_a_square_twice(subtests):
    """∂²(x²)/∂x² = 2 on an axis-aligned element twice the reference size; x² needs order ≥ 2."""
    for elemType in [e for e in ALL if reference_element(e).order >= 2]:
        with subtests.test(elemType=str(elemType)):
            dim = reference_element(elemType).dim
            groupElem = reference_element(elemType, 2 * np.eye(3)[:, :dim], (1, 2, 3))
            x_n = groupElem.coord[:, 0]

            ddN_e_pg = groupElem.Get_ddN_e_pg(MatrixType.mass)

            values = np.asarray(ddN_e_pg) @ x_n**2
            expected = np.zeros(dim)
            expected[0] = 2
            expected = np.broadcast_to(expected, values.shape)
            assert rel_error(values, expected, scale=2.0) < 1e-10


def test_GroupElem__dddN_sums_to_zero_at_the_nodes():
    groupElem = reference_element(ElemType.QUAD9)

    dddN_n = _GroupElem._Eval_Functions(groupElem._dddN(), groupElem.Get_Local_Coords())

    assert np.abs(dddN_n.sum(-1)).max() < 1e-9


def test_GroupElem_Get_dddN_pg_sums_to_zero():
    dddN_pg = reference_element(ElemType.QUAD9).Get_dddN_pg(MatrixType.mass)

    assert np.abs(dddN_pg.sum(-1)).max() < 1e-9


def test_GroupElem__ddddN_sums_to_zero_at_the_nodes():
    groupElem = reference_element(ElemType.QUAD9)

    ddddN_n = _GroupElem._Eval_Functions(
        groupElem._ddddN(), groupElem.Get_Local_Coords()
    )

    assert np.abs(ddddN_n.sum(-1)).max() < 1e-9


def test_GroupElem_Get_ddddN_pg_sums_to_zero():
    ddddN_pg = reference_element(ElemType.QUAD9).Get_ddddN_pg(MatrixType.mass)

    assert np.abs(ddddN_pg.sum(-1)).max() < 1e-9


def test_GroupElem_Get_dddN_pg_and_others_of_a_point_are_none(subtests):
    groupElem = GroupElemFactory.Create(
        ElemType.POINT, np.array([[0], [1]]), np.array([[0.0, 0, 0], [1, 0, 0]])
    )
    for name in ["Get_dddN_pg", "Get_ddddN_pg", "Get_ddN_pg", "Get_N_pg"]:
        with subtests.test(name=name):
            assert getattr(groupElem, name)(MatrixType.rigi) is None


# ----------------------------------------------------------------------------
# finite element matrices
# ----------------------------------------------------------------------------

DIM_2_3 = ElemType.Get_2D() + ElemType.Get_3D()


def _strain(dim: int) -> np.ndarray:
    E = np.array([[1.0, 0.2, -0.3], [0.2, -0.5, 0.4], [-0.3, 0.4, 2.0]])
    return E[:dim, :dim]


def _affine_displacement_e(mesh: Mesh, groupElem: _GroupElem, E: np.ndarray):
    """u = E x, as element dofs (Ne, nPe·dim)."""
    return affine(mesh.coord, E)[groupElem.Get_assembly_e(E.shape[0])]


def test_GroupElem_Get_B_e_pg_gives_the_mandel_strain_of_an_affine_displacement(
    mesh_of: MeshOf, subtests
):
    for elemType in DIM_2_3:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            E = _strain(groupElem.dim)
            u_e = _affine_displacement_e(mesh, groupElem, E)

            B_e_pg = groupElem.Get_B_e_pg(MatrixType.rigi)

            strain_e_pg = np.einsum("epij,ej->epi", B_e_pg, u_e)
            expected = np.broadcast_to(mandel(E), strain_e_pg.shape)
            assert rel_error(strain_e_pg, expected) < 1e-10


def test_GroupElem_Get_leftDispPart_e_pg_integrates_the_strain_work(
    mesh_of: MeshOf, subtests
):
    """uᵀ ∫ Bᵀ σ dΩ = measure · ε:σ for an affine u and a constant σ."""
    for elemType in DIM_2_3:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            dim = groupElem.dim
            E = _strain(dim)
            sigma = mandel(np.eye(dim) + E.T @ E)
            u_e = _affine_displacement_e(mesh, groupElem, E)

            leftDispPart_e_pg = groupElem.Get_leftDispPart_e_pg(MatrixType.rigi)

            work = np.einsum("ei,epij,j->", u_e, leftDispPart_e_pg, sigma)
            assert rel_error(work, MEASURES[dim] * mandel(E) @ sigma) < 1e-10


def test_GroupElem_Get_ReactionPart_e_pg_sums_to_dof_n_times_the_measure(
    mesh_of: MeshOf, subtests
):
    for elemType in DIM_2_3:
        for dof_n in [1, 2]:
            with subtests.test(elemType=str(elemType), dof_n=dof_n):
                groupElem = _main(mesh_of(elemType))

                reactionPart_e_pg = groupElem.Get_ReactionPart_e_pg(
                    MatrixType.mass, dof_n
                )

                assert reactionPart_e_pg.shape[-1] == groupElem.nPe * dof_n
                expected = dof_n * MEASURES[groupElem.dim]
                assert rel_error(reactionPart_e_pg.sum(), expected) < 1e-12


def test_GroupElem_Get_DiffusePart_e_pg_integrates_the_gradient_energy(
    mesh_of: MeshOf, subtests
):
    """fᵀ ∫ ∇Nᵀ ∇N dΩ f = measure · |a|² for f = a·x."""
    for elemType in DIM_2_3:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            dim = groupElem.dim
            a = np.array([1.0, -2.0, 0.5])[:dim]
            f_e = affine(mesh.coord, a)[groupElem.connect]

            diffusePart_e_pg = groupElem.Get_DiffusePart_e_pg(MatrixType.rigi)
            dN_e_pg = groupElem.Get_dN_e_pg(MatrixType.rigi)

            energy = np.einsum("ei,epid,epdj,ej->", f_e, diffusePart_e_pg, dN_e_pg, f_e)
            assert rel_error(energy, MEASURES[dim] * a @ a) < 1e-10


def test_GroupElem_Get_SourcePart_e_pg_sums_to_dof_n_times_the_measure(
    mesh_of: MeshOf, subtests
):
    for elemType in DIM_2_3:
        for dof_n in [1, 3]:
            with subtests.test(elemType=str(elemType), dof_n=dof_n):
                groupElem = _main(mesh_of(elemType))

                sourcePart_e_pg = groupElem.Get_SourcePart_e_pg(MatrixType.mass, dof_n)

                assert sourcePart_e_pg.shape[-2:] == (groupElem.nPe * dof_n, dof_n)
                expected = dof_n * MEASURES[groupElem.dim]
                assert rel_error(sourcePart_e_pg.sum(), expected) < 1e-12


def test_GroupElem_Get_Gradient_e_pg_of_an_affine_displacement_is_its_matrix(
    mesh_of: MeshOf, subtests
):
    G = _strain(3) + np.array([[0, 1, 0], [-1, 0, 0.5], [0, 0, 0]])
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = _main(mesh)
            dim = groupElem.dim
            u = affine(mesh.coord, G[:dim, :dim])

            grad_e_pg = groupElem.Get_Gradient_e_pg(u)

            expected = np.zeros((3, 3))
            expected[:dim, :dim] = G[:dim, :dim]
            expected = np.broadcast_to(expected, grad_e_pg.shape)
            assert rel_error(np.asarray(grad_e_pg), expected) < 1e-10


def test_GroupElem_Get_Gradient_e_pg_rejects_a_field_of_the_wrong_size():
    groupElem = reference_element(ElemType.TRI3)

    with pytest.raises(AssertionError):
        groupElem.Get_Gradient_e_pg(np.ones(4))


# ----------------------------------------------------------------------------
# nodes and elements
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid() -> Mesh:
    """4 × 3 QUAD4 grid of unit squares."""
    contour = Domain((0, 0), (4, 3), 1.0)
    return contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)


def test_GroupElem_Get_Elements_Nodes_exclusively_or_not(grid: Mesh):
    groupElem = grid.groupElem
    corner = groupElem.Get_Nodes_Conditions(lambda x, y, z: (x <= 1) & (y <= 1))

    only = groupElem.Get_Elements_Nodes(corner)
    touching = groupElem.Get_Elements_Nodes(list(corner), exclusively=False)

    assert only.size == 1
    assert touching.size == 4
    assert set(only) <= set(touching)


def test_GroupElem_Get_Elements_Nodes_ignores_nodes_beyond_the_mesh():
    """A group using nodes 5 and 6 of 7 keeps them when node 99 comes along."""
    groupElem = GroupElemFactory.Create(
        ElemType.SEG2, np.array([[5, 6]]), np.zeros((7, 3))
    )

    elements = groupElem.Get_Elements_Nodes(np.array([5, 6, 99]))

    assert np.array_equal(elements, [0])


def test_GroupElem_Get_Nodes_Conditions_selects_by_coordinates(grid: Mesh):
    nodes = grid.groupElem.Get_Nodes_Conditions(lambda x, y, z: x == 0)

    assert nodes.size == 4
    assert np.all(grid.coord[nodes, 0] == 0)


def test_GroupElem_Get_Nodes_Conditions_needs_a_boolean_of_x_y_z(grid: Mesh):
    groupElem = grid.groupElem

    assert groupElem.Get_Nodes_Conditions(lambda x, y, z: x) is None
    assert groupElem.Get_Nodes_Conditions(lambda x: x == 0) is None


def test_GroupElem_Get_Nodes_Point_within_tolerance():
    """A point off by round-off finds its node, a point one tolerance away finds none."""
    groupElem = _segments([(0, 0, 0), (1, 1, 0)])

    with np.errstate(all="raise"):
        nodes = groupElem.Get_Nodes_Point((1 + 1e-9, 1))

    assert np.array_equal(nodes, [1])
    assert np.array_equal(groupElem.Get_Nodes_Point((0, 0)), [0])
    assert groupElem.Get_Nodes_Point((1 + 1e-3, 1)).size == 0


def test_GroupElem_Get_Nodes_Line_selects_the_nodes_on_the_segment(grid: Mesh):
    line = Line((1, 0), (1, 2))

    nodes = grid.groupElem.Get_Nodes_Line(line)

    assert np.array_equal(np.sort(grid.coord[nodes, 1]), [0, 1, 2])
    assert np.all(grid.coord[nodes, 0] == 1)


def test_GroupElem_Get_Nodes_Domain_selects_the_nodes_in_the_box(grid: Mesh):
    domain = Domain((1, 1), (2, 3))

    nodes = grid.groupElem.Get_Nodes_Domain(domain)

    assert nodes.size == 6


def test_GroupElem_Get_Nodes_Circle_selects_in_or_on_the_circle(grid: Mesh, subtests):
    """Radius 1 around a node: itself and its four neighbours."""
    circle = Circle((2, 1), 2)
    for onlyOnEdge, count in [(False, 5), (True, 4)]:
        with subtests.test(onlyOnEdge=onlyOnEdge):
            nodes = grid.groupElem.Get_Nodes_Circle(circle, onlyOnEdge)

            assert nodes.size == count


@pytest.fixture(scope="module")
def slab() -> Mesh:
    """4 × 3 × 2 HEXA8 grid of unit cubes."""
    contour = Domain((0, 0), (4, 3), 1.0)
    return contour.Mesh_Extrude([], (0, 0, 2), [2], ElemType.HEXA8, isOrganised=True)


def test_GroupElem_Get_Nodes_Cylinder_selects_through_the_thickness(
    slab: Mesh, subtests
):
    circle = Circle((2, 1), 2)
    for onlyOnEdge, count in [(False, 5 * 3), (True, 4 * 3)]:
        with subtests.test(onlyOnEdge=onlyOnEdge):
            nodes = slab.groupElem.Get_Nodes_Cylinder(circle, onlyOnEdge=onlyOnEdge)

            assert nodes.size == count


def test_GroupElem_Get_Nodes_Cylinder_along_another_axis(slab: Mesh):
    """Radius 1 around the x = 2, z = 1 line, along y: 5 nodes per y layer."""
    circle = Circle((2, 0, 1), 2, n=(0, 1, 0))

    nodes = slab.groupElem.Get_Nodes_Cylinder(circle, direction=[0, 1, 0])

    assert nodes.size == 5 * 4
    distances = np.linalg.norm(slab.coord[nodes][:, [0, 2]] - [2, 1], axis=1)
    assert np.all(distances <= 1 + 1e-12)


def test_GroupElem_Get_Nodes_Cylinder_tilted_from_its_circle(slab: Mesh):
    """A circle in the xy plane swept along y selects within |x - 2| ≤ 1 at z = 1 only... and nothing outside the radius."""
    circle = Circle((2, 1, 1), 2)

    nodes = slab.groupElem.Get_Nodes_Cylinder(circle, direction=[0, 1, 0])

    distances = np.linalg.norm(slab.coord[nodes][:, [0, 2]] - [2, 1], axis=1)
    assert nodes.size > 0 and np.all(distances <= 1 + 1e-12)


def test_GroupElem_Set_Tag_tags_the_nodes_and_their_elements(grid: Mesh):
    groupElem = grid.copy().groupElem
    corner = groupElem.Get_Nodes_Conditions(lambda x, y, z: (x <= 1) & (y <= 1))

    groupElem.Set_Tag(corner, "corner")
    groupElem.Set_Tag(corner, "picked", np.array([0, 1]))
    groupElem.Set_Tag(np.array([], dtype=int), "nothing")

    assert np.array_equal(groupElem.Get_Nodes_Tag("corner"), corner)
    assert groupElem.Get_Elements_Tag("corner").size == 1
    assert np.array_equal(groupElem.Get_Elements_Tag("picked"), [0, 1])
    assert "nothing" not in groupElem.nodeTags


def test_GroupElem_Set_Tag_rejects_out_of_range_nodes_or_elements(grid: Mesh):
    groupElem = grid.copy().groupElem

    with pytest.raises(Exception, match="nodes must be"):
        groupElem.Set_Tag(np.array([grid.Nn]), "out")
    with pytest.raises(Exception, match="elements must be"):
        groupElem.Set_Tag(np.array([0]), "out", np.array([grid.Ne]))
    with pytest.raises(Exception, match="1D array of integers"):
        groupElem.Set_Tag(np.array([0]), "out", np.array([0.5]))


def test_GroupElem_nodeTags_lists_the_node_tags(grid: Mesh):
    assert grid.Get_list_groupElem(0)[0].nodeTags == ["P0", "P1", "P2", "P3"]
    assert grid.Get_list_groupElem(1)[0].nodeTags == ["L0", "L1", "L2", "L3"]
    assert grid.groupElem.nodeTags == ["S0"]


def test_GroupElem__dict_nodes_tags_maps_tags_to_nodes(grid: Mesh):
    points = grid.Get_list_groupElem(0)[0]

    tags = points._dict_nodes_tags

    assert list(tags) == points.nodeTags
    assert np.array_equal(tags["P2"], grid.Nodes_Point((4, 3)))


def test_GroupElem_elementTags_lists_the_element_tags(grid: Mesh):
    assert grid.groupElem.elementTags == ["S0"]


def test_GroupElem__dict_elements_tags_maps_tags_to_elements(grid: Mesh):
    tags = grid.groupElem._dict_elements_tags

    assert np.array_equal(np.sort(tags["S0"]), np.arange(grid.Ne))


def test_GroupElem_Get_Elements_Tag_and_Get_Nodes_Tag_reject_an_unknown_tag(subtests):
    groupElem = _segments([(0, 0, 0), (1, 0, 0)])
    groupElem.Set_Tag(np.array([0, 1]), "L0")
    for name in ["Get_Elements_Tag", "Get_Nodes_Tag"]:
        with subtests.test(name=name):
            assert getattr(groupElem, name)("L0").size > 0
            with pytest.raises(KeyError, match="L1"):
                getattr(groupElem, name)("L1")


def test_GroupElem_Get_Elements_Tag_gives_the_tagged_elements(grid: Mesh):
    edges = grid.Get_list_groupElem(1)[0]

    elements = edges.Get_Elements_Tag("L0")

    assert elements.size == 4
    assert np.all(
        edges.coord[edges._global_to_local_nodes[edges.connect[elements]], 1] == 0
    )


def test_GroupElem_Get_Nodes_Tag_gives_the_tagged_nodes(grid: Mesh):
    nodes = grid.Get_list_groupElem(1)[0].Get_Nodes_Tag("L1")

    assert np.all(grid.coord[nodes, 0] == 4)


def test_GroupElem_Locates_sol_e_reads_nodal_or_dof_values(grid: Mesh):
    groupElem = grid.groupElem
    connect = groupElem.connect
    scalar = np.arange(grid.Nn, dtype=float)
    vector = np.arange(2 * grid.Nn, dtype=float)

    assert np.array_equal(groupElem.Locates_sol_e(scalar), connect)
    assert np.array_equal(groupElem.Locates_sol_e(vector), groupElem.Get_assembly_e(2))
    assert np.array_equal(groupElem.Locates_sol_e(scalar, 1), connect)
    with pytest.raises(Exception, match="Wrong dimension"):
        groupElem.Locates_sol_e(np.ones(grid.Nn + 1))


def test_GroupElem_Get_pointsInElem_finds_the_centroid_but_not_a_far_point(subtests):
    """POINT takes its own branch."""
    quad = reference_element(ElemType.QUAD9, 2 * np.eye(3)[:, :2], (1, 2, 3))
    for groupElem, inside in [(quad, quad.center), (_point(), np.ones(3))]:
        with subtests.test(elemType=str(groupElem.elemType)):
            points = np.array([inside, inside + 10])

            assert np.array_equal(groupElem.Get_pointsInElem(points, 0), [0])
            assert groupElem.Get_pointsInElem(np.empty((0, 3)), 0).size == 0


def _gauss_points(mesh: Mesh) -> tuple[_GroupElem, np.ndarray, np.ndarray]:
    """Interior points of each element: its mass rule, or 8 points for prisms, whose 6-point rule sits on the edges."""
    groupElem = _main(mesh)
    nPg = 8 if groupElem.topology == "PRISM" else MatrixType.mass
    xi_pg = Gauss(groupElem.elemType, nPg).coord  # type: ignore [arg-type]
    N_pg = _GroupElem._Eval_Functions(groupElem._N(), xi_pg)[:, 0]
    x_e_pg = np.einsum("pn,end->epd", N_pg, mesh.coord[groupElem.connect])
    return groupElem, x_e_pg.reshape(-1, 3), xi_pg


def test_GroupElem_Get_Mapping_finds_each_gauss_point_in_its_element(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem, points, xi_pg = _gauss_points(mesh_of(elemType))
            nPg = xi_pg.shape[0]

            nodes, elements, connect_e_n, coordInElem = groupElem.Get_Mapping(
                points, needCoordinates=True
            )

            assert np.array_equal(nodes, np.arange(points.shape[0]))
            assert np.array_equal(elements, groupElem.elements)
            assert all(
                np.array_equal(c, e * nPg + np.arange(nPg))
                for e, c in zip(elements, connect_e_n)
            )
            expected = np.tile(xi_pg, (groupElem.Ne, 1))
            assert rel_error(coordInElem, expected, scale=1.0) < 1e-8


def test_GroupElem_Get_Mapping_on_given_elements_of_a_distorted_mesh():
    """Distorted elements take the iterative inverse mapping."""
    contour = Domain((0, 0), (2, 2), 1.0)
    mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    coord = mesh.coord
    coord[(coord[:, 0] == 1) & (coord[:, 1] == 1), :2] = [1.3, 0.8]
    mesh.coord = coord
    groupElem, points, xi_pg = _gauss_points(mesh)

    nodes, elements, _, coordInElem = groupElem.Get_Mapping(
        points, np.array([0]), needCoordinates=True
    )

    assert np.array_equal(elements, [0])
    assert np.array_equal(nodes, np.arange(4))
    assert rel_error(coordInElem[:4], xi_pg, scale=1.0) < 1e-8
    assert np.all(np.isinf(coordInElem[4:]))


def test_GroupElem__Get_xi_inverts_the_mapping_of_every_element(
    mesh_of: MeshOf, subtests
):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem, points, xi_pg = _gauss_points(mesh_of(elemType))
            elements_n = groupElem.elements.repeat(xi_pg.shape[0])

            xi_n = groupElem._Get_xi(points, elements_n)

            expected = np.tile(xi_pg, (groupElem.Ne, 1))
            assert rel_error(xi_n, expected, scale=1.0) < 1e-8


def test_GroupElem__Get_xi_solves_a_bilinear_element():
    coord = np.array([[0, 0, 0], [2, 0, 0], [3, 2, 0], [0, 1, 0]], dtype=float)
    groupElem = GroupElemFactory.Create(ElemType.QUAD4, np.array([[0, 1, 2, 3]]), coord)
    xi = np.array([[0.3, -0.4]])
    x = _GroupElem._Eval_Functions(groupElem._N(), xi)[:, 0] @ coord

    assert rel_error(groupElem._Get_xi(x, np.array([0])), xi, scale=1.0) < 1e-12


def test_GroupElem_Get_pointsInElem_includes_the_boundary(subtests):
    for elemType in ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)
            corner = groupElem.coord[groupElem.connect[0, 0]]
            points = np.array([corner, groupElem.center, corner - 0.5])

            assert np.array_equal(groupElem.Get_pointsInElem(points, 0), [0, 1])


def test_GroupElem_Get_pointsInElem_needs_the_point_on_a_lower_dimensional_element():
    segment = _segments([[0, 0, 0], [2, 0, 0]])
    points = np.array([[1, 0, 0], [1, 1e-3, 0], [3, 0, 0]], dtype=float)

    assert np.array_equal(segment.Get_pointsInElem(points, 0), [0])


def test_GroupElem__Get_all_containing_elements_lists_a_point_on_a_shared_edge_in_both_elements(
    grid: Mesh,
):
    points = np.array([[1, 0.5, 0], [10, 10, 0]], dtype=float)

    nodes, elements, xi, refDistance = grid.groupElem._Get_all_containing_elements(
        points
    )

    assert np.array_equal(nodes, [0, 0])
    assert np.unique(elements).size == 2
    assert np.allclose(refDistance, 0) and xi.shape == (2, 2)


def test_GroupElem__Get_containing_elements_keeps_the_most_inside_element():
    """Overlapping segments [0, 2] and [1, 3]: x = 1.2 is deeper in the first."""
    coord = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=float)
    groupElem = GroupElemFactory.Create(
        ElemType.SEG2, np.array([[0, 1], [2, 3]]), coord
    )
    points = np.array([[1.2, 0, 0]])

    assert groupElem._Get_containing_elements(points)[0][0] == 0
    keep = lambda nodes_p, elements_p: elements_p == 1  # noqa: E731
    assert groupElem._Get_containing_elements(points, keep=keep)[0][0] == 1


def test_GroupElem__Get_containing_elements_finds_the_element_and_its_xi(grid: Mesh):
    groupElem = grid.groupElem
    points = np.array([[1.2, 0.5, 0], [10, 10, 0]])

    elem_n, xi_n = groupElem._Get_containing_elements(points)

    assert groupElem.Get_pointsInElem(points[:1], elem_n[0]).size == 1
    N = _GroupElem._Eval_Functions(groupElem._N(), xi_n[:1])[0, 0]
    assert rel_error(N @ grid.coord[groupElem.connect[elem_n[0]]], points[0]) < 1e-12
    assert elem_n[1] == -1 and np.all(np.isinf(xi_n[1]))


# ----------------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------------


def test_GroupElemFactory_creates_the_element_class(subtests):
    for elemType in [ElemType.POINT] + ALL:
        with subtests.test(elemType=str(elemType)):
            groupElem = reference_element(elemType)

            assert type(groupElem).__name__ == elemType.name


def test_GroupElemFactory_Get_ElemInFos_reads_the_gmsh_table():
    assert GroupElemFactory.Get_ElemInFos(9) == (ElemType.TRI6, 6, 2, 2, 3, 3, 0, 0)
    with pytest.raises(KeyError):
        GroupElemFactory.Get_ElemInFos(7)


def test_GroupElemFactory__Create_builds_from_a_gmsh_id():
    groupElem = GroupElemFactory._Create(3, np.array([[0, 1, 2, 3]]), np.zeros((4, 3)))

    assert groupElem.elemType is ElemType.QUAD4


def test_GroupElemFactory_Create_rejects_an_unknown_type():
    with pytest.raises(KeyError):
        GroupElemFactory.Create("PYRA5", np.array([[0]]), np.zeros((1, 3)))  # type: ignore [arg-type]


def test_GroupElemFactory__Get_2d_element_types_are_the_faces_of_the_mesh(subtests):
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            contour = Domain((0, 0), (1, 1), 1.0)
            mesh = contour.Mesh_Extrude([], (0, 0, 1), [1], elemType, isOrganised=True)

            types = GroupElemFactory._Get_2d_element_types(elemType)

            assert {g.elemType for g in mesh.Get_list_groupElem(2)} == set(types)


def test_GroupElemFactory__Get_2d_element_types_needs_a_volume():
    with pytest.raises(AssertionError):
        GroupElemFactory._Get_2d_element_types(ElemType.TRI3)
