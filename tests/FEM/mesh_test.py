# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.


import pytest
import numpy as np
from scipy.spatial import cKDTree

from EasyFEA.FEM import (
    ElemType,
    MatrixType,
    Mesh,
    GroupElemFactory,
    Load_Mesh,
    Calc_projector,
    Mesh_Optim,
)
from EasyFEA.Geoms import Points, Circle, Domain, Line, Rotate
from EasyFEA.Utilities import _IObserver, Observable

from _numerics import rel_error

L = 2
H = 1


def _grid(elemType: ElemType = ElemType.QUAD4) -> Mesh:
    """4 × 3 rectangle of unit cells."""
    contour = Domain((0, 0), (4, 3), 1.0)
    return contour.Mesh_2D([], elemType, isOrganised=True)


def _slab() -> Mesh:
    """4 × 3 × 2 box of unit HEXA8 cells."""
    contour = Domain((0, 0), (4, 3), 1.0)
    return contour.Mesh_Extrude([], (0, 0, 2), [2], ElemType.HEXA8, isOrganised=True)


@pytest.fixture(scope="module")
def grid() -> Mesh:
    return _grid()


@pytest.fixture
def boxes(make_mesh_2D, make_mesh_3D) -> list[Mesh]:
    """An organised L × H rectangle of TRI3 and L × H × 1 box of TETRA4."""
    return [
        make_mesh_2D(elemType=ElemType.TRI3, isOrganised=True),
        make_mesh_3D(elemType=ElemType.TETRA4, isOrganised=True),
    ]


@pytest.fixture(scope="module")
def mixed(unit_square) -> Mesh:
    """A QUAD4 unit square beside a TRI3 one: two groups of dimension 2."""
    left = unit_square(1 / 2)
    right = unit_square(1 / 2, ElemType.TRI3, x0=1)
    return Mesh.Merge([left, right])


# ----------------------------------------------------------------------------
# construction and properties
# ----------------------------------------------------------------------------


def test_Mesh_needs_groups_sharing_their_coordinates(grid: Mesh):
    other = GroupElemFactory.Create(ElemType.SEG2, np.array([[0, 1]]), np.zeros((2, 3)))

    with pytest.raises(AssertionError):
        Mesh({})
    with pytest.raises(AssertionError):
        Mesh({ElemType.QUAD4: grid.groupElem, ElemType.SEG2: other})


def test_Mesh__ResetMatrix_drops_every_cached_matrix(grid: Mesh):
    mesh = grid.copy()
    jacobian = mesh.groupElem.Get_jacobian_e_pg(MatrixType.rigi)

    mesh._ResetMatrix()

    assert mesh.groupElem.Get_jacobian_e_pg(MatrixType.rigi) is not jacobian


def test_Mesh_Get_list_groupElem_filters_by_dimension(grid: Mesh):
    assert [g.elemType for g in grid.Get_list_groupElem()] == [ElemType.QUAD4]
    assert [g.elemType for g in grid.Get_list_groupElem(1)] == [ElemType.SEG2]
    assert [g.elemType for g in grid.Get_list_groupElem(0)] == [ElemType.POINT]
    assert grid.Get_list_groupElem(3) == []


def test_Mesh_orphanNodes_are_used_by_no_element(capsys):
    groupElem = GroupElemFactory.Create(
        ElemType.SEG2, np.array([[0, 2]]), np.zeros((3, 3))
    )

    mesh = Mesh({ElemType.SEG2: groupElem}, verbosity=True)

    assert mesh.orphanNodes == [1]
    assert "Orphan nodes" in capsys.readouterr().out


def test_Mesh_dict_groupElem_holds_every_dimension(grid: Mesh):
    assert set(grid.dict_groupElem) == {ElemType.POINT, ElemType.SEG2, ElemType.QUAD4}


def test_Mesh_groupElem_is_the_main_group_or_ambiguous(grid: Mesh, mixed: Mesh):
    assert grid.groupElem.elemType is ElemType.QUAD4
    with pytest.raises(Exception, match="several groups|ragged"):
        mixed.groupElem


def test_Mesh_elemType_is_the_main_type(grid: Mesh):
    assert grid.elemType is ElemType.QUAD4


def test_Mesh_Ne_counts_every_main_group(grid: Mesh, mixed: Mesh):
    assert grid.Ne == 12
    assert mixed.Ne == sum(g.Ne for g in mixed.Get_list_groupElem()) == 4 + 8


def test_Mesh_Nn_counts_the_coordinates(grid: Mesh):
    assert grid.Nn == 20


def test_Mesh_dim_is_the_largest_element_dimension(grid: Mesh):
    assert grid.dim == 2 and _slab().dim == 3


def test_Mesh_inDim_follows_the_embedding(grid: Mesh):
    tilted = grid.copy()
    tilted.Rotate(30, direction=(1, 0, 0))

    assert grid.inDim == 2
    assert tilted.inDim == 3


def test_Mesh__Get_realistic_vector_magnitude_scales_the_mesh_radius(grid: Mesh):
    radius = np.linalg.norm([2, 1.5])

    assert rel_error(grid._Get_realistic_vector_magnitude(0.1), 0.1 * radius) < 1e-12


def test_Mesh_copy_is_independent(grid: Mesh):
    copied = grid.copy()

    copied.Translate(1)

    assert copied.coord[:, 0].min() == 1 and grid.coord[:, 0].min() == 0


def _same_measures(mesh: Mesh, other: Mesh) -> None:
    assert (mesh.dim, mesh.Nn, mesh.Ne) == (other.dim, other.Nn, other.Ne)
    for name in ["length", "area", "volume"]:
        value = getattr(mesh, name)
        if value is None:
            assert getattr(other, name) is None
        else:
            assert rel_error(getattr(other, name), value) < 1e-12


def test_Mesh_Translate_moves_the_center_and_keeps_the_measures(
    boxes: list[Mesh], subtests
):
    shift = np.array([0.3, -1.2, 2.5])
    for mesh in boxes:
        with subtests.test(elemType=str(mesh.elemType)):
            moved = mesh.copy()

            moved.Translate(*shift)

            assert rel_error(moved.center, mesh.center + shift) < 1e-12
            _same_measures(mesh, moved)


def test_Mesh_Rotate_turns_the_center_and_keeps_the_measures(
    boxes: list[Mesh], subtests
):
    pivot = (0.5, -1, 2)
    for mesh in boxes:
        for name, axis in [("z", (0, 0, 1)), ("oblique", (1, 3, -1))]:
            with subtests.test(elemType=str(mesh.elemType), axis=name):
                moved = mesh.copy()

                moved.Rotate(37, pivot, axis)

                expected = Rotate(mesh.center, 37, pivot, axis)[0]
                assert rel_error(moved.center, expected) < 1e-12
                _same_measures(mesh, moved)


def test_Mesh_Symmetry_mirrors_the_center_and_keeps_the_measures(
    boxes: list[Mesh], subtests
):
    for mesh in boxes:
        with subtests.test(elemType=str(mesh.elemType)):
            mirrored = mesh.copy()

            mirrored.Symmetry((1, 0, 0), (1, 0, 0))

            expected = mesh.center * [-1, 1, 1] + [2, 0, 0]
            assert rel_error(mirrored.center, expected) < 1e-12
            _same_measures(mesh, mirrored)


def test_Mesh_nodes_are_those_of_the_main_groups(grid: Mesh, mixed: Mesh):
    assert np.array_equal(grid.nodes, np.arange(grid.Nn))
    assert np.array_equal(mixed.nodes, np.arange(mixed.Nn))


def test_Mesh__Get_mpi_owned_nodes_in_serial_are_all_the_nodes(grid: Mesh, mixed: Mesh):
    assert np.array_equal(grid._Get_mpi_owned_nodes(), grid.nodes)
    assert np.array_equal(mixed._Get_mpi_owned_nodes(), mixed.nodes)


class _Recorder(_IObserver):
    def __init__(self):
        self.events: list[str] = []

    def _Update(self, observable: Observable, event: str) -> None:
        self.events.append(event)


def test_Mesh_coord_setter_moves_every_group_and_notifies(grid: Mesh):
    mesh = grid.copy()
    recorder = _Recorder()
    mesh._Add_observer(recorder)
    coord = mesh.coord
    coord[:, 0] *= 2

    mesh.coord = coord

    assert np.array_equal(mesh.coord, coord)
    assert rel_error(mesh.area, 2 * grid.area) < 1e-12
    assert np.array_equal(mesh.Get_list_groupElem(0)[0].coord[:, 0].max(), 8)
    assert recorder.events == ["The mesh has been modified"]


def test_Mesh_connect_is_the_main_connectivity(grid: Mesh):
    assert np.array_equal(grid.connect, grid.groupElem.connect)


def test_Mesh__Gather_in_serial_is_the_mesh(grid: Mesh):
    assert grid._Gather() is grid


def test_Mesh_verbosity_prints_the_mesh(capsys, grid: Mesh):
    mesh = Mesh(grid.dict_groupElem, verbosity=True)

    assert mesh.verbosity and not grid.verbosity
    assert "elemType: QUAD4" in capsys.readouterr().out


def test_Mesh_Get_connect_n_e_stacks_the_main_groups(grid: Mesh, mixed: Mesh):
    assert grid.Get_connect_n_e().shape == (grid.Nn, grid.Ne)
    connect_n_e = mixed.Get_connect_n_e()
    assert connect_n_e.shape == (mixed.Nn, mixed.Ne)
    nPe = np.concatenate([[g.nPe] * g.Ne for g in mixed.Get_list_groupElem()])
    assert np.array_equal(np.ravel(connect_n_e.sum(0)), nPe)


def test_Mesh_Get_assembly_e_is_the_main_one(grid: Mesh):
    assert np.array_equal(grid.Get_assembly_e(2), grid.groupElem.Get_assembly_e(2))


def test_Mesh_length_sums_the_lines(grid: Mesh):
    assert rel_error(grid.length, 2 * (4 + 3)) < 1e-12
    point = GroupElemFactory.Create(ElemType.POINT, np.array([[0]]), np.zeros((1, 3)))
    assert Mesh({ElemType.POINT: point}).length is None


def _moved(mesh: Mesh) -> Mesh:
    moved = mesh.copy()
    moved.Rotate(45, direction=(1, 0, 0))
    moved.Rotate(45, direction=(0, 1, 0))
    moved.Translate(-1, 2, 3)
    return moved


def test_Mesh_area_of_a_moved_rectangle_or_box_surface(boxes: list[Mesh], subtests):
    """2 × 1 rectangle, or the 2 × 1 × 1 box whose surface is 10."""
    for mesh in boxes:
        with subtests.test(elemType=str(mesh.elemType)):
            expected = L * H if mesh.dim == 2 else 2 * (L * H + L + H)

            assert rel_error(_moved(mesh).area, expected) < 1e-10
    edges = Mesh({ElemType.SEG2: _grid().Get_list_groupElem(1)[0]})
    assert edges.area is None


def test_Mesh_volume_of_a_moved_box(boxes: list[Mesh], subtests):
    for mesh in boxes:
        with subtests.test(elemType=str(mesh.elemType)):
            if mesh.dim == 2:
                assert mesh.volume is None
            else:
                assert rel_error(_moved(mesh).volume, L * H) < 1e-12


def test_Mesh_center_weighs_the_main_groups(grid: Mesh, mixed: Mesh):
    assert rel_error(grid.center, [2, 1.5, 0]) < 1e-12
    assert rel_error(mixed.center, [1, 0.5, 0]) < 1e-12


def test_Mesh_Get_normals_point_out_of_the_boundary(grid: Mesh):
    nodes = grid.Nodes_Conditions(lambda x, y, z: (x == 4) & (y > 0) & (y < 3))

    normals, used = grid.Get_normals(nodes)

    assert np.array_equal(np.sort(used), np.sort(nodes))
    assert (
        rel_error(np.abs(normals), np.tile([1, 0, 0], (nodes.size, 1)), scale=1.0)
        < 1e-12
    )


def test_Mesh_Get_normals_rejects_an_out_of_range_node(grid: Mesh):
    with pytest.raises(AssertionError):
        grid.Get_normals(np.array([grid.Nn]))


# ----------------------------------------------------------------------------
# node and element selection
# ----------------------------------------------------------------------------


def test_Mesh_Nodes_Conditions_selects_by_coordinates(grid: Mesh, mixed: Mesh):
    assert grid.Nodes_Conditions(lambda x, y, z: x == 0).size == 4
    nodes = mixed.Nodes_Conditions(lambda x, y, z: y == 0)
    assert np.array_equal(np.sort(mixed.coord[nodes, 0]), np.linspace(0, 2, 5))


def test_Mesh_Nodes_Point_takes_points_or_lists_of_points(grid: Mesh):
    nodes = grid.Nodes_Point((0, 0), [(4, 3), (4, 0)])

    assert {tuple(c) for c in grid.coord[nodes, :2]} == {(0, 0), (4, 3), (4, 0)}


def test_Mesh_Nodes_Line_selects_the_nodes_on_the_segments(grid: Mesh):
    nodes = grid.Nodes_Line(Line((0, 0), (4, 0)), [Line((0, 0), (0, 3))])

    assert nodes.size == 5 + 4 - 1


def test_Mesh_Nodes_Domain_selects_the_nodes_in_the_box():
    contour = Domain((0, 0), (4, 3), meshSize=1.0)
    mesh = contour.Mesh_2D([], ElemType.QUAD4, isOrganised=True)
    corner = Domain((0, 0), (1, 1))
    nodes = mesh.Nodes_Domain(corner)

    assert nodes.size == 4
    assert np.all(mesh.coord[nodes][:, :2] <= 1 + 1e-12)


def test_Mesh_Nodes_Circle_selects_on_or_in_the_circles(grid: Mesh, subtests):
    circles = [Circle((1, 1), 2), [Circle((3, 2), 2)]]
    for onlyOnEdge, count in [(True, 4), (False, 5)]:
        with subtests.test(onlyOnEdge=onlyOnEdge):
            nodes = grid.Nodes_Circle(*circles, onlyOnEdge=onlyOnEdge)

            assert nodes.size == 2 * count


def test_Mesh_Nodes_Cylinder_selects_through_the_thickness():
    slab = _slab()

    nodes = slab.Nodes_Cylinder(Circle((2, 1), 2), [Circle((0, 0), 0.1)])

    assert nodes.size == 5 * 3 + 3
    distances = np.linalg.norm(slab.coord[nodes, :2] - [2, 1], axis=1)
    assert np.sum(distances <= 1 + 1e-12) == 5 * 3


def test_Mesh_Elements_Nodes_grows_by_neighbor_layers(grid: Mesh, mixed: Mesh, capsys):
    corner = grid.Nodes_Point((0, 0))

    touching = grid.Elements_Nodes(corner, exclusively=False)
    second = grid.Elements_Nodes(corner, exclusively=False, neighborLayer=2)
    everything = grid.Elements_Nodes(corner, exclusively=False, neighborLayer=10)

    assert touching.size == 1 and second.size == 4
    assert everything.size == grid.Ne
    assert "All the neighbors" in capsys.readouterr().out
    # global numbering follows Get_list_groupElem: the TRI3 square is at its offset
    groups = mixed.Get_list_groupElem()
    offset = sum(
        g.Ne for g in groups[: [g.elemType for g in groups].index(ElemType.TRI3)]
    )
    right = mixed.Elements_Nodes(mixed.Nodes_Conditions(lambda x, y, z: x >= 1))
    assert np.array_equal(np.sort(right), offset + np.arange(8))


def test_Mesh_Nodes_Tags_unites_the_tagged_nodes(grid: Mesh):
    nodes = grid.Nodes_Tags("L0", ["L1"])

    assert nodes.size == 5 + 4 - 1
    assert np.all((grid.coord[nodes, 1] == 0) | (grid.coord[nodes, 0] == 4))


def test_Mesh_Nodes_Tags_of_an_untagged_mesh_is_empty(capsys):
    groupElem = GroupElemFactory.Create(
        ElemType.SEG2, np.array([[0, 1]]), np.zeros((2, 3))
    )

    nodes = Mesh({ElemType.SEG2: groupElem}).Nodes_Tags("L0")

    assert nodes.size == 0 and "no tags" in capsys.readouterr().out


def test_Mesh_Elements_Tags_gives_the_tagged_main_elements(grid: Mesh, capsys):
    assert np.array_equal(np.sort(grid.Elements_Tags("S0")), np.arange(grid.Ne))
    groupElem = GroupElemFactory.Create(
        ElemType.SEG2, np.array([[0, 1]]), np.zeros((2, 3))
    )
    assert Mesh({ElemType.SEG2: groupElem}).Elements_Tags("L0").size == 0
    assert "no tags" in capsys.readouterr().out


def test_Mesh_Set_Tag_tags_every_group_but_the_points(grid: Mesh):
    mesh = grid.copy()
    nodes = mesh.Nodes_Conditions(lambda x, y, z: x <= 1)

    mesh.Set_Tag(nodes, "left")

    assert "left" not in mesh.Get_list_groupElem(0)[0].nodeTags
    assert "left" in mesh.Get_list_groupElem(1)[0].nodeTags
    assert mesh.Elements_Tags("left").size == 3


def test_Mesh_Locates_sol_e_reads_the_main_group(grid: Mesh):
    sol = np.arange(grid.Nn * 2, dtype=float)

    assert np.array_equal(grid.Locates_sol_e(sol), sol[grid.Get_assembly_e(2)])


def test_Mesh_Get_Node_Values_averages_the_element_values(grid: Mesh):
    """Constant element values stay constant, a scalar stays a vector."""
    scalar = grid.Get_Node_Values(np.full(grid.Ne, 3.0))
    vector = grid.Get_Node_Values(np.tile([1.0, -2.0], (grid.Ne, 1)))

    assert scalar.shape == (grid.Nn,) and np.allclose(scalar, 3)
    assert vector.shape == (grid.Nn, 2) and np.allclose(vector, [1, -2])
    with pytest.raises(AssertionError):
        grid.Get_Node_Values(np.ones(grid.Ne + 1))


def test_Mesh_Evaluate_dofsValues_at_coordinates_is_exact_at_the_nodes(
    mesh_of, subtests
):
    for elemType in ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            dofsValues = np.arange(mesh.Nn * 2) + 1.0

            values = mesh.Evaluate_dofsValues_at_coordinates(mesh.coord, dofsValues)

            # Newton inverse mapping in Get_Mapping converges to ~1e-12, amplified by max(dofsValues)
            expected = dofsValues.reshape(-1, 2)
            np.testing.assert_allclose(values, expected, rtol=1e-9, atol=1e-12)


def test_Mesh_Evaluate_dofsValues_at_coordinates_on_a_mixed_mesh_is_exact_on_affine_fields(
    mixed: Mesh,
):
    points = np.array([[0.3, 0.6, 0], [1.7, 0.2, 0], [1.0, 0.5, 0]])
    f = mixed.coord @ [2.0, -1.0, 0] + 1

    values = mixed.Evaluate_dofsValues_at_coordinates(points, f)

    assert rel_error(values[:, 0], points @ [2.0, -1.0, 0] + 1) < 1e-12


def test_Mesh_Get_Paired_Nodes_pairs_opposite_edges(grid: Mesh):
    corners = grid.Nodes_Point((0, 0), (4, 0), (4, 3), (0, 3))
    corners = grid.coord[corners][
        np.lexsort((grid.coord[corners, 0], grid.coord[corners, 1]))
    ]
    corners = corners[[0, 1, 3, 2]]  # counterclockwise from the origin

    paired = grid.Get_Paired_Nodes(corners)

    assert paired.shape == (3 + 2, 2)
    offsets = grid.coord[paired[:, 1]] - grid.coord[paired[:, 0]]
    assert {tuple(o) for o in offsets} == {(0, 3, 0), (-4, 0, 0)}


def test_Mesh_Get_Paired_Nodes_from_corner_nodes(grid: Mesh):
    corners = [grid.Nodes_Point(p)[0] for p in [(0, 0), (4, 0), (4, 3), (0, 3)]]

    assert grid.Get_Paired_Nodes(np.array(corners)).shape == (5, 2)


def test_Mesh_Get_meshSize_is_the_cell_size(grid: Mesh, mixed: Mesh):
    assert np.allclose(grid.Get_meshSize(), 1)
    assert np.allclose(grid.Get_meshSize(doMean=False), np.ones((grid.Ne, 4)))
    assert mixed.Get_meshSize().size == mixed.Ne
    with pytest.raises(Exception, match="several groups|ragged"):
        mixed.Get_meshSize(doMean=False)


def test_Mesh_Get_Quality_of_squares_is_perfect(grid: Mesh, subtests):
    for criteria in ["aspect", "angular", "jacobian"]:
        with subtests.test(criteria=criteria):
            assert rel_error(grid.Get_Quality(criteria), 1.0) < 1e-12
            quality_n = grid.Get_Quality(criteria, nodeValues=True)
            assert rel_error(quality_n, 1.0) < 1e-12


def test_Mesh_Get_Quality_gamma_of_right_isosceles_triangles():
    """2 r_inscribed / r_circumscribed = 2(√2 - 1)."""
    mesh = _grid(ElemType.TRI3)

    assert rel_error(mesh.Get_Quality("gamma"), 2 * (np.sqrt(2) - 1)) < 1e-12


def test_Mesh_Get_Quality_of_an_unavailable_criterion_is_none(capsys, subtests):
    for name, make, criteria in [
        ("gamma on quads", _grid, "gamma"),
        ("angular in 3D", _slab, "angular"),
        ("unknown", _grid, "unknown"),
    ]:
        with subtests.test(name):
            capsys.readouterr()

            assert make().Get_Quality(criteria) is None
            assert capsys.readouterr().out


def test_Mesh_Get_New_meshSize_n_shrinks_where_the_error_peaks(grid: Mesh):
    error_e = np.ones(grid.Ne)

    meshSize_n = grid.Get_New_meshSize_n(error_e, 1 / 4)

    assert np.allclose(meshSize_n, 1 / 4)
    with pytest.raises(AssertionError):
        grid.Get_New_meshSize_n(np.ones(grid.Ne + 1))


def test_Mesh_Calc_regulation_projector_averages_the_neighbours(grid: Mesh):
    projector = grid.Calc_regulation_projector(1.0)

    assert rel_error(np.ravel(projector.sum(1)), 1.0) < 1e-12
    inner = grid.Nodes_Point((1, 1))[0]
    assert projector[inner].nnz == 5
    assert np.allclose(grid.Calc_regulation_projector(0.5).toarray(), np.eye(grid.Nn))


def test_Mesh_Save_and_Load_Mesh_round_trip(tmp_path, grid: Mesh, subtests):
    path = grid.Save(str(tmp_path), "grid")

    loaded = Load_Mesh(path)

    assert np.array_equal(loaded.coord, grid.coord)
    for elemType in [ElemType.POINT, ElemType.SEG2, ElemType.QUAD4]:
        with subtests.test(elemType=str(elemType)):
            groupElem = grid.dict_groupElem[elemType]
            loadedGroup = loaded.dict_groupElem[elemType]
            assert np.array_equal(loadedGroup.connect, groupElem.connect)
            assert loadedGroup.nodeTags == groupElem.nodeTags


def test_Load_Mesh_rejects_a_missing_file(tmp_path):
    with pytest.raises(AssertionError):
        Load_Mesh(str(tmp_path / "missing.pickle"))


def test_Calc_projector_carries_affine_fields_exactly(subtests):
    finer = Domain((0, 0), (4, 3), 0.4)
    new = finer.Mesh_2D([], ElemType.TRI3)
    a = np.array([1.0, -2.0, 0])
    for elemType in [ElemType.TRI3, ElemType.QUAD4, ElemType.TRI6]:
        with subtests.test(elemType=str(elemType)):
            contour = Domain((0, 0), (4, 3), 1.0)
            old = contour.Mesh_2D([], elemType, isOrganised=True)

            projector = Calc_projector(old, new)

            assert projector.shape == (new.Nn, old.Nn)
            projected = projector @ (old.coord @ a + 3)
            assert rel_error(projected, new.coord @ a + 3) < 1e-10


def test_Calc_projector_needs_meshes_of_one_dimension(grid: Mesh):
    with pytest.raises(AssertionError):
        Calc_projector(grid, _slab())


def test_Calc_projector_warns_on_distorted_elements(grid: Mesh, capsys):
    distorted = grid.copy()
    coord = distorted.coord
    coord[distorted.Nodes_Point((1, 1)), :2] = [1.2, 0.9]
    distorted.coord = coord

    Calc_projector(distorted, grid)

    assert "distorted" in capsys.readouterr().out


def _DoMesh(refine: str | None) -> Mesh:
    contour = Domain((0, 0), (1, 1), 0.3)
    return contour.Mesh_2D([], ElemType.TRI3, refineGeoms=[refine] if refine else [])


def test_Mesh_Optim_returns_at_once_when_every_element_meets_the_quality(tmp_path):
    mesh, ratio = Mesh_Optim(_DoMesh, str(tmp_path), "aspect", quality=0.1)

    assert ratio == 1
    assert np.all(mesh.Get_Quality("aspect") >= 0.1)


def test_Mesh_Optim_refines_and_cleans_its_pos_files(tmp_path):
    mesh, ratio = Mesh_Optim(_DoMesh, str(tmp_path), "aspect", quality=0.99, iterMax=1)

    qualities = mesh.Get_Quality("aspect")
    assert ratio == np.mean(qualities >= 0.99)
    assert not list(tmp_path.glob("*.pos"))
    with pytest.raises(AssertionError):
        Mesh_Optim(_DoMesh, str(tmp_path), ratio=0)


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------


def _no_duplicate_coords(mesh: Mesh, atol: float = 1e-12) -> bool:
    """Returns True when no two nodes share the same position."""
    return len(cKDTree(mesh.coord).query_pairs(atol)) == 0


def test_Mesh_Merge_adjacent_2d_area_and_nodes(unit_square, subtests):
    """Two adjacent unit squares: area = 2, shared edge nodes merged, no duplicates."""
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            leftMesh = unit_square(elemType=elemType)
            rightMesh = unit_square(elemType=elemType, x0=1)

            merged = Mesh.Merge([leftMesh, rightMesh])

            assert abs(merged.area - 2.0) / 2.0 < 1e-10
            assert merged.Nn < leftMesh.Nn + rightMesh.Nn
            assert _no_duplicate_coords(merged)


def test_Mesh_Merge_adjacent_3d_volume_and_nodes(subtests):
    """Two adjacent unit cubes: volume = 2, shared face nodes merged, no duplicates."""
    h = 0.5
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            leftContour = Points([(0, 0), (1, 0), (1, 1), (0, 1)], h)
            left = leftContour.Mesh_Extrude(
                [], [0, 0, 1], [2], elemType, isOrganised=True
            )
            rightContour = Points([(1, 0), (2, 0), (2, 1), (1, 1)], h)
            right = rightContour.Mesh_Extrude(
                [], [0, 0, 1], [2], elemType, isOrganised=True
            )

            merged = Mesh.Merge([left, right])

            assert abs(merged.volume - 2.0) / 2.0 < 1e-10
            assert merged.Nn < left.Nn + right.Nn
            assert _no_duplicate_coords(merged)


def test_Mesh_Merge_of_a_single_mesh_returns_it(unit_square):
    mesh = unit_square(0.4)

    result, mapping = Mesh.Merge([mesh], return_mapping=True)

    assert Mesh.Merge([mesh]) is mesh and result is mesh
    assert len(mapping) == 1
    np.testing.assert_array_equal(mapping[0], np.arange(mesh.Nn))


def test_Mesh_Merge_construct_unique_elements_removes_duplicates(unit_square):
    mesh = unit_square(0.4)

    merged = Mesh.Merge([mesh, mesh], constructUniqueElements=True)
    kept = Mesh.Merge([mesh, mesh], constructUniqueElements=False)

    assert merged.Ne == mesh.Ne and merged.Nn == mesh.Nn
    assert kept.Ne == 2 * mesh.Ne


def test_Mesh_Merge_tolerance_merges_near_coincident_nodes(unit_square):
    """Nodes offset by < mergePointsTol are merged; nodes beyond it are not."""
    mesh = unit_square(0.4)

    def perturbed(d):
        m = mesh.copy()
        m.coord = mesh.coord + d
        return m

    assert Mesh.Merge([mesh, perturbed(1e-13)]).Nn == mesh.Nn
    assert Mesh.Merge([mesh, perturbed(1e-11)]).Nn == 2 * mesh.Nn


def test_Mesh_Merge_without_merging_points_concatenates_them(unit_square):
    left = unit_square()
    right = unit_square(x0=1)

    merged = Mesh.Merge([left, right], mergePoints=False)

    assert merged.Nn == left.Nn + right.Nn
    assert abs(merged.area - 2.0) / 2.0 < 1e-10


def test_Mesh_Merge_mapping_sends_each_node_to_its_merged_position(unit_square):
    left = unit_square()
    right = unit_square(x0=1)

    merged, mapping = Mesh.Merge([left, right], return_mapping=True)

    assert len(mapping) == 2
    np.testing.assert_allclose(merged.coord[mapping[0]], left.coord, atol=1e-12)
    np.testing.assert_allclose(merged.coord[mapping[1]], right.coord, atol=1e-12)
    shared_left = mapping[0][left.coord[:, 0] == 1.0]
    shared_right = mapping[1][right.coord[:, 0] == 1.0]
    assert set(shared_left) == set(shared_right)


def test_Mesh_Merge_tags_survive_a_mixed_merge(unit_square, subtests):
    """A QUAD4 square beside a TRI3 square: each group keeps its element and node tags."""
    left = unit_square()
    right = unit_square(elemType=ElemType.TRI3, isOrganised=False, x0=1)

    merged, mapping = Mesh.Merge([left, right], return_mapping=True)

    for mesh, nodes in zip([left, right], mapping):
        groupElem = mesh.groupElem
        with subtests.test(elemType=str(groupElem.elemType)):
            mergedGroup = merged.dict_groupElem[groupElem.elemType]
            assert mergedGroup.elementTags == ["S0"]
            np.testing.assert_array_equal(
                mergedGroup.Get_Elements_Tag("S0"), np.arange(groupElem.Ne)
            )
            np.testing.assert_array_equal(
                np.sort(mergedGroup.Get_Nodes_Tag("S0")),
                np.unique(nodes[groupElem.Get_Nodes_Tag("S0")]),
            )


def _tagged_elements(groupElem) -> dict[str, set[tuple]]:
    """Each element tag's elements, as sets of sorted node tuples."""
    return {
        tag: {
            tuple(sorted(e)) for e in groupElem.connect[groupElem.Get_Elements_Tag(tag)]
        }
        for tag in groupElem.elementTags
    }


def test_Mesh_Merge_tags_follow_the_kept_duplicates(unit_square, subtests):
    """Merging a mesh with itself: each tag points at the copy that survived."""
    mesh = unit_square(0.4)

    merged = Mesh.Merge([mesh, mesh])

    for elemType in [ElemType.POINT, ElemType.SEG2, ElemType.QUAD4]:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh.dict_groupElem[elemType]
            mergedGroup = merged.dict_groupElem[elemType]
            assert mergedGroup.elementTags == groupElem.elementTags
            assert _tagged_elements(mergedGroup) == _tagged_elements(groupElem)
