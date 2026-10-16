# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest
import pyvista as pv
from PIL import Image

from EasyFEA import ElemType, Mesh, Models, PyVista, Simulations
from EasyFEA.FEM import GroupElemFactory
from EasyFEA.Geoms import Circle, CircleArc, Domain, Line, Point, Points

from ._simus import L, H, element_tags, far_meshes


def _meshes(plotter: pv.Plotter, kind=pv.DataSet) -> list:
    return [mesh for mesh in plotter.meshes if isinstance(mesh, kind)]


def _actor(plotter: pv.Plotter, dataset):
    """The actor drawing `dataset`."""
    for actor in plotter.renderer.actors.values():
        if isinstance(actor, pv.Actor) and actor.mapper.dataset is dataset:
            return actor
    raise KeyError("dataset not drawn")


def _n_legend(plotter: pv.Plotter) -> int:
    return plotter.legend.GetNumberOfEntries() if plotter.legend else 0


def _legend_labels(plotter: pv.Plotter) -> set[str]:
    legend = plotter.legend
    return {legend.GetEntryString(i) for i in range(_n_legend(plotter))}


def _has_title(plotter: pv.Plotter) -> bool:
    return "title" in plotter.renderer.actors


# ----------------------------------------------
# Plot
# ----------------------------------------------
def test_Plot_colors_the_nodes_by_the_result_times_coef(bar_simu):
    plotter = PyVista.Plot(bar_simu, "ux", coef=2.0)

    (grid,) = _meshes(plotter)
    assert np.allclose(grid.point_data["ux"], 2 * bar_simu.Result("ux"))
    assert list(plotter.scalar_bars.keys()) == ["ux"]


def test_Plot_colors_the_elements_by_the_result(bar_simu):
    plotter = PyVista.Plot(bar_simu, "Svm", nodeValues=False)

    (grid,) = _meshes(plotter)
    assert np.allclose(grid.cell_data["Svm"], bar_simu.Result("Svm", nodeValues=False))


def test_Plot_averages_the_Gauss_points_of_an_element_array(bar_simu):
    Ne = bar_simu.mesh.Ne
    values = np.arange(Ne)[:, None] + np.array([-1.0, 0.0, 0.0, 1.0])

    plotter = PyVista.Plot(bar_simu, values, nodeValues=False)

    (grid,) = _meshes(plotter)
    assert np.allclose(grid.cell_data["array"], np.arange(Ne))


def test_Plot_spreads_a_per_group_value_over_its_elements(bar_simu):
    mesh = bar_simu.mesh

    plotter = PyVista.Plot(mesh, {mesh.groupElem: 3.0}, nodeValues=False)

    (grid,) = _meshes(plotter)
    assert np.allclose(grid.cell_data["array"], 3.0)


def test_Plot_colorbarTitle_and_scalar_bar_kwargs_name_the_colorbar(bar_simu):
    plotter = PyVista.Plot(
        bar_simu,
        "ux",
        colorbarTitle="u",
        verticalColorbar=False,
        scalar_bar_kwargs={"title": "u_x"},
    )

    (bar,) = plotter.scalar_bars.values()
    assert list(plotter.scalar_bars.keys()) == ["u_x"]
    assert bar.GetOrientation() == 0  # horizontal


def test_Plot_plotColorbar_false_draws_no_colorbar(bar_simu):
    plotter = PyVista.Plot(bar_simu, "ux", plotColorbar=False)

    assert len(plotter.scalar_bars) == 0


def test_Plot_without_result_draws_solid_edged_faces(bar_simu):
    plotter = PyVista.Plot(
        bar_simu.mesh,
        color="red",
        plotMesh=True,
        alpha=0.5,
        linewidth=3,
        title="mesh",
    )

    (grid,) = _meshes(plotter, pv.UnstructuredGrid)
    (edges,) = _meshes(plotter, pv.PolyData)
    prop = _actor(plotter, grid).prop
    assert grid.n_cells == bar_simu.mesh.Ne
    assert prop.color == pv.Color("red")
    assert prop.opacity == 0.5
    assert _actor(plotter, edges).prop.line_width == 3
    assert _has_title(plotter)
    assert len(plotter.scalar_bars) == 0


def test_Plot_plotMesh_draws_each_element_outline_not_the_vtk_edges(subtests, mesh_of):
    # loops per element: 1 in 2D; in 3D, only the boundary faces
    for elemType in (ElemType.TRI6, ElemType.QUAD8, ElemType.TETRA10, ElemType.HEXA8):
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem

            plotter = PyVista.Plot(mesh, plotMesh=True)

            (grid,) = _meshes(plotter, pv.UnstructuredGrid)
            (edges,) = _meshes(plotter, pv.PolyData)
            assert not _actor(plotter, grid).prop.show_edges
            # every loop goes through all the nodes of its face, mid-side nodes included
            nPoly = groupElem.surfaces.shape[1]
            assert (
                edges.lines.reshape(-1, nPoly + 2)[:, 0].tolist()
                == [nPoly + 1] * edges.n_lines
            )
            if mesh.dim == 2:
                assert edges.n_lines == mesh.Ne
            else:
                assert (
                    edges.n_lines
                    == grid.extract_surface(
                        nonlinear_subdivision=0, algorithm="dataset_surface"
                    ).n_cells
                )


def test_Plot_plotMesh_keeps_the_vtk_edges_of_a_pyvista_dataset():
    plotter = PyVista.Plot(pv.Sphere(), plotMesh=True)

    (sphere,) = _meshes(plotter)
    assert _actor(plotter, sphere).prop.show_edges


def test_Plot_colors_a_pyvista_dataset_by_its_named_array():
    sphere = pv.Sphere()
    sphere["z"] = sphere.points[:, 2]

    plotter = PyVista.Plot(sphere, "z")

    assert list(plotter.scalar_bars.keys()) == ["z"]


def test_Plot_draws_a_pyvista_dataset_solid_for_an_unknown_array():
    sphere = pv.Sphere()

    plotter = PyVista.Plot(sphere, "z")

    assert len(plotter.scalar_bars) == 0


def test_Plot_draws_on_the_given_plotter(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh)

    assert PyVista.Plot(bar_simu, "ux", plotter=plotter) is plotter
    assert len(_meshes(plotter)) == 2


def test_Plot_showGrid_adds_the_axes_grid(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh, showGrid=True)

    assert plotter.renderer.cube_axes_actor is not None


def test_Plot_refuses_line_width(bar_simu):
    with pytest.raises(ValueError, match="linewidth"):
        PyVista.Plot(bar_simu.mesh, line_width=2)


def test_Plot_forwards_other_kwargs_to_add_mesh(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh, ambient=0.25)

    (grid,) = _meshes(plotter)
    assert _actor(plotter, grid).prop.ambient == 0.25


def test_Plot_bounds_of_four_values_fix_a_parallel_2D_view(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh, bounds=(0, 1, 0, 0.5))

    assert plotter.camera.parallel_projection
    assert np.allclose(plotter.camera.focal_point, (0.5, 0.25, 0))


# ----------------------------------------------
# Plot_Mesh
# ----------------------------------------------
def test_Plot_Mesh_draws_one_cell_per_element(subtests, mesh_of):
    for elemType in (ElemType.TRI3, ElemType.HEXA8):
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)

            plotter = PyVista.Plot_Mesh(mesh)

            (grid,) = _meshes(plotter, pv.UnstructuredGrid)
            assert grid.n_cells == mesh.Ne
            assert _actor(plotter, grid).prop.color == pv.Color("cyan")


def test_Plot_Mesh_draws_the_deformed_mesh(bar_simu):
    ux = bar_simu.Result("ux")

    plotter = PyVista.Plot_Mesh(bar_simu, deformFactor=10)

    (grid,) = _meshes(plotter, pv.UnstructuredGrid)
    (edges,) = _meshes(plotter, pv.PolyData)
    assert np.isclose(grid.bounds[1], L + 10 * ux.max())
    assert np.isclose(edges.bounds[1], grid.bounds[1])


def test_Plot_Mesh_view_covers_every_mesh_drawn():
    meshBig, meshSmall = far_meshes()

    plotter = PyVista.Plot_Mesh(meshBig)
    PyVista.Plot_Mesh(meshSmall, plotter=plotter)

    focal = np.asarray(plotter.camera.focal_point)
    assert np.allclose(focal[:2], 10.5, atol=0.5)


# ----------------------------------------------
# Plot_Nodes
# ----------------------------------------------
def test_Plot_Nodes_marks_the_given_nodes(bar_simu):
    mesh = bar_simu.mesh
    nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)

    plotter = PyVista.Plot_Nodes(mesh, nodes, color="blue")

    points = _meshes(plotter, pv.PolyData)[-1]
    assert np.allclose(points.points, mesh.coord[nodes])
    assert _actor(plotter, points).prop.color == pv.Color("blue")


def test_Plot_Nodes_marks_every_node_by_default(bar_simu):
    mesh = bar_simu.mesh

    plotter = PyVista.Plot_Nodes(mesh, title="nodes")

    points = _meshes(plotter, pv.PolyData)[-1]
    assert np.allclose(points.points, mesh.coord)
    assert _has_title(plotter)


def test_Plot_Nodes_marks_points_given_by_their_coordinates(bar_simu):
    points = np.array([[0.5, 0.5, 0.0], [1.0, 0.25, 0.0]])

    plotter = PyVista.Plot_Nodes(bar_simu.mesh, points)

    assert np.allclose(_meshes(plotter, pv.PolyData)[-1].points, points)


def test_Plot_Nodes_showId_labels_each_node(bar_simu):
    nodes = np.array([0, 3, 5])

    plotter = PyVista.Plot_Nodes(bar_simu.mesh, nodes, showId=True)

    labelled = [m for m in plotter.meshes if "myLabels" in m.array_names]
    assert list(labelled[0]["myLabels"]) == ["0", "3", "5"]


def test_Plot_Nodes_returns_none_for_unusable_nodes(subtests, bar_simu):
    unusable = {
        "empty": [],
        "too many": np.arange(1000),
        "not 3D points": np.zeros((2, 2)),
    }
    for case, nodes in unusable.items():
        with subtests.test(case=case):
            assert PyVista.Plot_Nodes(bar_simu.mesh, nodes) is None


# ----------------------------------------------
# Plot_Elements
# ----------------------------------------------
def test_Plot_Elements_draws_the_elements_made_of_the_nodes(bar_simu):
    mesh = bar_simu.mesh
    nodes = mesh.Nodes_Conditions(lambda x, y, z: x < L / 4)

    plotter = PyVista.Plot_Elements(mesh, nodes, showId=True, title="first column")

    (grid,) = _meshes(plotter, pv.UnstructuredGrid)
    # one column of the 6 × 3 grid
    assert grid.n_cells == 3
    labelled = [m for m in plotter.meshes if "myLabels" in m.array_names]
    assert len(labelled[0]["myLabels"]) == 3


def test_Plot_Elements_dimElem_1_draws_the_edges(bar_simu):
    mesh = bar_simu.mesh

    plotter = PyVista.Plot_Elements(mesh, dimElem=1)

    (grid,) = _meshes(plotter, pv.UnstructuredGrid)
    assert grid.n_cells == mesh.Get_list_groupElem(1)[0].Ne


def test_Plot_Elements_draws_nothing_when_no_element_is_made_of_the_nodes(bar_simu):
    mesh = bar_simu.mesh
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    plotter = PyVista.Plot_Elements(mesh, nodesRight)

    assert _meshes(plotter) == []


# ----------------------------------------------
# Plot_Arrows
# ----------------------------------------------
def test_Plot_Arrows_draws_each_vector_from_its_node(bar_simu):
    mesh = bar_simu.mesh
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    vectors = np.tile([1.0, 0, 0], (nodesRight.size, 1))

    plotter = PyVista.Plot_Arrows(mesh, nodesRight, vectors, magnitudeCoef=0.2)

    (arrows,) = _meshes(plotter)
    magnitude = mesh._Get_realistic_vector_magnitude(0.2)
    assert np.isclose(arrows.bounds[0], L)
    assert np.isclose(arrows.bounds[1], L + magnitude)


def test_Plot_Arrows_needs_one_vector_per_node(bar_simu):
    with pytest.raises(AssertionError):
        PyVista.Plot_Arrows(bar_simu.mesh, [0, 1], np.ones((3, 3)))


# ----------------------------------------------
# Plot_BoundaryConditions
# ----------------------------------------------
def test_Plot_BoundaryConditions_ends_each_load_arrow_on_its_nodes(bar_simu):
    plotter = PyVista.Plot_BoundaryConditions(bar_simu, title="bar")

    *_, clamp, load = _meshes(plotter, pv.PolyData)
    nodesLeft = bar_simu.mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    assert np.allclose(
        np.sort(clamp.points[:, 1]), np.sort(bar_simu.mesh.coord[nodesLeft, 1])
    )
    assert np.isclose(load.bounds[1], L)
    assert load.bounds[0] < L
    assert _n_legend(plotter) == 2


def test_Plot_BoundaryConditions_draws_one_arrow_for_a_load_on_every_node(bar_simu):
    mesh = bar_simu.mesh
    bar_simu.Bc_Init()
    bar_simu.add_volumeLoad(mesh.nodes, [1], ["y"])

    plotter = PyVista.Plot_BoundaryConditions(bar_simu, plotLegend=False)

    arrow = _meshes(plotter, pv.PolyData)[-1]
    assert np.isclose(arrow.bounds[3], H / 2)
    assert np.isclose(arrow.center[0], L / 2)
    assert _n_legend(plotter) == 0


def test_Plot_BoundaryConditions_marks_a_temperature_with_points(bar_simu):
    mesh = bar_simu.mesh
    simu = Simulations.Thermal(mesh, Models.Thermal(1.0))
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodesLeft, [1.0], ["t"])

    plotter = PyVista.Plot_BoundaryConditions(simu)

    points = _meshes(plotter, pv.PolyData)[-1]
    assert np.allclose(points.points[:, 0], 0)
    assert points.n_points == nodesLeft.size


def test_Plot_BoundaryConditions_draws_a_moment_as_a_rotation_arrow(frame_simu):
    plotter = PyVista.Plot_BoundaryConditions(frame_simu)

    # rz → along z, starting at the tip (L, L)
    (moment,) = [m for m in _meshes(plotter, pv.PolyData) if m.bounds[5] > 0]
    assert np.isclose(moment.bounds[4], 0, atol=1e-9)
    assert np.allclose(moment.center[:2], (L, L), atol=0.1)
    assert _legend_labels(plotter) == {
        "Dirichlet  [x, y, rz]",
        "Neumann  [rz]",
        "Connection Hinged [x, y]",
    }


def test_Plot_BoundaryConditions_needs_a_simulation(bar_simu):
    assert PyVista.Plot_BoundaryConditions(bar_simu.mesh) is None


# ----------------------------------------------
# Plot_Tags
# ----------------------------------------------
def test_Plot_Tags_draws_and_labels_each_tag(subtests, mesh_of):
    for elemType in (ElemType.QUAD4, ElemType.HEXA8):
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)

            plotter = PyVista.Plot_Tags(mesh, title="tags", showGrid=True)

            assert _n_legend(plotter) == len(element_tags(mesh))
            assert len(_meshes(plotter, pv.UnstructuredGrid)) == 1 + len(
                element_tags(mesh)
            )


def test_Plot_Tags_useColorCycler_colors_each_tag_from_tab10(make_mesh_2D):
    mesh = make_mesh_2D(elemType=ElemType.QUAD4)

    plotter = PyVista.Plot_Tags(
        mesh, useColorCycler=True, showId=False, plotLegend=False
    )

    tagged = _meshes(plotter, pv.UnstructuredGrid)[1:]
    colors = {_actor(plotter, grid).prop.color.hex_rgb for grid in tagged}
    assert colors <= set(PyVista.tab10_colors)
    assert _n_legend(plotter) == 0


def test_Plot_Tags_returns_none_withoutelement_tags(bar_simu):
    groupElem = bar_simu.mesh.groupElem
    untagged = GroupElemFactory.Create(
        groupElem.elemType, groupElem.connect, groupElem.coord
    )

    assert PyVista.Plot_Tags(Mesh({untagged.elemType: untagged})) is None


# ----------------------------------------------
# Plot_Geoms
# ----------------------------------------------
def test_Plot_Geoms_draws_each_geom_with_its_own_color_and_name():
    line = Line((0, 0), (1, 0))
    circle = Circle((0, 0), 1)
    domain = Domain((0, 0), (1, 1, 1))

    plotter = PyVista.Plot_Geoms([line, circle], domain, title="geoms")

    meshes = _meshes(plotter)
    assert len(meshes) == 4  # the circle is two arcs
    colors = {_actor(plotter, m).prop.color.hex_rgb for m in meshes}
    assert len(colors) == 3
    assert _n_legend(plotter) == 3


def test_Plot_Geoms_color_and_label_apply_to_every_geom():
    arc = CircleArc((1, 0), (-1, 0), center=(0, 0))
    corners = Points([(0, 0), (1, 0), Point(1, 1, r=0.2)])

    plotter = PyVista.Plot_Geoms(
        arc, corners, color="red", label="geom", plotLegend=False
    )

    colors = {_actor(plotter, m).prop.color for m in _meshes(plotter)}
    assert colors == {pv.Color("red")}


def test_Plot_Geoms_draws_a_point_and_a_contour():
    point = Point(1, 1)
    contour = Points([(0, 0), (1, 0), (0, 1)]).Get_Contour()

    plotter = PyVista.Plot_Geoms(point, contour)

    assert len(_meshes(plotter)) == 1 + 3


# ----------------------------------------------
# Movie_simu / Movie_func
# ----------------------------------------------
def test_Movie_simu_writes_one_frame_per_iteration_up_to_N(bar_simu, tmp_path):
    PyVista.Movie_simu(bar_simu, "ux", str(tmp_path), "ux.gif", N=2, plotMesh=True)

    with Image.open(tmp_path / "ux.gif") as gif:
        assert gif.n_frames == 2


def test_Movie_simu_needs_a_simulation(bar_simu, tmp_path):
    PyVista.Movie_simu(bar_simu.mesh, "ux", str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_Movie_func_calls_func_once_per_frame(tmp_path):
    calls = []

    def func(plotter, i):
        calls.append(i)
        plotter.add_mesh(pv.Sphere(center=(i, 0, 0)))

    PyVista.Movie_func(func, 3, str(tmp_path), "f.gif", fps=5)

    assert calls == [0, 1, 2]
    with Image.open(tmp_path / "f.gif") as gif:
        assert gif.n_frames == 3


def test_Movie_func_keeps_the_view_of_the_first_frame(tmp_path):
    positions = []

    def func(plotter, i):
        sphere = pv.Sphere(center=(10 * i, 0, 0))
        PyVista.Plot(sphere, plotter=plotter)
        positions.append(plotter.camera_position)

    PyVista.Movie_func(func, 3, str(tmp_path), "f.gif")

    assert positions[1] == positions[0] and positions[2] == positions[0]


def test_Movie_func_writes_an_mp4(tmp_path):
    imageio = pytest.importorskip("imageio", reason="needs imageio-ffmpeg")
    pytest.importorskip("imageio_ffmpeg")

    PyVista.Movie_func(
        lambda plotter, i: plotter.add_mesh(pv.Sphere()), 2, str(tmp_path), "f.mp4"
    )

    with imageio.get_reader(tmp_path / "f.mp4") as video:
        assert video.count_frames() == 2


# ----------------------------------------------
# _Plotter
# ----------------------------------------------
def test_Plotter_honours_off_screen(monkeypatch):
    monkeypatch.setattr(pv, "OFF_SCREEN", False)

    plotter = PyVista._Plotter(off_screen=True)

    assert plotter.off_screen
    plotter.close()


def test_Plotter_opens_the_given_subplots():
    plotter = PyVista._Plotter(shape=(1, 2), add_axes=False, linkViews=False)

    assert len(plotter.renderers) == 2
    plotter.close()


# ----------------------------------------------
# _Annotate
# ----------------------------------------------
def test_Annotate_adds_the_title_and_grid(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh)

    PyVista._Annotate(plotter, 2, "title", True)

    assert _has_title(plotter)
    assert plotter.renderer.cube_axes_actor is not None


# ----------------------------------------------
# _setCameraPosition
# ----------------------------------------------
def test_setCameraPosition_looks_at_a_2D_scene_from_z(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh)

    PyVista._setCameraPosition(plotter, 2)

    position, focal, _ = plotter.camera_position
    assert np.allclose(np.subtract(position, focal)[:2], 0)


def test_setCameraPosition_tilts_a_3D_scene(bar_simu_3D):
    plotter = PyVista.Plot(bar_simu_3D.mesh)

    PyVista._setCameraPosition(plotter, 3, elevation=30, azimuth=0)

    assert plotter.camera.elevation == 30


def test_setCameraPosition_bounds_frame_the_camera(bar_simu):
    plotter = PyVista.Plot(bar_simu.mesh)

    PyVista._setCameraPosition(plotter, 2, bounds=(0, 1, 0, 1, 0, 0))

    assert plotter.camera.parallel_projection
    assert np.allclose(plotter.camera.focal_point, (0.5, 0.5, 0))


# ----------------------------------------------
# _pvMesh
# ----------------------------------------------
def test_pvMesh_holds_the_result_on_nodes_or_elements(bar_simu):
    nodal = PyVista._pvMesh(bar_simu, "ux")
    perElement = PyVista._pvMesh(bar_simu, "Svm", nodeValues=False)

    assert np.allclose(nodal.point_data["ux"], bar_simu.Result("ux"))
    assert np.allclose(
        perElement.cell_data["Svm"], bar_simu.Result("Svm", nodeValues=False)
    )


def test_pvMesh_names_an_array_result_array(bar_simu):
    values = np.arange(bar_simu.mesh.Nn, dtype=float)

    grid = PyVista._pvMesh(bar_simu.mesh, values)

    assert grid.active_scalars_name == "array"
    assert np.allclose(grid["array"], values)


def test_pvMesh_clipAxis_keeps_the_half_behind_the_center(bar_simu):
    grid = PyVista._pvMesh(bar_simu.mesh, clipAxis="x")

    assert np.isclose(grid.bounds[1], L / 2)


# ----------------------------------------------
# _pvGeom
# ----------------------------------------------
def test_pvGeom_turns_each_geom_into_pyvista_data(subtests):
    line = Line((0, 0), (1, 0))
    arc = CircleArc((1, 0), (-1, 0), center=(0, 0))
    point = Point(1, 1)
    domain = Domain((0, 0), (1, 1, 1))
    circle = Circle((0, 0), 1)
    corners = Points([(0, 0), (1, 0), (0, 1)])
    contour = corners.Get_Contour()
    cases = [
        (line, 1),
        (arc, 1),
        (point, 1),
        (domain, 1),
        (circle, 2),
        (corners, 2),  # open, as in Matplotlib
        (contour, 3),
    ]
    for geom, nData in cases:
        with subtests.test(geom=type(geom).__name__):
            data = PyVista._pvGeom(geom)

            data = data if isinstance(data, list) else [data]
            assert len(data) == nData
            assert all(isinstance(d, pv.DataSet) for d in data)


def test_pvGeom_returns_none_for_an_unknown_object():
    assert PyVista._pvGeom("circle") is None
