# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation, colors
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from EasyFEA import ElemType, Matplotlib, Mesh, Models, Simulations, Tic
from EasyFEA.FEM import GroupElemFactory
from EasyFEA.Geoms import Circle, CircleArc, Domain, Line, Point

from Geoms._shapes import rounded_corners
from Simulations._simus import LOADS

from ._simus import L, H, element_tags, far_meshes

needs_ffmpeg = pytest.mark.skipif(
    not animation.FFMpegWriter.isAvailable(), reason="needs ffmpeg"
)


def _drawn(ax, kind) -> list:
    return [c for c in ax.collections if type(c) is kind]


def _n_drawn(collection) -> int:
    """Number of polygons or segments in `collection`, 3D ones included."""
    collection.axes.figure.canvas.draw()
    return len(collection.get_paths())


def _line_points(line) -> np.ndarray:
    return np.column_stack(line.get_data())


def _texts(ax) -> list[str]:
    return [text.get_text() for text in ax.texts]


# ----------------------------------------------
# _Node_to_element_values
# ----------------------------------------------
def test_Node_to_element_values_gives_the_centroid_value_of_an_affine_field(
    make_mesh_2D,
):
    mesh = make_mesh_2D(elemType=ElemType.TRI3)
    x = mesh.coord[:, 0]

    values = Matplotlib._Node_to_element_values(mesh, 2 * x + 1, 2)

    centroids = mesh.coord[mesh.connect].mean(1)
    assert np.allclose(values, 2 * centroids[:, 0] + 1)


# ----------------------------------------------
# Plot
# ----------------------------------------------
def test_Plot_colors_each_element_by_the_result_times_coef(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", nodeValues=False, coef=2.0)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_array(), 2 * bar_simu.Result("Svm", nodeValues=False))


def test_Plot_contours_a_nodal_result_over_its_whole_range(bar_simu):
    ux = bar_simu.Result("ux")

    ax = Matplotlib.Plot(bar_simu, "ux", nColors=11)

    levels = ax.collections[-1].levels
    assert levels.size == 11
    assert levels[0] <= ux.min() and levels[-1] >= ux.max()


def test_Plot_nColors_splits_the_colorbar_into_as_many_levels(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", nodeValues=False, nColors=5)

    (pc,) = _drawn(ax, PolyCollection)
    assert isinstance(pc.norm, colors.BoundaryNorm)
    assert pc.norm.boundaries.size == 5


def test_Plot_clim_fixes_the_color_range(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", nodeValues=False, clim=(0, 100))

    (pc,) = _drawn(ax, PolyCollection)
    assert pc.get_clim() == (0, 100)
    assert pc.colorbar.ax.get_ylim() == (0, 100)


def test_Plot_averages_the_Gauss_points_of_an_element_array(bar_simu):
    Ne = bar_simu.mesh.Ne
    values = np.arange(Ne)[:, None] + np.array([-1.0, 0.0, 0.0, 1.0])

    ax = Matplotlib.Plot(bar_simu, values, nodeValues=False)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_array(), np.arange(Ne))


def test_Plot_spreads_a_per_group_value_over_its_elements(bar_simu):
    mesh = bar_simu.mesh

    ax = Matplotlib.Plot(mesh, {mesh.groupElem: 3.0}, nodeValues=False)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_array(), 3.0)


def test_Plot_colors_the_faces_of_a_3D_mesh_with_their_nodal_mean(bar_simu_3D):
    mesh = bar_simu_3D.mesh
    ux = bar_simu_3D.Result("ux")

    ax = Matplotlib.Plot(bar_simu_3D, "ux", plotMesh=True)

    (pc,) = _drawn(ax, Poly3DCollection)
    assert pc.get_array().size == sum(g.Ne for g in mesh.Get_list_groupElem(2))
    assert np.allclose(pc.get_array(), Matplotlib._Node_to_element_values(mesh, ux, 2))


def test_Plot_colors_an_inclined_2D_mesh_with_element_values(make_mesh_2D):
    mesh = make_mesh_2D(elemType=ElemType.QUAD4)
    mesh.Rotate(30, direction=(1, 0, 0))

    ax = Matplotlib.Plot(mesh, np.arange(mesh.Ne, dtype=float), nodeValues=False)

    (pc,) = _drawn(ax, Poly3DCollection)
    assert np.allclose(pc.get_array(), np.arange(mesh.Ne))


def test_Plot_without_result_draws_solid_faces(bar_simu):
    ax = Matplotlib.Plot(bar_simu.mesh, color="red", plotMesh=True)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_facecolor(), colors.to_rgba("red"))
    assert pc.get_array() is None


def test_Plot_alpha_fades_the_faces_but_not_the_edges(bar_simu):
    ax = Matplotlib.Plot(bar_simu.mesh, color="red", alpha=0.3, plotMesh=True)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_facecolor()[:, 3], 0.3)
    assert np.allclose(pc.get_edgecolor()[:, 3], 1.0)


def test_Plot_without_result_draws_solid_faces_in_3D(bar_simu_3D):
    ax = Matplotlib.Plot(bar_simu_3D.mesh, color="red")

    (pc,) = _drawn(ax, Poly3DCollection)
    assert pc.get_array() is None


def test_Plot_plotMesh_adds_the_element_edges_under_a_result(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "ux", plotMesh=True)

    (edges,) = _drawn(ax, LineCollection)
    assert len(edges.get_segments()) == bar_simu.mesh.Ne


def test_Plot_plotNodes_marks_every_node(bar_simu):
    mesh = bar_simu.mesh

    ax = Matplotlib.Plot(mesh, plotNodes=True)

    (line,) = ax.lines
    assert np.allclose(_line_points(line), mesh.coord[:, :2])


def test_Plot_colors_each_segment_of_a_1D_mesh(subtests, mesh_of):
    for elemType in ElemType.Get_1D():
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)

            ax = Matplotlib.Plot(mesh, np.arange(mesh.Ne, dtype=float), plotMesh=True)

            (pc,) = _drawn(ax, LineCollection)
            assert np.allclose(pc.get_array(), np.arange(mesh.Ne))
            assert len(ax.lines[0].get_xdata()) == mesh.Nn


def test_Plot_marks_the_nodes_of_a_1D_mesh_lying_in_3D():
    line = Line((0, 0), (1, 1, 1), 0.25)
    mesh = line.Mesh_1D(ElemType.SEG2)

    ax = Matplotlib.Plot(mesh, plotMesh=True)

    assert ax.name == "3d"
    assert len(ax.lines[0].get_xdata()) == mesh.Nn


def test_Plot_titles_the_figure_after_the_result(subtests, bar_simu):
    cases = [
        ("Svm", True, r"$\sigma_{vm}$"),
        ("Exx", False, r"$\epsilon_{xx}^{e}$"),
        ("ux", True, "$ux$"),
    ]
    for result, nodeValues, title in cases:
        with subtests.test(result=result):
            ax = Matplotlib.Plot(bar_simu, result, nodeValues=nodeValues)

            assert ax.get_title() == title


def test_Plot_titles_a_damage_phi_with_a_colorbar_reaching_1(phasefield_simu):
    ax = Matplotlib.Plot(phasefield_simu, "damage", nodeValues=False)

    (pc,) = _drawn(ax, PolyCollection)
    assert ax.get_title() == r"$\phi^{e}$"
    assert pc.colorbar.ax.get_ylim()[1] >= 1


def test_Plot_title_replaces_the_result_name(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", title="stress")

    assert ax.get_title() == "stress"


def test_Plot_horizontal_colorbar_close_to_the_axes(bar_simu):
    ax = Matplotlib.Plot(
        bar_simu,
        "Svm",
        nodeValues=False,
        verticalColorbar=False,
        colorbarIsClose=True,
    )

    (pc,) = _drawn(ax, PolyCollection)
    assert pc.colorbar.orientation == "horizontal"


def test_Plot_plotColorbar_false_draws_no_colorbar(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", nodeValues=False, plotColorbar=False)

    (pc,) = _drawn(ax, PolyCollection)
    assert pc.colorbar is None


def test_Plot_showGrid_shows_the_grid(bar_simu):
    ax = Matplotlib.Plot(bar_simu.mesh, showGrid=True)

    assert all(line.get_visible() for line in ax.get_xgridlines())


def test_Plot_on_3D_axes_draws_a_2D_mesh_in_3D(bar_simu):
    ax = Matplotlib.Init_Axes(3)

    Matplotlib.Plot(bar_simu.mesh, ax=ax)

    assert len(_drawn(ax, Poly3DCollection)) == 1


def test_Plot_folder_saves_the_figure_after_the_result(subtests, bar_simu, tmp_path):
    for result, filename in [("Svm", "Svm"), (None, "mesh")]:
        with subtests.test(filename=filename):
            Matplotlib.Plot(bar_simu, result, folder=str(tmp_path))

            assert (tmp_path / f"{filename}.pdf").is_file()


def test_Plot_refuses_lw(bar_simu):
    with pytest.raises(ValueError, match="linewidth"):
        Matplotlib.Plot(bar_simu.mesh, lw=2)


def test_Plot_forwards_other_kwargs_to_matplotlib(bar_simu):
    ax = Matplotlib.Plot(bar_simu.mesh, color="c", hatch="/")

    assert ax.collections[0].get_hatch() == "/"


def test_Plot_bounds_of_four_values_fix_a_2D_view(bar_simu):
    ax = Matplotlib.Plot(bar_simu.mesh, bounds=(0, 1, 0, 0.5))

    assert ax.get_xlim() == (0, 1)
    assert ax.get_ylim() == (0, 0.5)


def test_Plot_bounds_of_five_values_are_refused(bar_simu):
    with pytest.raises(ValueError, match="bounds"):
        Matplotlib.Plot(bar_simu.mesh, bounds=(0, 1, 0, 1, 0))


# ----------------------------------------------
# Plot_Mesh
# ----------------------------------------------
def test_Plot_Mesh_draws_one_shape_per_element(subtests, mesh_of):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D() + ElemType.Get_3D():
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)
            dim = 2 if mesh.dim == 3 else mesh.dim
            Ne = sum(g.Ne for g in mesh.Get_list_groupElem(dim))

            ax = Matplotlib.Plot_Mesh(mesh)

            assert _n_drawn(ax.collections[0]) == Ne


def test_Plot_Mesh_titles_the_figure_with_the_mesh(bar_simu):
    mesh = bar_simu.mesh

    ax = Matplotlib.Plot_Mesh(mesh)

    assert ax.get_title() == str(mesh).replace("\n", ", ")


def test_Plot_Mesh_marks_the_nodes_of_a_1D_mesh():
    line = Line((0, 0), (1, 0), 0.25)
    mesh = line.Mesh_1D(ElemType.SEG2)

    ax = Matplotlib.Plot_Mesh(mesh)

    assert np.allclose(_line_points(ax.lines[0]), mesh.coord[:, :2])


def test_Plot_Mesh_draws_the_deformed_mesh_in_red_over_the_undeformed(bar_simu):
    ux = bar_simu.Result("ux")

    ax = Matplotlib.Plot_Mesh(bar_simu, deformFactor=10, plotNodes=True)

    deformed, undeformed = _drawn(ax, LineCollection)
    assert np.allclose(deformed.get_color(), colors.to_rgba("red"))
    xDef = np.concatenate(deformed.get_segments())[:, 0]
    x = np.concatenate(undeformed.get_segments())[:, 0]
    assert np.isclose(x.max(), L)
    assert np.isclose(xDef.max(), L + 10 * ux.max())
    nodes, nodesDef = ax.lines
    assert np.isclose(nodesDef.get_xdata().max(), L + 10 * ux.max())


def test_Plot_Mesh_draws_a_deformed_3D_mesh_in_3D(bar_simu_3D):
    ax = Matplotlib.Plot_Mesh(bar_simu_3D, deformFactor=10)

    assert ax.name == "3d"
    assert len(ax.collections) == 2


def test_Plot_Mesh_folder_saves_mesh_pdf(bar_simu, tmp_path):
    Matplotlib.Plot_Mesh(bar_simu.mesh, folder=str(tmp_path))

    assert (tmp_path / "mesh.pdf").is_file()


def test_Plot_Mesh_3D_view_covers_every_mesh_drawn():
    meshBig, meshSmall = far_meshes()

    ax = Matplotlib.Plot_Mesh(meshBig)
    Matplotlib.Plot_Mesh(meshSmall, ax=ax)

    xmin, xmax = ax.get_xlim()
    assert xmin <= 0 and xmax >= 21


def test_Plot_Mesh_bounds_fix_the_3D_view():
    meshBig, _ = far_meshes()

    ax = Matplotlib.Plot_Mesh(meshBig, bounds=(0, 5, 0, 5, 0, 5))

    assert ax.get_xlim() == (0, 5)
    assert ax.get_zlim() == (0, 5)


# ----------------------------------------------
# _Plot_obj
# ----------------------------------------------
def test_Plot_obj_draws_transparent_gray_faces(bar_simu):
    ax = Matplotlib._Plot_obj(bar_simu, alpha=0.5)

    (pc,) = _drawn(ax, PolyCollection)
    assert np.allclose(pc.get_facecolor(), colors.to_rgba("gray", 0.5))


# ----------------------------------------------
# Plot_Nodes
# ----------------------------------------------
def test_Plot_Nodes_marks_each_given_node_once(bar_simu):
    mesh = bar_simu.mesh
    nodes = mesh.Nodes_Conditions(lambda x, y, z: x == 0)

    ax = Matplotlib.Plot_Nodes(mesh, np.concatenate([nodes, nodes]))

    points = _line_points(ax.lines[0])
    assert len(points) == len(nodes)
    assert np.allclose(points[:, 0], 0)


def test_Plot_Nodes_marks_every_node_by_default(bar_simu):
    mesh = bar_simu.mesh

    ax = Matplotlib.Plot_Nodes(mesh)

    assert np.allclose(_line_points(ax.lines[0]), mesh.coord[:, :2])


def test_Plot_Nodes_showId_writes_each_node_number(bar_simu):
    nodes = np.array([0, 3, 5])

    ax = Matplotlib.Plot_Nodes(bar_simu.mesh, nodes, showId=True)

    assert sorted(_texts(ax)) == ["0", "3", "5"]


def test_Plot_Nodes_marks_the_displaced_nodes(bar_simu):
    ux = bar_simu.Result("ux")

    ax = Matplotlib.Plot_Nodes(bar_simu, deformFactor=10)

    assert np.isclose(ax.lines[0].get_xdata().max(), L + 10 * ux.max())


def test_Plot_Nodes_draws_nothing_for_no_node(bar_simu):
    ax = Matplotlib.Plot_Nodes(bar_simu.mesh, [])

    assert len(ax.lines) == 0


def test_Plot_Nodes_on_3D_axes_marks_the_nodes_in_3D(bar_simu):
    ax = Matplotlib.Init_Axes(3)

    Matplotlib.Plot_Nodes(bar_simu.mesh, ax=ax)

    assert np.allclose(np.column_stack(ax.lines[0].get_data_3d()), bar_simu.mesh.coord)


def test_Plot_Nodes_3D_view_covers_every_mesh_drawn():
    meshBig, meshSmall = far_meshes()

    ax = Matplotlib.Plot_Mesh(meshBig)
    Matplotlib.Plot_Nodes(meshSmall, ax=ax)

    xmin, xmax = ax.get_xlim()
    assert xmin <= 0 and xmax >= 21


# ----------------------------------------------
# Plot_Elements
# ----------------------------------------------
def test_Plot_Elements_draws_the_elements_made_of_the_nodes(bar_simu):
    mesh = bar_simu.mesh
    nodes = mesh.Nodes_Conditions(lambda x, y, z: x < L / 4)

    ax = Matplotlib.Plot_Elements(mesh, nodes, showId=True, plotNodes=True)

    (pc,) = _drawn(ax, PolyCollection)
    # one column of the 6 × 3 grid
    assert len(pc.get_paths()) == 3
    assert len(ax.texts) == 3
    assert np.allclose(ax.lines[0].get_xdata().max(), L / 6)


def test_Plot_Elements_dimElem_1_draws_the_edges(bar_simu):
    mesh = bar_simu.mesh

    ax = Matplotlib.Plot_Elements(mesh, dimElem=1)

    (lines,) = _drawn(ax, LineCollection)
    assert len(lines.get_segments()) == mesh.Get_list_groupElem(1)[0].Ne


def test_Plot_Elements_returns_none_without_elements_of_that_dimension(bar_simu):
    assert Matplotlib.Plot_Elements(bar_simu.mesh, dimElem=3) is None


def test_Plot_Elements_draws_nothing_when_no_element_is_made_of_the_nodes(bar_simu):
    mesh = bar_simu.mesh
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)

    ax = Matplotlib.Plot_Elements(mesh, nodesRight)

    assert len(ax.collections) == 0


def test_Plot_Elements_3D_view_covers_every_mesh_drawn():
    meshBig, meshSmall = far_meshes()

    ax = Matplotlib.Plot_Mesh(meshBig)
    Matplotlib.Plot_Elements(meshSmall, ax=ax)

    xmin, xmax = ax.get_xlim()
    assert xmin <= 0 and xmax >= 21


# ----------------------------------------------
# Plot_BoundaryConditions
# ----------------------------------------------
def _markers(ax) -> dict[str, str]:
    return {line.get_label(): line.get_marker() for line in ax.lines}


def test_Plot_BoundaryConditions_marks_each_condition_by_its_direction(bar_simu):
    mesh = bar_simu.mesh
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    nodesTop = mesh.Nodes_Conditions(lambda x, y, z: y == H)
    nodesBottom = mesh.Nodes_Conditions(lambda x, y, z: y == 0)
    bar_simu.Bc_Init()
    bar_simu.add_dirichlet(nodesLeft, [0, 0], ["x", "y"], description="clamp")
    bar_simu.add_neumann(nodesRight, [-1], ["x"], description="left")
    bar_simu.add_neumann(nodesRight, [1], ["x"], description="right")
    bar_simu.add_neumann(nodesTop, [-1], ["y"], description="down")
    bar_simu.add_neumann(nodesBottom, [1], ["y"], description="up")

    ax = Matplotlib.Plot_BoundaryConditions(bar_simu)

    assert _markers(ax) == {
        "Dirichlet clamp [x, y]": "X",
        "Neumann left [x]": "<",
        "Neumann right [x]": ">",
        "Neumann down [y]": "v",
        "Neumann up [y]": "^",
    }
    assert [t.get_text() for t in ax.get_legend().get_texts()] == list(_markers(ax))


def test_Plot_BoundaryConditions_marks_z_and_3D_conditions(bar_simu_3D):
    mesh = bar_simu_3D.mesh
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesRight = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    bar_simu_3D.Bc_Init()
    bar_simu_3D.add_dirichlet(nodesLeft, [0, 0, 0], ["x", "y", "z"], description="a")
    bar_simu_3D.add_neumann(nodesRight, [1], ["z"], description="b")

    ax = Matplotlib.Plot_BoundaryConditions(bar_simu_3D)

    assert _markers(ax) == {"Dirichlet a [x, y, z]": "s", "Neumann b [z]": "d"}


def test_Plot_BoundaryConditions_marks_a_load_on_every_node_at_the_centroid(
    bar_simu,
):
    mesh = bar_simu.mesh
    bar_simu.Bc_Init()
    bar_simu.add_volumeLoad(mesh.nodes, [1], ["y"])

    ax = Matplotlib.Plot_BoundaryConditions(bar_simu, alpha=0.5)

    (line,) = ax.lines
    assert np.allclose(_line_points(line), [[L / 2, H / 2]])
    assert len(_drawn(ax, PolyCollection)) == 1


def test_Plot_BoundaryConditions_marks_a_temperature_with_a_dot(bar_simu):
    mesh = bar_simu.mesh
    simu = Simulations.Thermal(mesh, Models.Thermal(1.0))
    nodesLeft = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    simu.add_dirichlet(nodesLeft, [1.0], ["t"], description="hot")

    ax = Matplotlib.Plot_BoundaryConditions(simu)

    assert _markers(ax) == {"Dirichlet hot [t]": "o"}


def test_Plot_BoundaryConditions_marks_a_beam_connection_with_a_dot(frame_simu):
    ax = Matplotlib.Plot_BoundaryConditions(frame_simu)

    markers = _markers(ax)
    assert markers["Connection Hinged [x, y]"] == "o"


# ----------------------------------------------
# Plot_Geoms
# ----------------------------------------------
def test_Plot_Geoms_draws_on_the_given_axes(subtests):
    line = Line((0, 0), (5, 1))
    domain = Domain((0, 0), (1, 1, 1))
    circle = Circle((0, 0), 5, n=(1, 1, 1))
    arc = CircleArc((3, 1, 3), (-3, 1, 3), center=(0, 1))
    corners = rounded_corners()
    contour = corners.Get_Contour()
    for geom in (line, domain, circle, arc, corners, contour):
        with subtests.test(geom=type(geom).__name__):
            ax = Matplotlib.Plot_Geoms(geom)
            nLines = len(ax.lines)

            moved = geom.Translate(10, copy=True)
            assert Matplotlib.Plot_Geoms(moved, ax=ax) is ax
            assert len(ax.lines) > nLines


def test_Plot_Geoms_label_replaces_each_name():
    line = Line((0, 0), (1, 0))
    circle = Circle((0, 0), 1)

    ax = Matplotlib.Plot_Geoms([line, circle], label="geom", plotPoints=False)

    assert [line.get_label() for line in ax.lines] == ["geom", "geom"]


def test_Plot_Geoms_skips_points():
    point = Point(1, 1)

    ax = Matplotlib.Plot_Geoms(point, plotLegend=False)

    assert ax.name == "rectilinear"
    assert len(ax.lines) == 0


def test_Plot_Geoms_out_of_plane_opens_3D_axes():
    line = Line((0, 0), (1, 1, 1))

    ax = Matplotlib.Plot_Geoms(line)

    assert ax.name == "3d"


# ----------------------------------------------
# Plot_Tags
# ----------------------------------------------
def test_Plot_Tags_writes_each_tag(subtests, mesh_of):
    for elemType in (ElemType.QUAD4, ElemType.HEXA8):
        with subtests.test(elemType=elemType):
            mesh = mesh_of(elemType)

            ax = Matplotlib.Plot_Tags(mesh)

            assert sorted(_texts(ax)) == sorted(element_tags(mesh))


def test_Plot_Tags_useColorCycler_colors_the_surfaces_from_tab10(make_mesh_2D):
    mesh = make_mesh_2D(elemType=ElemType.QUAD4)

    ax = Matplotlib.Plot_Tags(mesh, useColorCycler=True, showId=False)

    surfaces = [pc for pc in _drawn(ax, PolyCollection) if pc.get_label() == "S0"]
    tab10 = [colors.to_rgba(c) for c in plt.get_cmap("tab10").colors]
    assert tuple(surfaces[0].get_facecolor()[0]) in tab10
    assert len(ax.texts) == 0


def test_Plot_Tags_returns_none_withoutelement_tags(bar_simu):
    groupElem = bar_simu.mesh.groupElem
    untagged = GroupElemFactory.Create(
        groupElem.elemType, groupElem.connect, groupElem.coord
    )

    assert Matplotlib.Plot_Tags(Mesh({untagged.elemType: untagged})) is None


def test_Plot_Tags_folder_saves_geom_pdf(bar_simu, tmp_path):
    Matplotlib.Plot_Tags(bar_simu.mesh, folder=str(tmp_path))

    assert (tmp_path / "geom.pdf").is_file()


# ----------------------------------------------
# Plot_Energy
# ----------------------------------------------
def _energies(simu) -> list[float]:
    values = []
    for i in range(simu.Niter):
        simu.Set_Iter(i)
        values.extend(simu.Results_dict_Energy().values())
    simu.Set_Iter(-1)
    return values


def test_Plot_Energy_draws_the_energy_of_each_saved_iteration(bar_simu):
    energies = _energies(bar_simu)

    Matplotlib.Plot_Energy(bar_simu, plotSolMax=False)

    (ax,) = plt.gcf().axes
    assert np.allclose(ax.lines[0].get_ydata(), energies)
    assert ax.get_xlabel() == "iter"


def test_Plot_Energy_N_caps_the_iterations_drawn(bar_simu):
    Matplotlib.Plot_Energy(bar_simu, plotSolMax=False, N=2)

    (ax,) = plt.gcf().axes
    assert list(ax.lines[0].get_xdata()) == [0, 2]


def test_Plot_Energy_adds_the_max_solution_and_the_load(bar_simu, tmp_path):
    load = np.array(LOADS)
    displacement = np.array([0.1, 0.2, 0.3])

    Matplotlib.Plot_Energy(bar_simu, load, displacement, folder=str(tmp_path))

    axEnergy, axSolMax, axLoad = plt.gcf().axes
    assert np.allclose(axEnergy.lines[0].get_xdata(), displacement)
    assert np.allclose(axLoad.lines[0].get_ydata(), load * 1e-3)
    assert (tmp_path / "Energy.pdf").is_file()


def test_Plot_Energy_draws_nothing_for_a_simulation_without_energy(bar_simu):
    simu = Simulations.Thermal(bar_simu.mesh, Models.Thermal(1.0))

    Matplotlib.Plot_Energy(simu)

    assert plt.get_fignums() == []


# ----------------------------------------------
# Plot_Iter_Summary
# ----------------------------------------------
def test_Plot_Iter_Summary_draws_one_graph_per_summary_value(phasefield_simu, tmp_path):
    iterations, list_label_values = phasefield_simu.Results_Iter_Summary()

    Matplotlib.Plot_Iter_Summary(phasefield_simu, folder=str(tmp_path))

    axs = plt.gcf().axes
    assert [ax.get_ylabel() for ax in axs] == [lv[0] for lv in list_label_values]
    for ax, (_, values) in zip(axs, list_label_values):
        assert np.allclose(ax.lines[0].get_ydata(), values)
    assert (tmp_path / "resumeConvergence.pdf").is_file()


def test_Plot_Iter_Summary_keeps_the_iterations_between_iterMin_and_iterMax(
    phasefield_simu,
):
    Matplotlib.Plot_Iter_Summary(phasefield_simu, iterMin=1, iterMax=1)

    assert list(plt.gcf().axes[0].lines[0].get_xdata()) == [1]


def test_Plot_Iter_Summary_draws_nothing_without_a_summary(bar_simu):
    Matplotlib.Plot_Iter_Summary(bar_simu)

    assert plt.get_fignums() == []


# ----------------------------------------------
# Plot_Tic_History
# ----------------------------------------------
@pytest.fixture
def tic_history():
    Tic.Clear()
    tic = Tic()
    tic.Tac("Assembly", "K")
    tic.Tac("Assembly", "F")
    tic.Tac("Solve", "x")
    yield
    Tic.Clear()


def test_Plot_Tic_History_draws_one_bar_per_category(tic_history, tmp_path):
    Matplotlib.Plot_Tic_History(str(tmp_path))

    (ax,) = plt.gcf().axes
    assert sorted(c.get_label() for c in ax.containers) == ["Assembly", "Solve"]
    assert (tmp_path / "TicTac_Summary.pdf").is_file()


def test_Plot_Tic_History_details_adds_a_figure_per_category_with_subcategories(
    tic_history, tmp_path
):
    Matplotlib.Plot_Tic_History(str(tmp_path), details=True)

    assert len(plt.get_fignums()) == 2
    assert len(list(tmp_path.glob("TicTac*_Assembly.pdf"))) == 1


def test_Plot_Tic_History_draws_nothing_without_history():
    Tic.Clear()

    Matplotlib.Plot_Tic_History()

    assert plt.get_fignums() == []


# ----------------------------------------------
# _Plot_Bar
# ----------------------------------------------
def test_Plot_Bar_writes_each_time_in_or_beside_its_bar():
    ax = plt.subplots()[1]

    Matplotlib._Plot_Bar(ax, ["long", "short"], [1.0, 0.1], [2, 1], "times")

    assert [p.get_width() for p in ax.patches] == [1.0, 0.1]
    long, short = ax.texts
    assert "(2 x " in long.get_text() and long.get_color() == "white"
    assert "(2 x " not in short.get_text() and short.get_color() == "black"
    assert ax.get_title() == "times"


# ----------------------------------------------
# Movie_simu / Movie_func
# ----------------------------------------------
@needs_ffmpeg
@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_Movie_simu_writes_one_frame_per_iteration_up_to_N(bar_simu, tmp_path):
    Matplotlib.Movie_simu(bar_simu, "ux", str(tmp_path), "ux.gif", N=2, plotMesh=True)

    with Image.open(tmp_path / "ux.gif") as gif:
        assert gif.n_frames == 2
        assert gif.info["duration"] == 100  # 10 fps


def test_Movie_simu_needs_a_simulation(bar_simu, tmp_path):
    Matplotlib.Movie_simu(bar_simu.mesh, "ux", str(tmp_path))

    assert list(tmp_path.iterdir()) == []


@needs_ffmpeg
def test_Movie_func_calls_func_once_per_frame(tmp_path):
    calls = []

    def func(fig, i):
        calls.append(i)
        fig.gca().plot([0, i])

    Matplotlib.Movie_func(func, 3, str(tmp_path), "f.gif", fps=5, show=False)

    assert calls == [0, 1, 2]
    with Image.open(tmp_path / "f.gif") as gif:
        assert gif.n_frames == 3
        assert gif.info["duration"] == 200


# ----------------------------------------------
# Save_fig
# ----------------------------------------------
def test_Save_fig_strips_the_characters_a_filename_cannot_hold(tmp_path):
    plt.plot([0, 1])

    Matplotlib.Save_fig(str(tmp_path), ' a:b?"c. ', extension="png")

    assert [p.name for p in tmp_path.iterdir()] == ["abc.png"]


def test_Save_fig_without_folder_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plt.plot([0, 1])

    Matplotlib.Save_fig("", "figure")

    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------
# _Get_list_surfaces
# ----------------------------------------------
def test_Get_list_surfaces_pads_every_group_to_the_same_length(make_mesh_3D):
    mesh = make_mesh_3D(elemType=ElemType.PRISM6)

    list_surfaces = Matplotlib._Get_list_surfaces(mesh, 2)

    tri, quad = list_surfaces
    assert len(tri) == len(quad) == 5
    assert tri[-1] == tri[0]


# ----------------------------------------------
# _Remove_colorbar
# ----------------------------------------------
def test_Remove_colorbar_removes_the_colorbar_of_the_axes(bar_simu):
    ax = Matplotlib.Plot(bar_simu, "Svm", nodeValues=False)
    assert len(ax.figure.axes) == 2

    Matplotlib._Remove_colorbar(ax)

    assert len(ax.figure.axes) == 1


# ----------------------------------------------
# Init_Axes
# ----------------------------------------------
def test_Init_Axes_opens_axes_of_the_dimension():
    names = [Matplotlib.Init_Axes(dim).name for dim in (1, 2, 3)]

    assert names == ["rectilinear", "rectilinear", "3d"]


def test_Init_Axes_orients_3D_axes():
    ax = Matplotlib.Init_Axes(3, elev=30, azim=45)

    assert (ax.elev, ax.azim) == (30, 45)


def test_Init_Axes_refuses_a_fourth_dimension():
    with pytest.raises(ValueError):
        Matplotlib.Init_Axes(4)


# ----------------------------------------------
# _Axis_equal_3D
# ----------------------------------------------
def test_Axis_equal_3D_centers_equal_ranges_on_the_points():
    ax = Matplotlib.Init_Axes(3)
    coord = np.array([[0, 0, 0], [2, 1, 4]], dtype=float)

    Matplotlib._Axis_equal_3D(ax, coord)

    half = 0.55 * 4
    assert np.allclose(ax.get_xlim(), (1 - half, 1 + half))
    assert np.allclose(ax.get_ylim(), (0.5 - half, 0.5 + half))
    assert np.allclose(ax.get_zlim(), (2 - half, 2 + half))


# ----------------------------------------------
# _Fit_view
# ----------------------------------------------
def test_Fit_view_keeps_every_3D_point_drawn_in_view():
    ax = Matplotlib.Init_Axes(3)

    Matplotlib._Fit_view(ax, np.array([[0.0, 0, 0], [1, 1, 1]]))
    Matplotlib._Fit_view(ax, np.array([[9.0, 9, 9], [10, 10, 10]]))

    xmin, xmax = ax.get_xlim()
    assert xmin <= 0 and xmax >= 10


def test_Fit_view_bounds_fix_the_view():
    ax = Matplotlib.Init_Axes(2)

    Matplotlib._Fit_view(ax, np.zeros((1, 3)), bounds=(-1, 1, -2, 2))

    assert ax.get_xlim() == (-1, 1) and ax.get_ylim() == (-2, 2)


def test_Fit_view_frames_2D_axes_on_what_they_hold():
    ax = Matplotlib.Init_Axes(2)
    ax.plot([0, 4], [0, 1])

    Matplotlib._Fit_view(ax, np.zeros((1, 3)))

    xmin, xmax = ax.get_xlim()
    assert xmin <= 0 and xmax >= 4


# ----------------------------------------------
# _Annotate
# ----------------------------------------------
def test_Annotate_adds_the_title_grid_and_legend():
    ax = Matplotlib.Init_Axes(2)
    ax.plot([0, 1], label="line")

    Matplotlib._Annotate(ax, np.zeros((1, 3)), "title", True, None, True)

    assert ax.get_title() == "title"
    assert all(line.get_visible() for line in ax.get_xgridlines())
    assert ax.get_legend() is not None
