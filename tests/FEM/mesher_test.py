# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from types import SimpleNamespace

import gmsh
import numpy as np
import pytest

from EasyFEA.FEM import ElemType, Mesh, Mesher
from EasyFEA.Geoms import Circle, CircleArc, Contour, Domain, Line, Point, Points

from _numerics import rel_error


def _reset_gmsh():
    """A case left mid-model must not leak its gmsh options into the next one."""
    if gmsh.isInitialized():
        gmsh.finalize()


@pytest.fixture(autouse=True)
def _finalize_gmsh():
    yield
    _reset_gmsh()


def generate(mesher: Mesher, dim: int, elemType: ElemType, **kwargs) -> Mesh:
    mesher._Set_PhysicalGroups()
    mesher._Mesh_Generate(dim, elemType, **kwargs)
    return mesher._Mesh_Get_Mesh()


def mass(dim: int, tags) -> float:
    gmsh.model.occ.synchronize()
    return sum(gmsh.model.occ.getMass(dim, tag) for tag in tags)


def nodes_at(mesh: Mesh, xyz) -> np.ndarray:
    return np.flatnonzero(np.linalg.norm(mesh.coord - np.asarray(xyz), axis=1) < 1e-9)


def square(size: float = 1.0, meshSize: float = 0.0) -> Contour:
    return Points([(0, 0), (size, 0), (size, size), (0, size)], meshSize).Get_Contour()


# --------------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------------


def test_Mesher_prints_its_section_only_when_verbose(capsys):
    Mesher()
    assert capsys.readouterr().out == ""

    Mesher(verbosity=True)
    assert "Init GMSH interface" in capsys.readouterr().out


def test_Mesher_silences_gmsh_by_default():
    Mesher()
    assert gmsh.option.getNumber("General.Verbosity") == 0


def test_Mesher__Init_gmsh_picks_the_factory(subtests):
    for factory in ["occ", "geo"]:
        with subtests.test(factory=factory):
            _reset_gmsh()
            mesher = Mesher()
            mesher._Init_gmsh(factory)
            assert mesher._factory is getattr(gmsh.model, factory)
            assert mesher._geoms == [] and not mesher._isMeshed


def test_Mesher__Init_gmsh_rejects_an_unknown_factory():
    with pytest.raises(Exception, match="factory"):
        Mesher()._Init_gmsh("brep")


def test_Mesher__Synchronize_exposes_the_entities_to_the_model(subtests):
    for factory in ["occ", "geo"]:
        with subtests.test(factory=factory):
            _reset_gmsh()
            mesher = Mesher()
            mesher._Init_gmsh(factory)
            mesher._factory.addPoint(0, 0, 0)
            assert gmsh.model.getEntities(0) == []

            mesher._Synchronize()
            mesher._Synchronize()

            assert len(gmsh.model.getEntities(0)) == 1


def test_Mesher_Get_Entities_pairs_each_tag_with_its_dimension():
    entities = Mesher.Get_Entities([1], [2, 3], [4], [5])
    assert entities == [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]


# --------------------------------------------------------------------------------------
# curves and loops
# --------------------------------------------------------------------------------------


LOOPS = {
    "Domain": (lambda: Domain((0, 0), (2, 1)), 2.0),
    "Circle": (lambda: Circle((0, 0), 2), np.pi),
    "Points": (lambda: Points([(0, 0), (1, 0), (0, 1)]), 0.5),
    "Contour": (lambda: Points([(0, 0), (1, 0), (0, 1)]).Get_Contour(), 0.5),
}


def test_Mesher__Loop_From_Geom_bounds_the_geom_area(subtests):
    for name, (make, area) in LOOPS.items():
        with subtests.test(geom=name):
            _reset_gmsh()
            mesher = Mesher()

            loop, lines, points = mesher._Loop_From_Geom(make())

            assert len(lines) == len(points)
            surface = mesher._Surface_From_Loops([loop])
            assert rel_error(mass(2, [surface]), area) < 1e-9


def test_Mesher__Loop_From_Geom_rejects_a_curve():
    with pytest.raises(TypeError, match="loop"):
        Mesher()._Loop_From_Geom(Line((0, 0), (1, 0)))


EDGES = {
    "Line": (lambda: Line((0, 0), (3, 4)), 1, 5.0),
    "minor CircleArc": (lambda: CircleArc((1, 0), (0, 1), (0, 0)), 1, np.pi / 2),
    "major CircleArc": (
        lambda: CircleArc((1, 0), (0, 1), (0, 0), coef=-1),
        1,
        3 * np.pi / 2,
    ),
    "Points": (lambda: Points([(0, 0), (1, 0), (2, 0)]), 1, 2.0),
}


def test_Mesher__Create_Lines_spans_the_edge(subtests):
    for name, (make, nLines, length) in EDGES.items():
        with subtests.test(edge=name):
            _reset_gmsh()
            edge = make()
            mesher = Mesher()

            lines = mesher._Create_Lines(edge, edge.pt1, edge.pt2)

            assert len(lines) == nLines
            assert rel_error(mass(1, lines), length) < 1e-9


def test_Mesher__Create_Lines_geo_arcs_stay_on_the_circle(subtests):
    for name, coef, angle in [("minor", 1, np.pi / 2), ("major", -1, 3 * np.pi / 2)]:
        with subtests.test(arc=name):
            _reset_gmsh()
            arc = CircleArc((1, 0), (0, 1), (0, 0), meshSize=0.1, coef=coef)
            mesher = Mesher()
            mesher._Init_gmsh("geo")
            p1 = mesher._factory.addPoint(1, 0, 0, 0.1)
            p2 = mesher._factory.addPoint(0, 1, 0, 0.1)

            mesher._Create_Lines(arc, p1, p2)
            mesh = generate(mesher, 1, ElemType.SEG2)

            # geo keeps the arc center, as a point entity of its own
            nodes = mesh.Get_list_groupElem(1)[0].nodes
            assert rel_error(np.linalg.norm(mesh.coord[nodes], axis=1), 1.0) < 1e-9
            assert mesh.Ne > 10
            # the polygon inscribed in the arc, so a few 1e-4 short of r·angle
            assert rel_error(mesh.length, angle) < 1e-3


def test_Mesher__Create_Lines_rejects_a_surface():
    with pytest.raises(TypeError, match="contour edge"):
        Mesher()._Create_Lines(Domain((0, 0), (1, 1)), 1, 2)


def test_Mesher__Create_Contour_reports_its_open_edges_and_points():
    contour = Contour(
        [
            Line(Point(0, 0, isOpen=True), (1, 0), isOpen=True),
            Line((1, 0), (0, 1)),
            Line((0, 1), (0, 0)),
        ]
    )
    mesher = Mesher()

    loop, lines, points, openLines, openPoints = mesher._Create_Contour(contour)

    assert len(lines) == len(points) == 3
    assert openLines == lines[:1] and openPoints == points[:1]
    assert rel_error(mass(2, [mesher._Surface_From_Loops([loop])]), 0.5) < 1e-9


def test_Mesher__Create_Circle_drops_its_center():
    mesher = Mesher()

    _, lines, points = mesher._Create_Circle(Circle((1, 1), 2))

    assert len(lines) == len(points) == 4
    assert rel_error(mass(1, lines), 2 * np.pi) < 1e-9
    assert len(gmsh.model.getEntities(0)) == 4


def test_Mesher__Create_Domain_has_the_box_perimeter():
    mesher = Mesher()

    _, lines, points = mesher._Create_Domain(Domain((1, 1), (3, 2)))

    assert len(lines) == len(points) == 4
    assert rel_error(mass(1, lines), 6.0) < 1e-9


def test_Mesher__Surface_From_Loops_cuts_the_inner_loops():
    mesher = Mesher()
    outer = mesher._Create_Domain(Domain((0, 0), (2, 2)))[0]
    hole = mesher._Create_Circle(Circle((1, 1), 1))[0]

    surface = mesher._Surface_From_Loops([outer, hole])

    assert rel_error(mass(2, [surface]), 4 - np.pi / 4) < 1e-9


def test_Mesher__Surfaces_holes_or_fills_the_inclusions(subtests):
    for isFilled in [False, True]:
        with subtests.test(isFilled=isFilled):
            _reset_gmsh()
            mesher = Mesher()
            inclusion = Circle((1, 1), 1, isFilled=isFilled)
            domain = Domain((0, 0), (2, 2))

            surfaces, lines, points = mesher._Surfaces(domain, [inclusion])

            assert len(surfaces) == (2 if isFilled else 1)
            assert len(lines) == len(points) == 4
            area = 4.0 if isFilled else 4 - np.pi / 4
            assert rel_error(mass(2, surfaces), area) < 1e-9


def test_Mesher__Surfaces_requires_a_geom():
    with pytest.raises(AssertionError):
        Mesher()._Surfaces("contour")


def test_Mesher__Surfaces_Organize_counts_the_elements(subtests):
    for elemType, Ne in [(ElemType.TRI3, 18), (ElemType.QUAD4, 9)]:
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            mesher = Mesher()
            surfaces, _, _ = mesher._Surfaces(Domain((0, 0), (1, 1)))

            mesher._Surfaces_Organize(surfaces, elemType, True, [3, 3, 3, 3])

            assert generate(mesher, 2, elemType).Ne == Ne


def test_Mesher__Additional_Surfaces_cuts_or_splits_the_entities(subtests):
    for isFilled in [False, True]:
        with subtests.test(isFilled=isFilled):
            _reset_gmsh()
            mesher = Mesher()
            mesher._Surfaces(Domain((0, 0), (2, 2)))
            inner = Domain((0.5, 0.5), (1, 1), isFilled=isFilled)

            mesher._Additional_Surfaces(2, [inner])

            surfaces = [tag for _, tag in mesher._factory.getEntities(2)]
            assert len(surfaces) == (2 if isFilled else 1)
            assert rel_error(mass(2, surfaces), 4.0 if isFilled else 3.75) < 1e-9


def test_Mesher__Additional_Surfaces_needs_a_surface_dimension():
    with pytest.raises(AssertionError):
        Mesher()._Additional_Surfaces(1, [])


def test_Mesher__Organise_Surfaces_meshes_a_structured_grid(subtests):
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            mesher = Mesher()
            mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))

            mesher._Organise_Surfaces(elemType, True, 1 / 2)

            coef = 1 if elemType.topology == "QUAD" else 2
            assert generate(mesher, 2, elemType).Ne == 4 * coef


def test_Mesher__Organise_Surfaces_lets_an_edge_size_itself():
    contour = Contour(
        [
            Line((0, 0), (1, 0), 1),
            Line((1, 0), (1, 1), 1 / 4),
            Line((1, 1), (0, 1), 1 / 4),
            Line((0, 1), (0, 0), 1 / 4),
        ]
    )
    mesher = Mesher()
    mesher._Surfaces(contour)

    mesher._Organise_Surfaces(ElemType.QUAD4, True, 1 / 2)

    # opposite sides share their coarsest count
    assert generate(mesher, 2, ElemType.QUAD4).Ne == 1 * 4


def test_Mesher__Organise_Surfaces_recombines_quads_unorganised():
    mesher = Mesher()
    mesher._Surfaces(Circle((0, 0), 2, 0.2))

    mesher._Organise_Surfaces(ElemType.QUAD4, False, 0.2)

    assert generate(mesher, 2, ElemType.QUAD4).elemType == ElemType.QUAD4


def test_Mesher__Geoms_From_Curves_maps_each_curve_to_its_geom():
    edge = Line((0, 0), (1, 0), 1 / 4)
    contour = Contour([edge, Line((1, 0), (0, 1)), Line((0, 1), (0, 0))])
    mesher = Mesher()
    assert mesher._Geoms_From_Curves() == {}

    _, lines, _ = mesher._Surfaces(contour)

    dict_geom = mesher._Geoms_From_Curves()
    assert dict_geom[lines[0]] is edge
    assert all(dict_geom[line] is contour for line in lines[1:])


def test_Mesher__Cyclic_Boundary_Order_walks_around_the_loop():
    mesher = Mesher()
    _, (l1, l2, l3, l4), _ = mesher._Create_Domain(Domain((0, 0), (1, 1)))
    mesher._Synchronize()

    assert mesher._Cyclic_Boundary_Order([l1, l3, l4, l2]) == [l1, l2, l3, l4]
    assert mesher._Cyclic_Boundary_Order([l3, l1]) == [l3, l1]


def test_Mesher__Cyclic_Boundary_Order_is_None_on_disjoint_loops():
    mesher = Mesher()
    lines1 = mesher._Create_Domain(Domain((0, 0), (1, 1)))[1]
    lines2 = mesher._Create_Domain(Domain((2, 0), (3, 1)))[1]
    mesher._Synchronize()

    assert mesher._Cyclic_Boundary_Order(lines1[:2] + lines2[:2]) is None


def test_Mesher__Additional_Lines_embeds_the_line():
    line = Line((0.5, 0), (0.5, 1))
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 4))

    mesher._Additional_Lines(2, [line])
    mesh = generate(mesher, 2, ElemType.TRI3)

    assert rel_error(mesh.area, 1.0) < 1e-9
    assert mesh.Nodes_Line(line).size >= 5


def test_Mesher__Additional_Lines_needs_a_curve_dimension():
    with pytest.raises(AssertionError):
        Mesher()._Additional_Lines(0, [])


def test_Mesher__Additional_Points_puts_a_node_on_each_point():
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))

    mesher._Additional_Points(2, [Point(0.3, 0.4)], 1 / 2)
    mesh = generate(mesher, 2, ElemType.TRI3)

    assert nodes_at(mesh, (0.3, 0.4, 0)).size == 1


def test_Mesher__Additional_Points_rejects_a_non_point():
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1)))
    with pytest.raises(Exception, match="list of point"):
        mesher._Additional_Points(2, [(0.3, 0.4)])


def test_Mesher__Spline_From_Points_keeps_only_its_ends():
    mesher = Mesher()

    spline, ends = mesher._Spline_From_Points(Points([(0, 0), (1, 0), (2, 0)]))

    assert rel_error(mass(1, [spline]), 2.0) < 1e-9
    assert len(ends) == 2 and len(gmsh.model.getEntities(0)) == 2


def test_Mesher__Set_PhysicalGroups_names_each_entity(subtests):
    lines = {"L0", "L1", "L2", "L3", "S0"}
    for setPoints in [True, False]:
        with subtests.test(setPoints=setPoints):
            _reset_gmsh()
            mesher = Mesher()
            mesher._Set_PhysicalGroups()
            assert gmsh.model.getPhysicalGroups() == []
            mesher._Surfaces(Domain((0, 0), (1, 1)))

            mesher._Set_PhysicalGroups(setPoints=setPoints)

            groups = gmsh.model.getPhysicalGroups()
            names = {gmsh.model.getPhysicalName(*group) for group in groups}
            assert names == (lines | {"P0", "P1", "P2", "P3"} if setPoints else lines)


# --------------------------------------------------------------------------------------
# volumes
# --------------------------------------------------------------------------------------


def test_Mesher__Extrude_sweeps_the_surface():
    mesher = Mesher()
    surfaces, _, _ = mesher._Surfaces(Domain((0, 0), (2, 1)))

    entities = mesher._Extrude(surfaces, (0, 0, 3))

    volumes = [tag for dim, tag in entities if dim == 3]
    assert rel_error(mass(3, volumes), 6.0) < 1e-9


def test_Mesher__Extrude_stacks_the_layers():
    mesher = Mesher()
    surfaces, _, _ = mesher._Surfaces(Domain((0, 0), (2, 1), 1))
    mesher._Organise_Surfaces(ElemType.HEXA8, True, 1)

    mesher._Extrude(surfaces, (0, 0, 1), ElemType.HEXA8, 3)

    assert generate(mesher, 3, ElemType.HEXA8).Ne == 2 * 3


def test_Mesher__Revolve_follows_Pappus(subtests):
    """Revolving [1, 2] × [0, 1] about the y-axis sweeps θ/2 (r2² - r1²) h."""
    for angle in [90, 360]:
        with subtests.test(angle=angle):
            _reset_gmsh()
            mesher = Mesher()
            surfaces, _, _ = mesher._Surfaces(Domain((1, 0), (2, 1)))

            entities = mesher._Revolve(surfaces, (0, 0), (0, 1), angle)

            volumes = [tag for dim, tag in entities if dim == 3]
            assert rel_error(mass(3, volumes), np.radians(angle) / 2 * 3) < 1e-9


def test_Mesher__Revolve_chords_the_hexahedra():
    """n straight-sided wedges sweep n/2 sin(θ/n) (r2² - r1²) h."""
    mesher = Mesher()
    surfaces, _, _ = mesher._Surfaces(Domain((1, 0), (2, 1), 1))
    mesher._Organise_Surfaces(ElemType.HEXA8, True, 1)

    mesher._Revolve(surfaces, (0, 0), (0, 1), 90, ElemType.HEXA8, [6])
    mesh = generate(mesher, 3, ElemType.HEXA8)

    assert mesh.Ne == 6
    assert rel_error(mesh.volume, 6 / 2 * np.sin(np.pi / 12) * 3) < 1e-9


def test_Mesher__Linking_Surfaces_needs_matching_contours():
    with pytest.raises(AssertionError, match="same number"):
        Mesher()._Linking_Surfaces([1, 2], [1, 2], [3], [3], [], ElemType.HEXA8)


FACE_CENTERS = [
    [Point(0.5, 0, 0.5)],
    [Point(1, 0.5, 0.5)],
    [Point(0.5, 1, 0.5)],
    [Point(0, 0.5, 0.5)],
]


@pytest.mark.parametrize(
    "listPoints", [[], FACE_CENTERS], ids=["plain", "through points"]
)
def test_Mesher__Linking_Surfaces_closes_a_unit_cube(listPoints):
    bottom, top = square(meshSize=1 / 2), square(meshSize=1 / 2)
    top.Translate(dz=1)
    mesher = Mesher()
    loop1, lines1, points1 = mesher._Loop_From_Geom(bottom)
    loop2, lines2, points2 = mesher._Loop_From_Geom(top)
    links = [mesher._factory.addLine(p, q) for p, q in zip(points1, points2)]
    caps = [mesher._factory.addSurfaceFilling(loop) for loop in (loop1, loop2)]

    sides = mesher._Linking_Surfaces(
        lines1, points1, lines2, points2, links, ElemType.TETRA4, 2, listPoints
    )
    mesher._factory.addVolume([mesher._factory.addSurfaceLoop(caps + sides)])
    mesh = generate(mesher, 3, ElemType.TETRA4)

    assert len(sides) == 4
    assert rel_error(mesh.volume, 1.0) < 1e-9


def test_Mesher__Link_Contours_fills_a_unit_cube(subtests):
    for elemType, Ne in [(ElemType.TETRA4, None), (ElemType.HEXA8, 8)]:
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            bottom, top = square(meshSize=1 / 2), square(meshSize=1 / 2)
            top.Translate(dz=1)
            mesher = Mesher()

            entities = mesher._Link_Contours(bottom, top, elemType, nLayers=2)
            mesh = generate(mesher, 3, elemType)

            assert sum(dim == 3 for dim, _ in entities) == 1
            assert rel_error(mesh.volume, 1.0) < 1e-9
            if Ne is not None:
                assert mesh.Ne == Ne


def test_Mesher__Link_Contours_warns_off_organising_a_triangle(capsys):
    bottom = Points([(0, 0), (1, 0), (0, 1)], 1 / 2).Get_Contour()
    top = bottom.copy()
    top.Translate(dz=1)
    mesher = Mesher()

    mesher._Link_Contours(bottom, top, ElemType.TETRA4)
    mesh = generate(mesher, 3, ElemType.TETRA4)

    assert "3 or 4 corners" in capsys.readouterr().out
    assert rel_error(mesh.volume, 0.5) < 1e-9


# --------------------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------------------


def test_Mesher_Mesh_Import_mesh_names_the_entities_and_scales(tmp_path):
    path = str(tmp_path / "square.msh")
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 2), 1 / 2))
    mesher._Mesh_Generate(2, ElemType.TRI3, path=path)
    gmsh.finalize()

    mesh = Mesher().Mesh_Import_mesh(path, meshOrder=2, coef=2)

    assert mesh.elemType == ElemType.TRI6
    assert rel_error(mesh.area, 8.0) < 1e-9
    assert {"S1", "L1", "L4", "P1"} <= set(
        mesh.Get_list_groupElem(0)[0].nodeTags
    ) | set(mesh.Get_list_groupElem(1)[0].nodeTags) | set(mesh.groupElem.nodeTags)


def test_Mesher_Mesh_Import_mesh_keeps_its_physical_groups(tmp_path):
    path = str(tmp_path / "square.msh")
    contour = Domain((0, 0), (1, 2), 1 / 2)
    contour.Mesh_2D(path=path)

    mesh = Mesher().Mesh_Import_mesh(path, setPhysicalGroupsFromEntities=False)

    assert set(mesh.groupElem.nodeTags) == {"S0"}
    assert rel_error(mesh.area, 2.0) < 1e-9


@pytest.fixture
def box_step(tmp_path) -> str:
    """A 1 × 2 × 3 box saved as STEP."""
    path = str(tmp_path / "box.stp")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.addBox(0, 0, 0, 1, 2, 3)
    gmsh.model.occ.synchronize()
    gmsh.write(path)
    gmsh.finalize()
    return path


def test_Mesher_Mesh_Import_part_meshes_the_step_volume(box_step: str):
    mesh = Mesher().Mesh_Import_part(box_step, 3, meshSize=0.5)

    assert mesh.elemType == ElemType.TETRA4
    assert rel_error(mesh.volume, 6.0) < 1e-9


def test_Mesher_Mesh_Import_part_falls_back_to_tetrahedra(box_step: str, capsys):
    mesh = Mesher().Mesh_Import_part(box_step, 3, 1.0, ElemType.HEXA8)

    assert mesh.elemType == ElemType.TETRA4
    assert "TETRA4" in capsys.readouterr().out


def test_Mesher__Cracks_SetPhysicalGroups_without_cracks_is_empty():
    assert Mesher()._Cracks_SetPhysicalGroups([], []) == (None, None, None, None)


def test_Mesher__Cracks_SetPhysicalGroups_rejects_a_surface():
    with pytest.raises(TypeError, match="crack"):
        Mesher()._Cracks_SetPhysicalGroups([Domain((0, 0), (1, 1))], [])


CRACKS_2D = {
    "Line": lambda isOpen: Line(
        Point(0, 0.5, isOpen=True), (0.5, 0.5), 0.1, isOpen=isOpen
    ),
    "Points": lambda isOpen: Points(
        [Point(0, 0.5, isOpen=True), (0.3, 0.5), (0.5, 0.6)], 0.1, isOpen=isOpen
    ),
    "CircleArc": lambda isOpen: CircleArc(
        Point(0, 0.5, isOpen=True),
        (0.5, 0.5),
        P=(0.25, 0.6),
        meshSize=0.1,
        isOpen=isOpen,
    ),
}


def test_Mesher__Cracks_SetPhysicalGroups_opens_only_open_cracks(subtests):
    for name, make_crack in CRACKS_2D.items():
        with subtests.test(crack=name):
            _reset_gmsh()
            contour = Domain((0, 0), (1, 1), 0.1)

            closed = Mesher().Mesh_2D(contour, cracks=[make_crack(False)])
            opened = Mesher().Mesh_2D(contour, cracks=[make_crack(True)])

            assert rel_error(closed.area, 1.0) < 1e-9
            assert rel_error(opened.area, 1.0) < 1e-9
            assert opened.Nn > closed.Nn


def test_Mesher__Cracks_SetPhysicalGroups_opens_a_crack_surface():
    contour = Domain((0, 0), (1, 1), 0.25)
    crack = Contour(
        [
            Line(Point(0, 0.5, 0, isOpen=True), Point(0.5, 0.5, 0)),
            Line(Point(0.5, 0.5, 0), Point(0.5, 0.5, 1)),
            Line(Point(0.5, 0.5, 1), Point(0, 0.5, 1, isOpen=True)),
            Line(
                Point(0, 0.5, 1, isOpen=True),
                Point(0, 0.5, 0, isOpen=True),
                isOpen=True,
            ),
        ],
        isOpen=True,
    )

    mesh = Mesher().Mesh_Extrude(contour, [], (0, 0, 1), [4], cracks=[crack])
    plain = Mesher().Mesh_Extrude(contour, [], (0, 0, 1), [4])

    assert rel_error(mesh.volume, 1.0) < 1e-9
    assert mesh.Nn > plain.Nn


def test_Mesher_Mesh_1D_meshes_each_curve(subtests):
    for elemType in ElemType.Get_1D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            contour = Points([(0, 0), (1, 0), (1, 1)], 0.25).Get_Contour()
            arc = CircleArc((2, 0), (3, 0), P=(2.5, 0.5), meshSize=0.1)

            mesh = Mesher().Mesh_1D([contour, arc], elemType, [Point(0.4, 0)])

            assert mesh.elemType == elemType
            assert nodes_at(mesh, (0.4, 0, 0)).size == 1
            assert rel_error(mesh.length, 2 + np.sqrt(2) + np.pi / 2) < 1e-3


def test_Mesher_Mesh_1D_rejects_a_surface():
    with pytest.raises(Exception, match="Line, CircleArc, Points or Contour"):
        Mesher().Mesh_1D([Domain((0, 0), (1, 1))])


def test_Mesher_Mesh_Beams_tags_each_beam_by_name():
    beams = [
        SimpleNamespace(line=Line((0, 0), (1, 0), 0.25), name="left"),
        SimpleNamespace(line=Line((1, 0), (1, 1), 0.25), name="right"),
    ]

    mesh = Mesher().Mesh_Beams(beams)

    for beam in beams:
        assert np.array_equal(mesh.Nodes_Tags(beam.name), mesh.Nodes_Line(beam.line))
    assert rel_error(mesh.length, 2.0) < 1e-12


def test_Mesher_Mesh_2D_keeps_the_area(subtests):
    for elemType in ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            contour = Domain((0, 0), (2, 1), 1 / 4)
            hole = Domain((0.25, 0.25), (0.5, 0.5))
            surface = Domain((1, 0.25), (1.5, 0.75))
            line = Line((1.75, 0), (1.75, 1))
            point = Point(0.75, 0.75)

            mesh = Mesher().Mesh_2D(
                contour,
                [hole],
                elemType,
                additionalSurfaces=[surface],
                additionalLines=[line],
                additionalPoints=[point],
            )

            # quads may leave a few triangles around the embedded line and point
            groupElems = mesh.Get_list_groupElem(2)
            assert groupElems[0].elemType == elemType
            area = sum(g.area for g in groupElems)
            assert rel_error(area, 2 - 1 / 16 - 1 / 4) < 1e-9
            assert nodes_at(mesh, (0.75, 0.75, 0)).size == 1


def test_Mesher_Mesh_2D_rejects_a_volume_element():
    with pytest.raises(Exception, match="Must be in"):
        Mesher().Mesh_2D(Domain((0, 0), (1, 1)), elemType=ElemType.TETRA4)


def test_Mesher_Mesh_2D_carries_the_geoms_of_a_failed_mesh():
    contour = Points([(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)])

    with pytest.raises(Exception, match="gmsh could not mesh") as error:
        Mesher().Mesh_2D(contour)

    assert error.value.geoms == [contour]


def test_Mesher_Mesh_Extrude_keeps_the_volume(subtests):
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            contour = Domain((0, 0), (2, 1), 1 / 2)
            inclusion = Domain((0.25, 0.25), (0.75, 0.75), isFilled=True)
            refine = Circle((1.5, 0.5), 0.5, 1 / 4)

            mesh = Mesher().Mesh_Extrude(
                contour,
                [inclusion],
                (0, 0, 2),
                [2],
                elemType,
                refineGeoms=[refine],
            )

            assert mesh.elemType == elemType
            assert rel_error(mesh.volume, 4.0) < 1e-9


def test_Mesher_Mesh_Revolve_chords_the_ring(subtests):
    """Polygonal Pappus, n/2 sin(θ/n) (r2² - r1²) h, exact for straight-sided elements."""
    exact = 6 / 2 * np.sin(np.pi / 12) * 3
    for elemType in ElemType.Get_3D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            contour = Domain((1, 0), (2, 1), 1 / 2)

            mesh = Mesher().Mesh_Revolve(
                contour,
                [],
                (0, 0),
                (0, 1),
                90,
                [6],
                elemType,
                isOrganised=True,
            )

            assert mesh.elemType == elemType
            # curved higher-order elements land between the chords and the arc
            assert exact - 1e-9 <= mesh.volume <= 3 * np.pi / 4 + 1e-9


def test_Mesher_Create_posFile_writes_a_scalar_view(tmp_path):
    coord = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

    path = Mesher().Create_posFile(coord, np.ones(3), str(tmp_path), "size")

    assert path.endswith("size.pos")
    with open(path) as file:
        assert "scalar points" in file.read()


def test_Mesher_Create_posFile_needs_3D_coordinates(tmp_path):
    with pytest.raises(AssertionError, match="dimension"):
        Mesher().Create_posFile(np.zeros((3, 2)), np.ones(3), str(tmp_path))


def test_Mesher_Set_meshSize_divides_each_side():
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1)))

    mesher.Set_meshSize(1 / 4)
    mesh = generate(mesher, 2, ElemType.TRI3)

    assert mesh.Nodes_Line(Line((0, 0), (1, 0))).size == 5


def _refined_Ne(refineGeoms) -> int:
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))
    mesher._Mesh_Refine(refineGeoms, 1 / 2)
    return generate(mesher, 2, ElemType.TRI3).Ne


REFINES = {
    "Domain": lambda folder: Domain((0, 0), (0.5, 0.5), 0.05),
    "Circle": lambda folder: Circle((0, 0), 1, 0.05),
    "pos": lambda folder: Mesher().Create_posFile(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        np.full(4, 0.05),
        folder,
    ),
}


def test_Mesher__Mesh_Refine_refines_the_background(tmp_path, subtests):
    for name, make_refine in REFINES.items():
        with subtests.test(refineGeom=name):
            _reset_gmsh()
            refineGeom = make_refine(str(tmp_path))
            assert _refined_Ne([refineGeom]) > 2 * _refined_Ne([])


def test_Mesher__Mesh_Refine_skips_what_it_cannot_read(tmp_path, capsys):
    text = tmp_path / "size.txt"
    text.write_text("")

    Ne = _refined_Ne([None, str(tmp_path / "missing.pos"), str(text)])

    out = capsys.readouterr().out
    assert "does not exist" in out and "Must provide a .pos" in out
    assert Ne == _refined_Ne([])


def test_Mesher__Mesh_Refine_rejects_an_unknown_type():
    with pytest.raises(TypeError, match="refineGeoms"):
        _refined_Ne([1.0])


def test_Mesher__Set_mesh_order_raises_the_element_order(subtests):
    for elemType in ElemType.Get_1D() + ElemType.Get_2D():
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            mesher = Mesher()
            dim = 1 if elemType in ElemType.Get_1D() else 2
            if dim == 1:
                line = Line((0, 0), (1, 0), 1 / 2)
                mesher._Create_Lines(line, line.pt1, line.pt2)
            else:
                mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))
                mesher._Organise_Surfaces(elemType, False, 1 / 2)
            mesher._Synchronize()
            gmsh.model.mesh.generate(dim)

            Mesher._Set_mesh_order(elemType)

            assert mesher._Mesh_Get_Mesh().elemType == elemType


def test_Mesher__Set_mesh_algorithm_sets_the_dimension_algorithm(subtests):
    for elemType, option, value in [
        (ElemType.TRI3, "Mesh.Algorithm", 6),
        (ElemType.TETRA4, "Mesh.Algorithm3D", 1),
    ]:
        with subtests.test(elemType=str(elemType)):
            _reset_gmsh()
            Mesher()._Set_mesh_algorithm(elemType)
            assert gmsh.option.getNumber(option) == value


def test_Mesher__Set_mesh_algorithm_rejects_a_point():
    with pytest.raises(ValueError):
        Mesher()._Set_mesh_algorithm(ElemType.POINT)


def test_Mesher__Raise_mesh_error_names_the_geoms(subtests):
    for message in ["bad loop", ""]:
        with subtests.test(message=message):
            _reset_gmsh()
            mesher = Mesher()
            domain = Domain((0, 0), (1, 1))
            mesher._Surfaces(domain)

            with pytest.raises(Exception) as error:
                mesher._Raise_mesh_error(ValueError(message))

            assert error.value.geoms == [domain]
            assert domain.name in str(error.value)
            assert (message or "ValueError") in str(error.value)


def test_Mesher__Mesh_Generate_writes_the_mesh(tmp_path):
    path = tmp_path / "out" / "square.msh"
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))

    mesher._Mesh_Generate(2, ElemType.TRI3, path=str(path))

    assert path.exists() and mesher._isMeshed


def test_Mesher__Mesh_Get_Mesh_scales_the_coordinates():
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1), 1 / 2))
    mesher._Mesh_Generate(2, ElemType.TRI3)

    mesh = mesher._Mesh_Get_Mesh(coef=3)

    assert rel_error(mesh.area, 9.0) < 1e-12
    assert not gmsh.isInitialized()


# --------------------------------------------------------------------------------------
# partitions: `_Mesh_Get_Meshes` builds every rank's piece in one process
# --------------------------------------------------------------------------------------


PARTITIONED = [ElemType.TRI3, ElemType.TETRA4]
NPROC = 3
SIZE = 10.0


def Get_partitions(elemType: ElemType, Nproc: int) -> list[Mesh]:
    """`Nproc` pieces of a SIZE-wide square or slab of `elemType`."""
    mesher = Mesher()
    dim = 2 if elemType in ElemType.Get_2D() else 3
    meshSize = 1.0 if dim == 2 else 2.0
    contour = Domain((0, 0), (SIZE, SIZE), meshSize)
    mesher._Surfaces(contour)
    mesher._Organise_Surfaces(elemType, True, meshSize)
    if dim == 3:
        surfaces = [entity[1] for entity in mesher._factory.getEntities(2)]
        mesher._Extrude(surfaces, (0, 0, 2 * meshSize), elemType, [2])

    mesher._Set_PhysicalGroups()
    mesher._Mesh_Generate(dim, elemType)
    return mesher._Mesh_Get_Meshes(Nproc)


def Get_partitioned_data(meshes: list[Mesh]) -> list[tuple]:
    """(rank, elements, ghostElements, nodes, ghostNodes) of the main group of each partition."""
    return [mesh.groupElem._Get_partitioned_data() for mesh in meshes]


@pytest.fixture(scope="module")
def partitions() -> dict[ElemType, tuple[Mesh, list[Mesh]]]:
    """Per partitioned type: the unpartitioned mesh, the reference to split against, and its NPROC pieces, as a pair."""
    partitions = {
        elemType: (Get_partitions(elemType, 1)[0], Get_partitions(elemType, NPROC))
        for elemType in PARTITIONED
    }
    _reset_gmsh()
    return partitions


def test_Mesher__Mesh_Get_Meshes_owned_elements_partition_the_mesh(
    partitions, subtests
):
    for elemType, (whole, pieces) in partitions.items():
        with subtests.test(elemType=str(elemType)):
            owned = [set(e.tolist()) for _, e, _, _, _ in Get_partitioned_data(pieces)]

            assert sum(len(elements) for elements in owned) == len(set().union(*owned))
            assert set().union(*owned) == set(range(whole.groupElem.Ne))


def test_Mesher__Mesh_Get_Meshes_owned_nodes_partition_the_mesh(partitions, subtests):
    for elemType, (whole, pieces) in partitions.items():
        with subtests.test(elemType=str(elemType)):
            owned = [set(n.tolist()) for _, _, _, n, _ in Get_partitioned_data(pieces)]

            assert sum(len(nodes) for nodes in owned) == len(set().union(*owned))
            assert set().union(*owned) == set(range(whole.Nn))


def test_Mesher__Mesh_Get_Meshes_ghost_elements_are_owned_elsewhere(
    partitions, subtests
):
    for elemType, (_, pieces) in partitions.items():
        with subtests.test(elemType=str(elemType)):
            data = Get_partitioned_data(pieces)
            owned = [set(elements.tolist()) for _, elements, _, _, _ in data]

            for rank, (_, _, ghostElements, _, _) in enumerate(data):
                others = set().union(*(owned[r] for r in range(NPROC) if r != rank))
                assert set(ghostElements.tolist()) <= others


def test_Mesher__Mesh_Get_Meshes_holds_owned_plus_ghost(partitions, subtests):
    for elemType, (_, pieces) in partitions.items():
        with subtests.test(elemType=str(elemType)):
            data = Get_partitioned_data(pieces)
            for mesh, (_, elements, ghosts, _, _) in zip(pieces, data):
                assert mesh.groupElem.Ne == elements.size + ghosts.size
                assert elements.size > 0


def test_Mesher__Mesh_Get_Meshes_is_reproducible(partitions, subtests):
    for elemType, (_, pieces) in partitions.items():
        with subtests.test(elemType=str(elemType)):
            data = Get_partitioned_data(pieces)
            again = Get_partitioned_data(Get_partitions(elemType, NPROC))

            for (_, e1, g1, n1, _), (_, e2, g2, n2, _) in zip(data, again):
                assert np.array_equal(np.sort(e1), np.sort(e2))
                assert np.array_equal(np.sort(g1), np.sort(g2))
                assert np.array_equal(np.sort(n1), np.sort(n2))


def test_Mesher__Mesh_Get_Meshes_needs_a_partition():
    mesher = Mesher()
    mesher._Surfaces(Domain((0, 0), (1, 1)))
    mesher._Mesh_Generate(2, ElemType.TRI3)

    with pytest.raises(AssertionError, match="Nproc"):
        mesher._Mesh_Get_Meshes(0)
