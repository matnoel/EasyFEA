# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import inspect
from typing import Callable, NamedTuple

import pytest
import numpy as np

from EasyFEA import ElemType, Mesher
from EasyFEA.Geoms import _Geom, Point, Line, Circle, CircleArc, Points, Domain, Contour

from _numerics import rel_error

from ._shapes import rounded_corners


def _sine() -> Points:
    x = np.linspace(0, 5, 10)
    return Points(list(zip(x, np.sin(x))))


def _arc_contour() -> Contour:
    bottom = Line((0, 0), (5, 0))
    arc = CircleArc((5, 0), (-5, 0), P=(0, 5))
    back = Line((-5, 0), (0, 0))
    return Contour([bottom, arc, back])


def _rounded_contour() -> Contour:
    return rounded_corners().Get_Contour()


L, H = 2.0, 1.0

OBLIQUE = np.array([0, np.cos(np.pi / 6), np.sin(np.pi / 6)])

GEOMS: dict[str, Callable[[], _Geom]] = {
    "Line": lambda: Line((0, 0), (5, 1)),
    "Points": _sine,
    "Domain": lambda: Domain((0, 0), (1, 1, 1)),
    "Circle": lambda: Circle((0, 0), 5, n=(1, 1, 1)),
    "CircleArc": lambda: CircleArc((3, 1, 3), (-3, 1, 3), center=(0, 1)),
    "Contour": _arc_contour,
    "rounded Points": rounded_corners,
    "rounded Contour": _rounded_contour,
}

# exact curves: every plotted point lies on the geometry
CURVES = ["Line", "Circle", "CircleArc", "Contour"]


class Region(NamedTuple):
    make: Callable[[], _Geom]
    onCurve: list
    inside: list
    """enclosed but not on the curves"""
    outside: list


REGIONS: dict[str, Region] = {
    "Line": Region(
        lambda: Line((0, 0), (2, 0)),
        [(0, 0), (1, 0), (2, 0)],
        [],
        [(3, 0), (1, 1), (1, 0, 1)],
    ),
    "Points": Region(
        lambda: Points([(0, 0), (2, 0), (2, 2), (0, 2)]),
        [(1, 0), (2, 1)],
        [(1, 1)],
        [(3, 1), (1, 1, 1)],
    ),
    "Domain": Region(
        lambda: Domain((0, 0), (2, 2)),
        [(1, 0), (0, 1), (2, 2)],
        [(1, 1)],
        [(3, 1), (1, -1), (1, 1, 1)],
    ),
    "Circle": Region(
        lambda: Circle((0, 0), 2),
        [(1, 0), (0, -1), (np.cos(1), np.sin(1))],
        [(0, 0), (0.5, 0)],
        [(2, 0), (0, 0, 1)],
    ),
    "CircleArc": Region(
        lambda: CircleArc((1, 0), (0, 1), center=(0, 0)),
        [(1, 0), (0, 1), (np.cos(0.5), np.sin(0.5))],
        [],
        [(0, -1), (-1, 0), (0, 0), (0.5, 0.5)],
    ),
    "Contour": Region(
        _arc_contour,
        [(2, 0), (-3, 0), (0, 5), (5 * np.cos(1), 5 * np.sin(1))],
        [(0, 2), (3, 3)],
        [(0, 6), (0, -1), (0, 2, 1)],
    ),
}


def _pad(coords: list) -> np.ndarray:
    return np.array([(*c, 0)[:3] for c in coords], dtype=float).reshape(-1, 3)


def _two_lines() -> Contour:
    there = Line((0, 0), (1, 0))
    back = Line((1, 0), (0, 0))
    return Contour([there, back])


def test_Geom_rejects_an_empty_point_list():
    with pytest.raises(AssertionError, match="non-empty"):
        Points([])


INVALID_ATTRIBUTES = [
    ("meshSize", -1.0),
    ("name", 1),
    ("isFilled", "yes"),
    ("isOpen", 1.0),
]


def test_Geom_rejects_an_invalid_attribute(subtests):
    for name, make in GEOMS.items():
        for attribute, value in INVALID_ATTRIBUTES:
            with subtests.test(geom=name, attribute=attribute):
                geom = make()
                with pytest.raises(AssertionError):
                    setattr(geom, attribute, value)


ONE_PER_CLASS: dict[str, Callable[[], _Geom]] = {
    "Domain": lambda: Domain((0, 0), (1, 1)),
    "Circle": lambda: Circle((0, 0), 1),
    "CircleArc": lambda: CircleArc((1, 0), (0, 1), center=(0, 0)),
    "Contour": _two_lines,
    "Line": lambda: Line((0, 0), (1, 0)),
    "Points": lambda: Points([(0, 0), (1, 0)]),
}


def test_Geom__Init_Ninstance_restarts_the_name_numbering(subtests):
    for name, make in ONE_PER_CLASS.items():
        with subtests.test(geom=name):
            cls = type(make())
            cls._Init_Ninstance()

            assert make().name == f"{cls.__name__}1"
            assert make().name == f"{cls.__name__}2"


def test_Geom_points_are_the_given_points(subtests):
    for cls in (Line, Domain):
        with subtests.test(geom=cls.__name__):
            p1, p2 = Point(), Point(1, 1)
            geom = cls(p1, p2)

            assert geom.points[0] is p1
            assert geom.points[1] is p2


def test_Geom_points_of_Points_are_the_given_points():
    corners = [Point(), Point(1), Point(1, 1)]

    assert all(a is b for a, b in zip(Points(corners).points, corners))


def test_Geom_coord_stacks_the_points(subtests):
    for name, make in GEOMS.items():
        with subtests.test(geom=name):
            geom = make()
            expected = np.array([point.coord for point in geom.points])

            assert geom.coord.shape == (len(geom.points), 3)
            assert np.array_equal(geom.coord, expected)


def test_Geom_Get_coord_for_plot_marks_points_of_the_geometry(subtests):
    for name, make in GEOMS.items():
        with subtests.test(geom=name):
            geom = make()
            lines, points = geom.Get_coord_for_plot()

            assert lines.ndim == 2 and lines.shape[1] == 3
            gaps = np.linalg.norm(points[:, None] - geom.coord[None], axis=2)
            assert np.all(gaps.min(axis=1) < 1e-12)


def test_Geom_Get_coord_for_plot_draws_on_the_curves(subtests):
    for name in CURVES:
        with subtests.test(geom=name):
            geom = GEOMS[name]()
            lines, _ = geom.Get_coord_for_plot()

            assert np.all(geom.Contains(lines, tol=1e-9))


def test_Geom_Contains_only_the_points_on_its_curves(subtests):
    for name, (make, onCurve, inside, outside) in REGIONS.items():
        with subtests.test(geom=name):
            geom = make()

            assert np.all(geom.Contains(_pad(onCurve)))
            assert not np.any(geom.Contains(_pad(inside + outside)))


def test_Geom_Encloses_the_curves_and_the_inside_only(subtests):
    for name, (make, onCurve, inside, outside) in REGIONS.items():
        with subtests.test(geom=name):
            geom = make()

            assert np.all(geom.Encloses(_pad(onCurve + inside)))
            assert not np.any(geom.Encloses(_pad(outside)))


def test_Geom_Encloses_through_the_thickness_of_a_Domain():
    domain = Domain((0, 0, 0), (2, 2, 2))

    assert domain.Encloses((1, 1, 1))
    assert not domain.Contains((1, 0, 1))


def test_Geom_copy_is_independent_and_renamed(subtests):
    for name, make in GEOMS.items():
        with subtests.test(geom=name):
            geom = make()
            copy = geom.copy()
            copy.Translate(1, 2, 3)

            assert copy.name == geom.name + "_copy"
            assert rel_error(copy.coord, geom.coord + (1, 2, 3)) < 1e-12


def test_Geom_Translate_moves_every_point_by_the_offset(subtests):
    _assert_transforms(
        subtests,
        {"offset": (dict(dx=1, dy=2, dz=3), lambda c: c + (1, 2, 3))},
        _Geom.Translate,
    )


ROTATIONS: dict[str, tuple[dict, Callable]] = {
    "about z": (
        dict(theta=90),
        lambda c: np.stack([-c[:, 1], c[:, 0], c[:, 2]], axis=1),
    ),
    "about a shifted x axis": (
        dict(theta=90, center=(0, 1, 0), direction=(1, 0, 0)),
        lambda c: np.stack([c[:, 0], 1 - c[:, 2], c[:, 1] - 1], axis=1),
    ),
}


def test_Geom_Rotate_moves_every_point_to_its_closed_form(subtests):
    _assert_transforms(subtests, ROTATIONS, _Geom.Rotate)


def _shoelace(coord: np.ndarray) -> float:
    x, y = coord[:, 0], coord[:, 1]
    return abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) / 2


def test_Geom_Rotate_by_quarter_turns_keeps_a_Domain_area(subtests):
    """A quarter turn about a coordinate axis maps an axis-aligned domain onto another one."""

    for theta in (90, 180, -90, 270):
        with subtests.test(theta=theta):
            domain = Domain((0, 0), (L, H))
            domain.Rotate(theta)

            assert _shoelace(domain.Get_Contour().coord) == pytest.approx(L * H)


def test_Geom_Rotate_obliquely_turns_a_Domain_into_the_box_its_corners_span():
    """Only pt1 and pt2 are rotated: the documented Domain behaviour, not the rotated rectangle."""

    domain = Domain((0, 0), (L, H))
    domain.Rotate(45)

    S = np.sqrt(2) / 2
    assert domain.pt2.Check((S * (L - H), S * (L + H)))
    assert _shoelace(domain.Get_Contour().coord) == pytest.approx((L**2 - H**2) / 2)


def test_Geom_Rotate_keeps_the_area_of_Points():
    """Unlike Domain, Points stores every corner, so it rotates exactly."""

    corners = Points([(0, 0), (L, 0), (L, H), (0, H)])
    corners.Rotate(45)

    assert _shoelace(corners.coord) == pytest.approx(L * H)


SYMMETRIES: dict[str, tuple[dict, Callable]] = {
    "default": (dict(), lambda c: c * (-1, 1, 1)),
    "oblique plane": (
        dict(n=OBLIQUE),
        lambda c: c - 2 * np.outer(c @ OBLIQUE, OBLIQUE),
    ),
    "xy": (dict(n=(0, 0, 1)), lambda c: c * (1, 1, -1)),
    "x = 1": (dict(point=(1, 0, 0), n=(1, 0, 0)), lambda c: c * (-1, 1, 1) + (2, 0, 0)),
}


def test_Geom_Symmetry_reflects_every_point(subtests):
    _assert_transforms(subtests, SYMMETRIES, _Geom.Symmetry)


def _assert_transforms(
    subtests, table: dict[str, tuple[dict, Callable]], method: Callable
):
    """Every geom × table entry, as a copy and in place."""

    for name, make in GEOMS.items():
        for case, (kwargs, closedForm) in table.items():
            for copy in (True, False):
                with subtests.test(geom=name, case=case, copy=copy):
                    geom = make()
                    before = geom.coord.copy()
                    moved = method(geom, **kwargs, copy=copy)
                    expected = closedForm(before)

                    if copy:
                        assert moved is not None
                        assert np.array_equal(geom.coord, before)
                        assert rel_error(moved.coord, expected) < 1e-12
                    else:
                        assert moved is None
                        assert rel_error(geom.coord, expected) < 1e-12


def _assert_matches_mesher(name: str) -> None:
    """The geom-side meshing methods restate the Mesher signature so that editors can see the arguments; this pins the two copies together."""

    # drop self, and on the Mesher side the leading geometry argument that the geom provides
    geomParams = list(inspect.signature(getattr(_Geom, name)).parameters.values())[1:]
    mesherParams = list(inspect.signature(getattr(Mesher, name)).parameters.values())[
        2:
    ]

    assert [(p.name, p.kind) for p in geomParams] == [
        (p.name, p.kind) for p in mesherParams
    ], f"_Geom.{name} and Mesher.{name} no longer take the same arguments."

    doc = getattr(_Geom, name).__doc__
    for param in geomParams:
        assert (
            f"{param.name} :" in doc
        ), f"_Geom.{name}: '{param.name}' is undocumented."


def test_Geom_Mesh_1D_takes_the_Mesher_arguments():
    _assert_matches_mesher("Mesh_1D")


def test_Geom_Mesh_1D_meshes_the_length():
    line = Line((0, 0), (2, 0), meshSize=0.5)
    mesh = line.Mesh_1D(ElemType.SEG3)

    assert mesh.length == pytest.approx(2)
    assert mesh.Ne == 4


def test_Geom_Mesh_2D_takes_the_Mesher_arguments():
    _assert_matches_mesher("Mesh_2D")


def test_Geom_Mesh_2D_meshes_the_area():
    domain = Domain((0, 0), (2, 1), meshSize=0.5)
    mesh = domain.Mesh_2D([], ElemType.QUAD4, isOrganised=True)

    assert mesh.area == pytest.approx(2)
    assert mesh.Ne == 8


def test_Geom_Mesh_2D_refines_inside_a_Domain():
    """refineGeoms goes through Domain, which is what maps to gmsh's Box field."""

    contour = Domain((0, 0), (4, 4), meshSize=1.0)
    refineZone = Domain((0, 0), (1, 1), 0.25)
    refined = contour.Mesh_2D([], ElemType.TRI3, refineGeoms=[refineZone])
    coarse = contour.Mesh_2D([], ElemType.TRI3)

    assert refined.Ne > coarse.Ne
    assert refined.area == pytest.approx(16)


def test_Geom_Mesh_Extrude_takes_the_Mesher_arguments():
    _assert_matches_mesher("Mesh_Extrude")


def test_Geom_Mesh_Extrude_meshes_the_volume():
    domain = Domain((0, 0), (2, 1), meshSize=0.5)
    mesh = domain.Mesh_Extrude(
        [],
        (0, 0, 3),
        [3],
        ElemType.HEXA8,
        isOrganised=True,
    )

    assert mesh.volume == pytest.approx(6)
    assert mesh.Ne == 24


def test_Geom_Mesh_Revolve_takes_the_Mesher_arguments():
    _assert_matches_mesher("Mesh_Revolve")


def test_Geom_Mesh_Revolve_meshes_the_prism_over_a_regular_polygon():
    """Linear elements over n layers: the revolved rectangle becomes a prism over a regular n-gon."""

    R, H, n = 1.0, 2.0, 8
    rectangle = Points([(0, 0), (R, 0), (R, H), (0, H)], meshSize=0.5)
    mesh = rectangle.Mesh_Revolve(
        [],
        (0, 0),
        (0, 1),
        360,
        [n],
        ElemType.TETRA4,
    )

    assert mesh.volume == pytest.approx(H * n / 2 * R**2 * np.sin(2 * np.pi / n))
