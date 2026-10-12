# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import inspect
from typing import Callable

import pytest
import numpy as np

from EasyFEA import Mesher
from EasyFEA.Geoms import _Geom, Point, Line, Circle, CircleArc, Points, Domain, Contour

from _numerics import rel_error


def _sine() -> Points:
    x = np.linspace(0, 5, 10)
    return Points([Point(xi, yi) for xi, yi in zip(x, np.sin(x))])


def _arc_contour() -> Contour:
    bottom = Line(Point(), Point(5, 0))
    arc = CircleArc(Point(5), Point(-5), P=Point(0, 5))
    back = Line(Point(-5), Point())
    return Contour([bottom, arc, back])


def _rounded_contour() -> Contour:
    corners = Points([Point(), Point(5, 0), Point(5, 5, r=2), Point(0, 5, r=-3)])
    return corners.Get_Contour()


GEOMS: dict[str, Callable[[], _Geom]] = {
    "Line": lambda: Line(Point(), Point(5, 1)),
    "Points": _sine,
    "Domain": lambda: Domain(Point(), Point(1, 1, 1)),
    "Circle": lambda: Circle(Point(), 5, n=(1, 1, 1)),
    "CircleArc": lambda: CircleArc(Point(3, 1, 3), Point(-3, 1, 3), center=Point(0, 1)),
    "Contour": _arc_contour,
    "rounded Contour": _rounded_contour,
}

# (copying transform, its closed form on the (N, 3) point coordinates)
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "translate": (
        lambda g: g.Translate(1, 2, 3, copy=True),
        lambda c: c + (1, 2, 3),
    ),
    "rotate about z": (
        lambda g: g.Rotate(90, copy=True),
        lambda c: np.stack([-c[:, 1], c[:, 0], c[:, 2]], axis=1),
    ),
    "rotate about a shifted x axis": (
        lambda g: g.Rotate(90, (0, 1, 0), (1, 0, 0), copy=True),
        lambda c: np.stack([c[:, 0], 1 - c[:, 2], c[:, 1] - 1], axis=1),
    ),
    "symmetry through xy": (
        lambda g: g.Symmetry(n=(0, 0, 1), copy=True),
        lambda c: c * (1, 1, -1),
    ),
    "symmetry through x = 1": (
        lambda g: g.Symmetry((1, 0, 0), (1, 0, 0), copy=True),
        lambda c: c * (-1, 1, 1) + (2, 0, 0),
    ),
}


@pytest.fixture(params=GEOMS, ids=str)
def geom(request) -> _Geom:
    return GEOMS[request.param]()


@pytest.fixture(params=TRANSFORMS, ids=str)
def transform(request) -> tuple[Callable, Callable]:
    return TRANSFORMS[request.param]


def test_transform_moves_every_point_to_its_closed_form(geom: _Geom, transform):
    apply, closedForm = transform
    expected = closedForm(geom.coord)

    assert rel_error(apply(geom).coord, expected) < 1e-12


def test_copying_transform_leaves_the_original(geom: _Geom, transform):
    apply, _ = transform
    before = geom.coord.copy()
    apply(geom)

    assert np.array_equal(geom.coord, before)


def test_in_place_transform_matches_the_copy(geom: _Geom):
    moved = geom.Rotate(90, (0, 1, 0), (1, 0, 0), copy=True)
    result = geom.Rotate(90, (0, 1, 0), (1, 0, 0))

    assert result is None
    assert np.array_equal(geom.coord, moved.coord)


@pytest.mark.parametrize("name", ["Mesh_1D", "Mesh_2D", "Mesh_Extrude", "Mesh_Revolve"])
def test_geom_meshing_matches_mesher(name: str):
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
