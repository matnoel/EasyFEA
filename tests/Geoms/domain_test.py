# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA import ElemType
from EasyFEA.Geoms import Domain, Point, Points

L, H = 2.0, 1.0


def test_points_are_read_only():
    """pt1/pt2 are views on points: assigning them would desync the meshed shape from the plotted one."""

    domain = Domain(Point(), Point(L, H))

    with pytest.raises(AttributeError):
        domain.pt1 = Point(9, 9)
    with pytest.raises(AttributeError):
        domain.pt2 = Point(9, 9)


@pytest.mark.parametrize("theta", [90, 180, -90, 270])
def test_quarter_turns_keep_the_shape(theta: int):
    """A quarter turn about a coordinate axis maps an axis-aligned domain onto another one."""

    domain = Domain(Point(), Point(L, H), meshSize=0.25)
    domain.Rotate(theta)

    assert domain.Mesh_2D([], ElemType.TRI3).area == pytest.approx(L * H)


def test_an_oblique_rotation_gives_the_spanned_box():
    """Only pt1 and pt2 are rotated, so the domain becomes the axis-aligned box those two corners span. This is the documented behaviour, not the rotated rectangle."""

    domain = Domain(Point(), Point(L, H), meshSize=0.25)
    domain.Rotate(45)

    dx, dy, _ = np.abs(domain.pt2.coord - domain.pt1.coord)
    assert domain.Mesh_2D([], ElemType.TRI3).area == pytest.approx(dx * dy)
    assert dx * dy != pytest.approx(L * H)


def test_symmetry_about_a_coordinate_plane():
    domain = Domain(Point(), Point(L, H), meshSize=0.25)
    domain.Symmetry(n=(1, 0, 0))

    assert domain.Mesh_2D([], ElemType.TRI3).area == pytest.approx(L * H)


def test_translate_is_always_allowed():
    domain = Domain(Point(), Point(L, H), meshSize=0.25)
    domain.Translate(3, -2, 1)

    assert domain.Mesh_2D([], ElemType.TRI3).area == pytest.approx(L * H)


def test_points_rotates_exactly():
    """The alternative the refusal points at: Points stores every corner, so it rotates exactly."""

    corners = Points([(0, 0), (L, 0), (L, H), (0, H)], meshSize=0.25)
    corners.Rotate(45)

    assert corners.Mesh_2D([], ElemType.TRI3).area == pytest.approx(L * H)


def test_nodes_domain_selects_nodes():
    """Covers Get_Nodes_Domain, which asserted on the wrong type and so always raised."""

    mesh = Domain(Point(), Point(4, 3), meshSize=1.0).Mesh_2D(
        [], ElemType.QUAD4, isOrganised=True
    )
    nodes = mesh.Nodes_Domain(Domain(Point(), Point(1, 1)))

    assert nodes.size == 4
    assert np.all(mesh.coord[nodes][:, 0] <= 1 + 1e-12)
    assert np.all(mesh.coord[nodes][:, 1] <= 1 + 1e-12)


def test_domain_refines():
    """refineGeoms goes through Domain, which is what maps to gmsh's Box field."""

    contour = Domain(Point(), Point(4, 4), meshSize=1.0)
    refined = contour.Mesh_2D(
        [], ElemType.TRI3, refineGeoms=[Domain(Point(), Point(1, 1), 0.25)]
    )
    coarse = contour.Mesh_2D([], ElemType.TRI3)

    assert refined.Ne > coarse.Ne
    assert refined.area == pytest.approx(16)
