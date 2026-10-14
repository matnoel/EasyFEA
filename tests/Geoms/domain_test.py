# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np
import pytest

from EasyFEA.Geoms import Domain, Line

L, H = 2.0, 1.0


def test_Domain_is_never_open():
    domain = Domain((0, 0), (L, H), meshSize=0.5, isFilled=True)

    assert not domain.isOpen
    assert domain.isFilled
    assert domain.meshSize == 0.5


def test_Domain_pt1_is_the_first_corner_and_read_only():
    """pt1/pt2 are views on points: assigning them would desync the meshed shape from the plotted one."""

    domain = Domain((1, 2), (L, H))
    assert domain.pt1.Check((1, 2))

    with pytest.raises(AttributeError):
        domain.pt1 = (9, 9)  # type: ignore [misc]


def test_Domain_pt2_is_the_opposite_corner_and_read_only():
    domain = Domain((0, 0), (L, H))
    assert domain.pt2.Check((L, H))

    with pytest.raises(AttributeError):
        domain.pt2 = (9, 9)  # type: ignore [misc]


def test_Domain_Get_Contour_is_the_four_edges_in_the_pt1_plane():
    domain = Domain((0, 0, 1), (L, H, 3), meshSize=0.5, isFilled=True)
    contour = domain.Get_Contour()

    assert [type(geom) for geom in contour.geoms] == [Line] * 4
    assert np.array_equal(contour.coord, [(0, 0, 1), (L, 0, 1), (L, H, 1), (0, H, 1)])
    assert contour.length == pytest.approx(2 * (L + H))
    assert contour.isFilled
    assert contour.meshSize == 0.5
