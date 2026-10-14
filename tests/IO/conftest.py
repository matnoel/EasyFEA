# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import ElemType, Simulations

from Simulations._simus import pulled_bar
from ._meshes import L, H


@pytest.fixture
def bar_simu(make_mesh_2D) -> Simulations.Elastic:
    """2D QUAD4 bar, one saved iteration per load."""
    mesh = make_mesh_2D(L=L, H=H, elemType=ElemType.QUAD4, isOrganised=True)
    return pulled_bar(mesh)


@pytest.fixture
def bar_simus(make_mesh_2D, make_mesh_3D) -> list[Simulations.Elastic]:
    """`bar_simu` on a QUAD4 rectangle and on a HEXA8 box."""
    rectangle = make_mesh_2D(L=L, H=H, elemType=ElemType.QUAD4, isOrganised=True)
    box = make_mesh_3D(L=L, H=H, elemType=ElemType.HEXA8, isOrganised=True)
    return [pulled_bar(rectangle), pulled_bar(box)]
