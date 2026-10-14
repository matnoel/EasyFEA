# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from EasyFEA import ElemType, Mesher, Models, Simulations
from EasyFEA.Geoms import Domain, Line

from Simulations._simus import pulled_bar

from ._simus import L, H


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def bar_simu(make_mesh_2D) -> Simulations.Elastic:
    """2D QUAD4 bar (6 × 3 elements), one saved iteration per load."""
    mesh = make_mesh_2D(L=L, H=H, elemType=ElemType.QUAD4, isOrganised=True)
    return pulled_bar(mesh, withBCs=True)


@pytest.fixture
def bar_simu_3D(make_mesh_3D) -> Simulations.Elastic:
    """`bar_simu` on a HEXA8 box."""
    mesh = make_mesh_3D(L=L, H=H, elemType=ElemType.HEXA8, isOrganised=True)
    return pulled_bar(mesh, withBCs=True)


@pytest.fixture
def phasefield_simu(make_mesh_2D) -> Simulations.PhaseField:
    """QUAD4 bar, its damage injected twice: half-way, then broken at x=L."""
    mesh = make_mesh_2D(L=L, H=H, elemType=ElemType.QUAD4, isOrganised=True)
    material = Models.Elastic.Isotropic(2, 210e3, 0.3, planeStress=True)
    pfm = Models.PhaseField(material, "Bourdin", "AT2", 2.7, 0.1)
    simu = Simulations.PhaseField(mesh, pfm)
    x = mesh.coord[:, 0]
    for d_max in (0.5, 1.0):
        u = np.zeros((mesh.Nn, 2))
        u[:, 0] = 0.01 * x
        simu._Set_solutions(simu.ProblemTypes.elastic, u.ravel())
        simu._Set_solutions(simu.ProblemTypes.damage, d_max * x / L)
        simu.Save_Iter()
    return simu


@pytest.fixture
def frame_simu() -> Simulations.Beam:
    """2D frame: clamped at the origin, hinged at (L, 0), a moment at (L, L)."""
    origin, corner, tip = (0, 0), (L, 0), (L, L)
    domain = Domain((-0.1, -0.1), (0.1, 0.1))
    section = domain.Mesh_2D([], ElemType.TRI3)
    line1 = Line(origin, corner, L / 4)
    line2 = Line(corner, tip, L / 4)
    beams = [
        Models.Beam.Isotropic(2, line, section, 210e3, 0.3) for line in (line1, line2)
    ]
    mesh = Mesher().Mesh_Beams(beams)
    simu = Simulations.Beam(mesh, Models.Beam.BeamStructure(beams))
    nodesClamped = mesh.Nodes_Point(origin)
    nodesCorner = mesh.Nodes_Point(corner)
    nodesTip = mesh.Nodes_Point(tip)
    simu.add_dirichlet(nodesClamped, [0, 0, 0], ["x", "y", "rz"])
    simu.add_connection_hinged(nodesCorner)
    simu.add_neumann(nodesTip, [1.0], ["rz"])
    return simu
