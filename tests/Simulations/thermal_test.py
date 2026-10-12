# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import Mesh, Models, Simulations, SolverType
from EasyFEA.Geoms import Domain, Point

from _numerics import rel_error
from . import _reference

T0, TL = 10.0, 40.0


def _heated_slab(mesh: Mesh) -> Simulations.Thermal:
    L = mesh.coord[:, 0].max()
    simu = Simulations.Thermal(mesh, Models.Thermal(k=1.0, c=1.0))
    simu.solver = SolverType.scipy  # direct: petsc's iterative default stops at ~1e-6
    nodesX0 = mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesX0, [T0], ["t"])
    simu.add_dirichlet(nodesXL, [TL], ["t"])
    simu.Solve()
    return simu


def test_heat_slab_is_solved_exactly(box_mesh: Mesh):
    simu = _heated_slab(box_mesh)
    L = box_mesh.coord[:, 0].max()

    expected = _reference.slab_temperature(box_mesh.coord[:, 0], L, T0, TL)
    assert rel_error(simu.thermal, expected) < 1e-10


def test_a_steady_slab_stays_steady_under_the_parabolic_scheme(box_mesh: Mesh):
    simu = _heated_slab(box_mesh)
    steady = simu.thermal.copy()

    simu.Solver_Set_Parabolic_Algorithm(dt=0.1)
    simu.Solve()

    assert rel_error(simu.thermal, steady) < 1e-10


CHANGES = {
    "k": lambda thermal: setattr(thermal, "k", 2 * thermal.k),
    "c": lambda thermal: setattr(thermal, "c", 0.2 * thermal.c),
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_a_material_change_flags_the_simulation_for_reassembly(change):
    domain = Domain(Point(), Point(1, 1))
    mesh = domain.Mesh_2D()
    thermal = Models.Thermal(1, 1)
    simu = Simulations.Thermal(mesh, thermal)
    simu.Get_K_C_M_F()
    assert not simu.needUpdate

    change(thermal)

    assert simu.needUpdate
