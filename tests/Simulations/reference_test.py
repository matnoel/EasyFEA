# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The exact references hold: bar and slab are solved exactly on every element, beam closed forms are self-consistent."""

import numpy as np
import pytest

from EasyFEA import Mesh, Models, Simulations, SolverType

from _numerics import rel_error
from . import _reference

E, v = 210e3, 0.3
UD = 1e-3
T0, TL = 10.0, 40.0
SYMMETRY_PLANES = {
    "x": lambda x, y, z: x == 0,
    "y": lambda x, y, z: y == 0,
    "z": lambda x, y, z: z == 0,
}


def test_bar_in_tension_is_solved_exactly(box_mesh: Mesh):
    dim = box_mesh.dim
    L = box_mesh.coord[:, 0].max()
    simu = Simulations.Elastic(
        box_mesh, Models.Elastic.Isotropic(dim, E, v, planeStress=True)
    )
    simu.solver = SolverType.scipy
    for unknown in simu.Get_unknowns():
        nodesSymmetry = box_mesh.Nodes_Conditions(SYMMETRY_PLANES[unknown])
        simu.add_dirichlet(nodesSymmetry, [0], [unknown])
    nodesPulled = box_mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesPulled, [UD], ["x"])
    simu.Solve()

    expected = _reference.bar_displacement(
        box_mesh.coord,
        L,
        UD,
        v,
        dim,
    )
    assert rel_error(simu.displacement, expected.ravel()) < 1e-10
    Sxx = simu.Result("Sxx", nodeValues=False)
    assert rel_error(Sxx, _reference.bar_stress(E, L, UD)) < 1e-10


def test_heat_slab_is_solved_exactly(box_mesh: Mesh):
    L = box_mesh.coord[:, 0].max()
    simu = Simulations.Thermal(box_mesh, Models.Thermal(k=1.0))
    simu.solver = SolverType.scipy  # direct: petsc's iterative default stops at ~1e-6
    nodesX0 = box_mesh.Nodes_Conditions(lambda x, y, z: x == 0)
    nodesXL = box_mesh.Nodes_Conditions(lambda x, y, z: x == L)
    simu.add_dirichlet(nodesX0, [T0], ["t"])
    simu.add_dirichlet(nodesXL, [TL], ["t"])
    simu.Solve()

    expected = _reference.slab_temperature(box_mesh.coord[:, 0], L, T0, TL)
    assert rel_error(simu.thermal, expected) < 1e-10


L_BEAM, A_LOAD, F, Q = 120.0, 80.0, -800.0, -10.0
EI = 210e3 * _reference.rect_Iz(13.0, 13.0)
BEAMS = {
    "cantilever_tip": _reference.cantilever_tip(L_BEAM, F, EI),
    "cantilever_distributed": _reference.cantilever_distributed(L_BEAM, Q, EI),
    "simply_supported": _reference.simply_supported(L_BEAM, A_LOAD, F, EI),
    "simply_supported_distributed": _reference.simply_supported_distributed(
        L_BEAM, Q, EI
    ),
    "fixed_fixed": _reference.fixed_fixed(L_BEAM, A_LOAD, F, EI),
}
# off the load point, where the Macaulay terms are smooth
X = np.concatenate([np.linspace(1, A_LOAD - 1, 20), np.linspace(A_LOAD + 1, 119, 10)])
H = 1e-3


def _derivative(f, x):
    return (f(x + H) - f(x - H)) / (2 * H)


@pytest.mark.parametrize("beam", BEAMS.values(), ids=BEAMS.keys())
def test_beam_rotation_is_the_slope(beam: _reference.BeamFields):
    assert rel_error(beam.rz(X), _derivative(beam.uy, X)) < 1e-6


@pytest.mark.parametrize("beam", BEAMS.values(), ids=BEAMS.keys())
def test_beam_moment_is_the_curvature(beam: _reference.BeamFields):
    assert rel_error(beam.Mz(X), EI * _derivative(beam.rz, X)) < 1e-6


@pytest.mark.parametrize("beam", BEAMS.values(), ids=BEAMS.keys())
def test_beam_shear_is_minus_the_moment_slope(beam: _reference.BeamFields):
    assert rel_error(beam.Ty(X), -_derivative(beam.Mz, X)) < 1e-6
