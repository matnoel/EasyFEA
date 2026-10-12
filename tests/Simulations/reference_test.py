# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The beam closed forms are self-consistent; the bar and slab are checked in `elastic_test` and `thermal_test`."""

import numpy as np
import pytest

from _numerics import rel_error
from . import _reference

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
