# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""The affine kinematics fixture: its expected ε and F are what Kinematics computes at every Gauss point."""

import numpy as np

from _numerics import rel_error
from .conftest import AffineKinematics


def test_epsilon_is_the_expected_one(affine: AffineKinematics):
    epsilon = np.asarray(affine.kinematics.Compute_Epsilon())
    assert rel_error(epsilon, np.broadcast_to(affine.epsilon, epsilon.shape)) < 1e-12


def test_F_is_the_expected_one(affine: AffineKinematics):
    F = np.asarray(affine.kinematics.Compute_F())
    assert rel_error(F, np.broadcast_to(affine.F, F.shape)) < 1e-12


def test_strain_is_not_trivial(affine: AffineKinematics):
    dim = affine.kinematics.dim
    assert np.all(affine.F[:dim, :dim] != np.eye(dim))
