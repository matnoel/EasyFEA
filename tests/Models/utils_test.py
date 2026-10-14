# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA.Models import Heterogeneous_Array, Thermal
from EasyFEA.Models.Elastic import Isotropic


def test_IModel_dim_is_the_model_dimension():
    assert Thermal(1.0).dim == 1
    assert Isotropic(2).dim == 2
    assert Isotropic(3).dim == 3


def test_IModel_thickness_is_the_given_one():
    assert Isotropic(2, thickness=5.0).thickness == 5.0
    assert Thermal(1.0, thickness=2.0).thickness == 2.0


def test_IModel_isHeterogeneous_when_a_parameter_is_a_field():
    assert Isotropic(3).isHeterogeneous is False
    assert Isotropic(3, E=np.full(4, 210e3)).isHeterogeneous is True


def test_IModel__Get_parameters_in_declaration_order():
    material = Isotropic(
        2,
        E=1.0,
        v=0.2,
        planeStress=False,
        thickness=3.0,
    )
    parameters = material._Get_parameters()

    assert parameters == {
        "dim": 2,
        "thickness": 3.0,
        "planeStress": False,
        "E": 1.0,
        "v": 0.2,
    }


def test_IModel__Get_parameters_drops_the_2d_ones_in_3d():
    assert list(Isotropic(3)._Get_parameters()) == ["dim", "E", "v"]


def test_IModel_Need_Update_clears_what_was_cached():
    material = Isotropic(3)
    C = material.C
    material.__dict__["E"] *= 2  # bypasses the setter, so only Need_Update refreshes C

    assert np.allclose(material.C, C)
    material.Need_Update()
    assert np.allclose(material.C, 2 * C)


def test_Heterogeneous_Array_of_scalars_is_the_matrix():
    assert np.array_equal(Heterogeneous_Array([[1, 2], [3, 4]]), [[1.0, 2], [3, 4]])


def test_Heterogeneous_Array_broadcasts_fields_to_leading_axes():
    field = np.arange(4.0)
    array = Heterogeneous_Array([[1.0, field], [field, 0.0]])

    assert array.shape == (4, 2, 2)
    assert np.array_equal(array[:, 0, 1], field) and np.array_equal(
        array[:, 0, 0], np.ones(4)
    )
