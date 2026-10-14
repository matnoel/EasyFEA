# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA import ElemType
from EasyFEA.FEM import BiLinearForm, Field, LinearForm
from EasyFEA.Geoms import Domain
from EasyFEA.Models import WeakForms

STIFFNESS = BiLinearForm(lambda u, v: u.grad.dot(v.grad))
MASS = BiLinearForm(lambda u, v: u.dot(v))
LOAD = LinearForm(lambda v: v)


@pytest.fixture(scope="module")
def field() -> Field:
    domain = Domain((0, 0), (1, 1), 0.5)
    return Field(domain.Mesh_2D([], ElemType.TRI3).groupElem, 2)


def test_WeakForms_thickness_is_the_given_one(field: Field):
    assert WeakForms(field, STIFFNESS, thickness=2.0).thickness == 2.0


def test_WeakForms_field_is_read_only(field: Field):
    weakForms = WeakForms(field, STIFFNESS)

    assert weakForms.field is field
    with pytest.raises(AttributeError):
        weakForms.field = field  # type: ignore[misc]


def test_WeakForms_computeK_is_the_stiffness_form(field: Field):
    assert WeakForms(field, STIFFNESS).computeK is STIFFNESS


def test_WeakForms_computeC_defaults_to_none(field: Field):
    assert WeakForms(field, STIFFNESS).computeC is None
    assert WeakForms(field, STIFFNESS, computeC=MASS).computeC is MASS


def test_WeakForms_computeM_defaults_to_none(field: Field):
    assert WeakForms(field, STIFFNESS).computeM is None
    assert WeakForms(field, STIFFNESS, computeM=MASS).computeM is MASS


def test_WeakForms_computeF_defaults_to_none(field: Field):
    assert WeakForms(field, STIFFNESS).computeF is None
    assert WeakForms(field, STIFFNESS, computeF=LOAD).computeF is LOAD


def test_WeakForms_dim_is_the_dofs_per_node(field: Field):
    assert WeakForms(field, STIFFNESS).dim == field.dof_n == 2
