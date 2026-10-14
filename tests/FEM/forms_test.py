# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Patch tests: on an L × H rectangle, the mass and `LinearForm(1)` sum to the area, and the stiffness of an affine field is exact."""

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, Mesh, Field, BiLinearForm, LinearForm

from _numerics import rel_error

AREA = 2.0


@pytest.fixture
def meshes(mesh_of) -> list[Mesh]:
    """2 × 1 rectangle of each 2D element type."""
    return [mesh_of(elemType) for elemType in ElemType.Get_2D()]


MASS = BiLinearForm(lambda u, v: u.dot(v))
ONE = LinearForm(lambda v: 1.0 * v)
FORMS = {"BiLinearForm": (MASS, 2), "LinearForm": (ONE, 1)}


def test_Form_calls_its_function():
    form = BiLinearForm(lambda u, v: u + v)

    assert form(1, 2) == 3


def test_Form_Integrate_e_sums_to_the_area(meshes: list[Mesh], subtests):
    for mesh in meshes:
        for formName, (form, rank) in FORMS.items():
            with subtests.test(elemType=str(mesh.elemType), form=formName):
                field = Field(mesh.groupElem, 1)

                values_e = form.Integrate_e(field)

                assert values_e.shape == (mesh.Ne,) + (mesh.groupElem.nPe,) * rank
                assert rel_error(values_e.sum(), AREA) < 1e-12


def test_Form_Assemble_sums_to_the_area(meshes: list[Mesh], subtests):
    for mesh in meshes:
        for formName, (form, rank) in FORMS.items():
            with subtests.test(elemType=str(mesh.elemType), form=formName):
                field = Field(mesh.groupElem, 1)

                assembled = form.Assemble(field)

                assert assembled.shape == (mesh.Nn, mesh.Nn if rank == 2 else 1)
                assert rel_error(assembled.sum(), AREA) < 1e-12


def test_BiLinearForm_stiffness_of_an_affine_field_is_exact(
    meshes: list[Mesh], subtests
):
    """fᵀ K f = ∫ |∇f|² = |a|² · area for f = a · x."""
    for mesh in meshes:
        with subtests.test(elemType=str(mesh.elemType)):
            a = np.array([1.0, -2.0])
            f = mesh.coord[:, :2] @ a
            field = Field(mesh.groupElem, 1)
            stiffness = BiLinearForm(lambda u, v: u.grad.dot(v.grad))

            K = stiffness.Assemble(field)

            assert rel_error(f @ K @ f, a @ a * AREA) < 1e-10
            assert np.abs(K @ np.ones(mesh.Nn)).max() < 1e-10


def test_BiLinearForm_vector_mass_keeps_the_components_apart(
    meshes: list[Mesh], subtests
):
    """Each of the 2 components carries the area, and x never meets y."""
    for mesh in meshes:
        with subtests.test(elemType=str(mesh.elemType)):
            field = Field(mesh.groupElem, 2)

            M = MASS.Assemble(field)

            assert rel_error(M.sum(), 2 * AREA) < 1e-12
            assert np.abs(M[0::2][:, 1::2]).max() == 0


def test_LinearForm_vector_load_sums_to_the_load_times_the_area(
    meshes: list[Mesh], subtests
):
    for mesh in meshes:
        with subtests.test(elemType=str(mesh.elemType)):
            load = np.array([3.0, -1.0])
            field = Field(mesh.groupElem, 2)
            form = LinearForm(lambda v: v.dot(load))

            F = form.Assemble(field).toarray().ravel()

            assert rel_error(F.reshape(-1, 2).sum(0), load * AREA) < 1e-12
