# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import operator

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, MatrixType, Mesh, FeArray, Field, Sym_Grad, Trace
from EasyFEA.Geoms import Domain

from _numerics import affine, rel_error

G = np.array([[0.1, 0.2], [-0.3, 0.4]])
"""gradient of the affine vector field x ↦ G x"""


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    """2 × 1 TRI6 rectangle."""
    contour = Domain((0, 0), (2, 1), 0.5)
    return contour.Mesh_2D([], ElemType.TRI6, isOrganised=True)


def _shape_function(field: Field) -> np.ndarray:
    """N of the active node at the Gauss points, (Ne, nPg)."""
    N_pg = field.groupElem.Get_N_pg(field.matrixType)[
        :, 0, field._Get_current_active_node()
    ]
    return np.broadcast_to(N_pg, (field.groupElem.Ne, N_pg.size))


def test_Field_needs_a_group_and_a_dof_n_within_its_dimension(mesh: Mesh):
    with pytest.raises(AssertionError):
        Field(mesh, 1)  # type: ignore [arg-type]
    with pytest.raises(AssertionError):
        Field(mesh.groupElem, 3)


@pytest.mark.parametrize("dof_n", [1, 2])
def test_Field_value_is_the_active_shape_function_on_the_active_dof(
    mesh: Mesh, dof_n: int
):
    field = Field(mesh.groupElem, dof_n)
    field._Set_current_active_node(2)
    field._Set_current_active_dof(dof_n - 1)

    values = np.asarray(field())

    assert values.shape == (*_shape_function(field).shape, dof_n)
    assert np.array_equal(values[..., dof_n - 1], _shape_function(field))
    assert not np.any(values[..., : dof_n - 1])


OPERATORS = {
    "*": operator.mul,
    "+": operator.add,
    "-": operator.sub,
    "/": operator.truediv,
}


@pytest.mark.parametrize("op", OPERATORS.values(), ids=OPERATORS.keys())
def test_Field_arithmetic_acts_on_its_value_in_either_order(mesh: Mesh, op):
    field = Field(mesh.groupElem, 1)
    N = np.asarray(field())

    assert rel_error(np.asarray(op(field, 2.0)), op(N, 2.0)) < 1e-12
    assert rel_error(np.asarray(op(2.0, field)), op(2.0, N)) < 1e-12


def test_Field_matmul_acts_on_its_value_in_either_order(mesh: Mesh):
    field = Field(mesh.groupElem, 2)
    field._Set_current_active_dof(1)
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    value = np.asarray(field())

    assert rel_error(np.asarray(field @ A), value @ A, scale=1.0) < 1e-12
    assert rel_error(np.asarray(A @ field), value @ A.T, scale=1.0) < 1e-12


def test_Field_groupElem_is_the_given_group(mesh: Mesh):
    assert Field(mesh.groupElem, 1).groupElem is mesh.groupElem


def test_Field_dof_n_is_the_given_one(mesh: Mesh):
    assert Field(mesh.groupElem, 2).dof_n == 2


def test_Field_matrixType_sets_the_gauss_points(mesh: Mesh):
    rigi = Field(mesh.groupElem, 1, MatrixType.rigi)
    mass = Field(mesh.groupElem, 1)

    assert rigi.matrixType == MatrixType.rigi and mass.matrixType == MatrixType.mass
    assert rigi().shape[1] < mass().shape[1]


def test_Field_Get_coords_are_the_gauss_point_coordinates(mesh: Mesh):
    field = Field(mesh.groupElem, 1)
    expected = np.asarray(mesh.groupElem.Get_GaussCoordinates_e_pg(MatrixType.mass))

    x, y, z = field.Get_coords()

    assert np.array_equal(np.asarray(field.Get_coords(concatenate=True)), expected)
    assert np.array_equal(np.stack([x, y, z], -1), expected)


def test_Field_copy_is_independent(mesh: Mesh):
    field = Field(mesh.groupElem, 1)

    copied = field.copy()
    copied._Set_current_active_node(1)
    copied._Get_dofsValues()[:] = 1

    assert field._Get_current_active_node() == 0
    assert not np.any(field._Get_dofsValues())


def test_Field__Get_current_active_node_starts_at_zero(mesh: Mesh):
    assert Field(mesh.groupElem, 1)._Get_current_active_node() == 0


def test_Field__Set_current_active_node_stays_within_the_element(mesh: Mesh):
    field = Field(mesh.groupElem, 1)

    field._Set_current_active_node(5)

    assert field._Get_current_active_node() == 5
    with pytest.raises(AssertionError):
        field._Set_current_active_node(6)


def test_Field__Get_current_active_dof_starts_at_zero(mesh: Mesh):
    assert Field(mesh.groupElem, 2)._Get_current_active_dof() == 0


def test_Field__Set_current_active_dof_stays_within_dof_n(mesh: Mesh):
    field = Field(mesh.groupElem, 2)

    field._Set_current_active_dof(1)

    assert field._Get_current_active_dof() == 1
    with pytest.raises(AssertionError):
        field._Set_current_active_dof(2)


def test_Field__Get_dofsValues_starts_at_zero(mesh: Mesh):
    values = Field(mesh.groupElem, 2)._Get_dofsValues()

    assert values.shape == (2 * mesh.Nn,) and not np.any(values)


def test_Field__Set_dofsValues_needs_one_value_per_dof(mesh: Mesh):
    field = Field(mesh.groupElem, 2)

    field._Set_dofsValues(np.ones(2 * mesh.Nn))

    assert np.all(field._Get_dofsValues() == 1)
    with pytest.raises(AssertionError):
        field._Set_dofsValues(np.ones(mesh.Nn))


def test_Field_dot_contracts_the_values(mesh: Mesh):
    u, v = Field(mesh.groupElem, 2), Field(mesh.groupElem, 2)
    v._Set_current_active_dof(1)

    assert rel_error(np.asarray(u.dot(u)), _shape_function(u) ** 2) < 1e-12
    assert not np.any(u.dot(v))


def test_Field_ddot_contracts_the_gradients(mesh: Mesh):
    u = Field(mesh.groupElem, 2)
    dN = np.asarray(mesh.groupElem.Get_dN_e_pg(MatrixType.mass))[..., 0]

    assert rel_error(np.asarray(u.grad.ddot(u.grad)), (dN**2).sum(-1)) < 1e-12
    with pytest.raises(ValueError):
        u.ddot(u)


@pytest.mark.parametrize("dof_n", [1, 2])
def test_Field_grad_of_the_shape_functions_reproduces_the_coordinates(
    mesh: Mesh, dof_n: int
):
    """Σₙ xₙ ⊗ ∇Nₙ = I: the gradient of x is the identity."""
    field = Field(mesh.groupElem, dof_n)
    connect = mesh.groupElem.connect

    total = 0.0
    for node in range(mesh.groupElem.nPe):
        field._Set_current_active_node(node)
        grad = np.asarray(field.grad)
        if dof_n == 2:
            grad = grad[..., field._Get_current_active_dof()]
        total += np.einsum("ei,epj->epij", mesh.coord[connect[:, node], :2], grad)

    assert rel_error(total, np.broadcast_to(np.eye(2), total.shape), scale=1.0) < 1e-10


def test_Field_grad_while_evaluated_is_the_field_gradient(mesh: Mesh):
    field = Field(mesh.groupElem, 2)

    grad_e = field.Evaluate_e(lambda u: u.grad, affine(mesh.coord, G))

    assert rel_error(grad_e, np.broadcast_to(G, grad_e.shape)) < 1e-10


def test_Field_Evaluate_e_gives_element_means_or_gauss_values(mesh: Mesh):
    field = Field(mesh.groupElem, 2)
    x_e_pg = np.asarray(field.Get_coords(concatenate=True))[..., :2]

    values_e = field.Evaluate_e(lambda u: u() * 1.0, affine(mesh.coord, G))
    values_e_pg = field.Evaluate_e(lambda u: u() * 1.0, affine(mesh.coord, G), False)

    assert rel_error(np.asarray(values_e_pg), x_e_pg @ G.T) < 1e-12
    assert rel_error(values_e, (x_e_pg @ G.T).mean(1)) < 1e-12


def test_Field_Evaluate_e_needs_a_FeArray(mesh: Mesh):
    field = Field(mesh.groupElem, 1)

    with pytest.raises(AssertionError):
        field.Evaluate_e(lambda u: 1.0, np.zeros(mesh.Nn))  # type: ignore [arg-type, return-value]


def test_Field_Evaluate_n_gives_nodal_values(mesh: Mesh):
    field = Field(mesh.groupElem, 2)

    values_n = field.Evaluate_n(lambda u: Trace(u.grad), affine(mesh.coord, G))

    assert values_n.shape == (mesh.Nn,)
    assert rel_error(values_n, np.trace(G)) < 1e-10


@pytest.mark.parametrize("dof_n", [1, 2])
def test_Field_Interpolate_is_exact_on_an_affine_field(mesh: Mesh, dof_n: int):
    field = Field(mesh.groupElem, 1)
    values = affine(mesh.coord, G).reshape(-1, 2)[:, :dof_n].ravel()
    x_e_pg = np.asarray(field.Get_coords(concatenate=True))[..., :2]

    values_e_pg = field.Interpolate(values)

    assert isinstance(values_e_pg, FeArray)
    assert rel_error(np.asarray(values_e_pg), (x_e_pg @ G.T)[..., :dof_n]) < 1e-12


def test_Sym_Grad_is_the_symmetric_gradient(mesh: Mesh):
    field = Field(mesh.groupElem, 2)

    strain_e = field.Evaluate_e(Sym_Grad, affine(mesh.coord, G))

    expected = (G + G.T) / 2
    assert rel_error(strain_e, np.broadcast_to(expected, strain_e.shape)) < 1e-10
