# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.FEM import BoundaryCondition, LagrangeCondition

UNKNOWNS = ["x", "y", "z"]


def _condition(problemType: str = "elastic", value: float = 1.0) -> BoundaryCondition:
    """x and z of nodes 0 and 2."""
    nodes = np.array([0, 2])
    dofs = BoundaryCondition.Get_dofs_nodes(UNKNOWNS, nodes, ["x", "z"])
    return BoundaryCondition(
        problemType,
        nodes,
        dofs,
        ["x", "z"],
        np.full(dofs.size, value),
        "clamped",
    )


def test_BoundaryCondition_keeps_its_description():
    assert _condition().description == "clamped"


def test_BoundaryCondition_needs_whole_nodes_of_dofs():
    with pytest.raises(AssertionError):
        BoundaryCondition("elastic", [0, 1], [0, 1, 2], ["x"], [0, 0, 0], "")


def test_BoundaryCondition_problemType_is_the_given_one():
    assert _condition("thermal").problemType == "thermal"


def test_BoundaryCondition_nodes_are_a_copy():
    bc = _condition()

    bc.nodes[:] = -1

    assert np.array_equal(bc.nodes, [0, 2])


def test_BoundaryCondition_dofs_are_a_copy():
    bc = _condition()

    bc.dofs[:] = -1

    assert np.array_equal(bc.dofs, [0, 2, 6, 8])


def test_BoundaryCondition_dofsValues_are_a_float_copy():
    bc = _condition(value=3)

    bc.dofsValues[:] = -1

    assert bc.dofsValues.dtype == float
    assert np.array_equal(bc.dofsValues, [3, 3, 3, 3])


def test_BoundaryCondition_unknowns_are_a_copy():
    bc = _condition()

    bc.unknowns.append("y")

    assert bc.unknowns == ["x", "z"]


def test_BoundaryCondition_Get_nBc_counts_the_conditions_of_a_problem():
    conditions = [_condition("elastic"), _condition("thermal"), _condition("elastic")]

    assert BoundaryCondition.Get_nBc("elastic", conditions) == 2
    assert BoundaryCondition.Get_nBc("damage", conditions) == 0


def test_BoundaryCondition_Get_dofs_concatenates_the_dofs_of_a_problem():
    conditions = [_condition("elastic"), _condition("thermal"), _condition("elastic")]

    dofs = BoundaryCondition.Get_dofs("elastic", conditions)

    assert np.array_equal(dofs, [0, 2, 6, 8] * 2)
    assert BoundaryCondition.Get_dofs("damage", conditions).dtype == int


def test_BoundaryCondition_Get_values_concatenates_the_values_of_a_problem():
    conditions = [_condition("elastic", 1), _condition("thermal", 2)]
    conditions.append(_condition("elastic", 3))

    values = BoundaryCondition.Get_values("elastic", conditions)

    assert np.array_equal(values, [1] * 4 + [3] * 4)


def test_BoundaryCondition_Get_dofs_nodes_numbers_node_by_node():
    """dof = node · len(availableUnknowns) + index of the unknown."""
    dofs = BoundaryCondition.Get_dofs_nodes(UNKNOWNS, [[1], [4]], ["z", "x"])

    assert np.array_equal(dofs, [5, 3, 14, 12])


def test_BoundaryCondition_Get_dofs_nodes_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="rz"):
        BoundaryCondition.Get_dofs_nodes(UNKNOWNS, [0, 1], ["x", "rz"])


def test_LagrangeCondition_is_a_boundary_condition_with_coefficients():
    """u0 - u1 = 0: two dofs, one value."""
    bc = LagrangeCondition("elastic", [0, 1], [0, 2], ["x"], [0.0], [1, -1])

    assert isinstance(bc, BoundaryCondition)
    assert bc.description == ""
    assert bc.dofs.size == 2 and bc.dofsValues.size == 1


def test_LagrangeCondition_lagrangeCoefs_are_a_copy():
    bc = LagrangeCondition("elastic", [0, 1], [0, 2], ["x"], [0.0], [1, -1])

    bc.lagrangeCoefs[:] = 0

    assert np.array_equal(bc.lagrangeCoefs, [1, -1])
