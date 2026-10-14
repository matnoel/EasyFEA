# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA.FEM import ElemType, Operators

from _numerics import H3, affine, element_dot, mandel, rel_error
from .._reference import MEASURES, MeshOf
from ._reference import assert_restricts

TYPES = [ElemType.TRI3, ElemType.TETRA4]


def test_V_of_a_constant_sums_to_its_integral(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        for dof_n in [1, 2]:
            with subtests.test(elemType=str(elemType), dof_n=dof_n):
                groupElem = mesh_of(elemType).groupElem

                values_e = Operators.Linear.V(groupElem, 3.0, dof_n)

                assert values_e.shape == (groupElem.Ne, groupElem.nPe * dof_n)
                expected = 3.0 * MEASURES[groupElem.dim] * dof_n
                assert rel_error(values_e.sum(), expected) < 1e-12


def test_V_per_element_and_restricted(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            f_e = np.arange(groupElem.Ne, dtype=float)
            elements = np.arange(0, groupElem.Ne, 2)

            values_e = Operators.Linear.V(groupElem, f_e, elements=elements)

            volume_e = groupElem.Integrate_e()
            expected = f_e[elements] * volume_e[elements]
            assert rel_error(values_e.sum(1)[elements], expected) < 1e-12
            assert not np.any(np.delete(values_e, elements, axis=0))


def test_InternalForce_of_a_constant_stress_does_its_virtual_work(
    mesh_of: MeshOf, subtests
):
    """δu = H x: F · δu = volume · σ : sym(H)."""
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            mesh = mesh_of(elemType)
            groupElem = mesh.groupElem
            dim = groupElem.dim
            H = H3[:dim, :dim]
            sigma = np.array([1.0, -2.0, 0.5, 0.3, -0.7, 1.1])[: 3 if dim == 2 else 6]
            du = affine(mesh.coord, H)

            F_e = Operators.Linear.InternalForce(groupElem, sigma)

            work = element_dot(groupElem, F_e, du)
            expected = MEASURES[dim] * sigma @ mandel(H)
            assert rel_error(work, expected) < 1e-10


def test_InternalForce_restricted_to_elements(mesh_of: MeshOf, subtests):
    for elemType in TYPES:
        with subtests.test(elemType=str(elemType)):
            groupElem = mesh_of(elemType).groupElem
            sigma = np.ones(3 if groupElem.dim == 2 else 6)

            assert_restricts(
                lambda e: Operators.Linear.InternalForce(groupElem, sigma, e),
                np.array([0]),
            )
