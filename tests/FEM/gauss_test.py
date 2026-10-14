# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from itertools import product
from math import factorial

import pytest
import numpy as np

from EasyFEA.FEM import ElemType, MatrixType, Gauss

from ._reference import REFERENCE_MEASURES


def _segment(a: int) -> float:
    """∫ x^a over [-1, 1]"""
    return 0.0 if a % 2 else 2 / (a + 1)


def _simplex(*powers: int) -> float:
    """∫ x^a y^b (z^c) over the unit simplex"""
    num = np.prod([factorial(p) for p in powers])
    return num / factorial(sum(powers) + len(powers))


def _assert_exact(points, weights, exact, order: int, dim: int) -> None:
    """The rule integrates every monomial of total degree ≤ `order` exactly."""
    coords = np.asarray(points).reshape(dim, -1)
    for powers in product(range(order + 1), repeat=dim):
        if sum(powers) > order:
            continue
        monomial = np.prod([c**p for c, p in zip(coords, powers)], axis=0)
        assert abs(np.dot(weights, monomial) - exact(*powers)) < 1e-12, powers


def _assert_raises_unknown_nPg(rule) -> None:
    with pytest.raises(NotImplementedError):
        rule(2)


def test_Gauss_weights_sum_to_the_reference_measure(subtests):
    for elemType in [e for e in ElemType if e != ElemType.POINT]:
        for matrixType in [MatrixType.rigi, MatrixType.mass]:
            with subtests.test(elemType=str(elemType), matrixType=str(matrixType)):
                gauss = Gauss(elemType, matrixType)

                measure = REFERENCE_MEASURES[elemType.topology]

                assert abs(gauss.weights.sum() - measure) / measure < 1e-14


def test_Gauss_rejects_an_unknown_matrix_type():
    with pytest.raises(NotImplementedError):
        Gauss(ElemType.TRI3, 1.5)  # type: ignore [arg-type]


def test_Gauss_coord_lies_in_the_closed_reference_element(subtests):
    for elemType in [e for e in ElemType if e != ElemType.POINT]:
        with subtests.test(elemType=str(elemType)):
            coord = Gauss(elemType, MatrixType.mass).coord

            topology = elemType.topology
            if topology in ["SEG", "QUAD", "HEXA"]:
                assert np.all(np.abs(coord) <= 1)
            elif topology in ["TRI", "TETRA"]:
                assert np.all(coord >= 0) and np.all(coord.sum(1) <= 1)
            else:
                assert np.all(coord[:, :2] >= 0) and np.all(coord[:, :2].sum(1) <= 1)
                assert np.all(np.abs(coord[:, 2]) <= 1)


def test_Gauss_nPg_counts_the_points():
    gauss = Gauss(ElemType.HEXA27, MatrixType.rigi)

    assert gauss.nPg == 27 == gauss.coord.shape[0]


def test_Gauss__Triangle_integrates_its_order_exactly(subtests):
    for nPg, order in [(1, 1), (3, 2), (6, 3), (7, 4), (12, 5)]:
        with subtests.test(nPg=nPg):
            ksis, etas, weights = Gauss._Triangle(nPg)

            _assert_exact(
                [ksis, etas],
                weights,
                _simplex,
                order,
                2,
            )


def test_Gauss__Triangle_rejects_an_unknown_nPg():
    _assert_raises_unknown_nPg(Gauss._Triangle)


def test_Gauss__Quadrangle_integrates_its_order_exactly(subtests):
    for nPg, order in [(4, 3), (9, 5)]:
        with subtests.test(nPg=nPg):
            ksis, etas, weights = Gauss._Quadrangle(nPg)

            def exact(a, b):
                return _segment(a) * _segment(b)

            _assert_exact(
                [ksis, etas],
                weights,
                exact,
                order,
                2,
            )


def test_Gauss__Quadrangle_rejects_an_unknown_nPg():
    _assert_raises_unknown_nPg(Gauss._Quadrangle)


def test_Gauss__Tetrahedron_integrates_its_order_exactly(subtests):
    for nPg, order in [(1, 1), (4, 2), (5, 3), (15, 5)]:
        with subtests.test(nPg=nPg):
            x, y, z, weights = Gauss._Tetrahedron(nPg)

            _assert_exact(
                [x, y, z],
                weights,
                _simplex,
                order,
                3,
            )


def test_Gauss__Tetrahedron_rejects_an_unknown_nPg():
    _assert_raises_unknown_nPg(Gauss._Tetrahedron)


def test_Gauss__Hexahedron_integrates_its_order_exactly(subtests):
    for nPg, order in [(8, 3), (27, 5)]:
        with subtests.test(nPg=nPg):
            x, y, z, weights = Gauss._Hexahedron(nPg)

            def exact(a, b, c):
                return _segment(a) * _segment(b) * _segment(c)

            _assert_exact(
                [x, y, z],
                weights,
                exact,
                order,
                3,
            )


def test_Gauss__Hexahedron_rejects_an_unknown_nPg():
    _assert_raises_unknown_nPg(Gauss._Hexahedron)


def test_Gauss__Prism_integrates_its_order_exactly(subtests):
    """Triangle in (x, y) times segment in z."""
    for nPg, order in [(6, 2), (8, 3), (21, 5)]:
        with subtests.test(nPg=nPg):
            x, y, z, weights = Gauss._Prism(nPg)

            def exact(a, b, c):
                return _simplex(a, b) * _segment(c)

            _assert_exact(
                [x, y, z],
                weights,
                exact,
                order,
                3,
            )


def test_Gauss__Prism_rejects_an_unknown_nPg():
    _assert_raises_unknown_nPg(Gauss._Prism)


def test_Gauss_Gauss_factory_reduces_beam_shear_to_one_point_less(subtests):
    """SEGn: n - 1 points for the Timoshenko shear term."""
    for elemType in ElemType.Get_1D():
        with subtests.test(elemType=str(elemType)):
            coord, weights = Gauss.Gauss_factory(elemType, MatrixType.beam_shear)

            nNodes = int(elemType.name.removeprefix("SEG"))
            assert weights.size == coord.shape[0] == nNodes - 1


def test_Gauss_Gauss_factory_segments_are_gauss_legendre(subtests):
    for elemType in ElemType.Get_1D():
        for matrixType in MatrixType.Get_types():
            with subtests.test(elemType=str(elemType), matrixType=str(matrixType)):
                coord, weights = Gauss.Gauss_factory(elemType, matrixType)

                expected = np.polynomial.legendre.leggauss(weights.size)
                assert np.array_equal(coord.ravel(), expected[0])
                assert np.array_equal(weights, expected[1])


def test_Gauss_Gauss_factory_rejects_a_beam_matrix_on_a_non_beam_element(subtests):
    for elemType in [ElemType.TRI3, ElemType.TRI6, ElemType.QUAD8, ElemType.TETRA4]:
        with subtests.test(elemType=str(elemType)):
            with pytest.raises(ValueError):
                Gauss.Gauss_factory(elemType, MatrixType.beam)


def test_Gauss_Gauss_factory_rejects_a_point():
    with pytest.raises(NotImplementedError):
        Gauss.Gauss_factory(ElemType.POINT, MatrixType.rigi)


def test_Gauss__Gauss_factory_nPg_takes_the_requested_number_of_points(subtests):
    for elemType, nPg in [
        (ElemType.SEG3, 7),
        (ElemType.TRI6, 12),
        (ElemType.QUAD4, 9),
        (ElemType.TETRA4, 15),
        (ElemType.HEXA8, 27),
        (ElemType.PRISM6, 21),
    ]:
        with subtests.test(elemType=str(elemType)):
            gauss = Gauss(elemType, nPg)  # type: ignore [arg-type]

            assert gauss.nPg == nPg == gauss.coord.shape[0]
            measure = REFERENCE_MEASURES[elemType.topology]
            assert abs(gauss.weights.sum() - measure) / measure < 1e-12


def test_Gauss__Gauss_factory_nPg_rejects_a_point():
    with pytest.raises(NotImplementedError):
        Gauss._Gauss_factory_nPg(ElemType.POINT, 1)
