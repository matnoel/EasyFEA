# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA.FEM._linalg import Transpose, Trace, Det, Inv


@pytest.fixture
def setup_matrices() -> list[np.ndarray]:

    mat0 = np.ones((1, 1))
    mat1 = np.eye(3)
    mat2 = mat1[np.newaxis, :, :].repeat(10, axis=0)
    mat3 = np.random.random((10, 4, 1, 1))
    mat4 = np.random.random((10, 4, 2, 2))
    mat5 = np.random.random((10, 4, 3, 3))
    mat6 = np.random.random((10, 4, 3, 5, 5))

    list_Mat = [mat0, mat1, mat2, mat3, mat4, mat5, mat6]

    return list_Mat


def Check(mat: np.ndarray, matVerif: np.ndarray):

    diff = mat - matVerif

    test = np.linalg.norm(diff) / np.linalg.norm(matVerif)

    assert test < 1e-12


class TestLinalg:

    def test_Tranpose(self, setup_matrices):

        mat0, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Transpose(mat0), mat0)
        Check(Transpose(mat1), mat1)
        Check(Transpose(mat2), mat2)
        Check(Transpose(mat3), np.transpose(mat3, (0, 1, 3, 2)))
        Check(Transpose(mat4), np.transpose(mat4, (0, 1, 3, 2)))
        Check(Transpose(mat5), np.transpose(mat5, (0, 1, 3, 2)))
        Check(Transpose(mat6), np.transpose(mat6, (0, 1, 2, 4, 3)))

    def test_Trace(self, setup_matrices):

        _, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Trace(mat1), np.trace(mat1, axis1=-2, axis2=-1))
        Check(Trace(mat2), np.trace(mat2, axis1=-2, axis2=-1))
        Check(Trace(mat3), np.trace(mat3, axis1=-2, axis2=-1))
        Check(Trace(mat4), np.trace(mat4, axis1=-2, axis2=-1))
        Check(Trace(mat5), np.trace(mat5, axis1=-2, axis2=-1))
        Check(Trace(mat6), np.trace(mat6, axis1=-2, axis2=-1))

    def test_Det(self, setup_matrices):

        mat0, mat1, mat2, mat3, mat4, mat5, mat6 = setup_matrices

        Check(Det(mat0), mat0[0, 0])
        Check(Det(mat1), np.linalg.det(mat1))
        Check(Det(mat2), np.linalg.det(mat2))
        Check(Det(mat3), np.linalg.det(mat3))
        Check(Det(mat4), np.linalg.det(mat4))
        Check(Det(mat5), np.linalg.det(mat5))
        Check(Det(mat6), np.linalg.det(mat6))

    def test_Inv(self, setup_matrices):

        mat0, mat1, mat2 = setup_matrices[:3]

        mat3 = np.array([[2, 1], [1, 2]])

        mat4 = np.array([[4, 3, 8], [6, 2, 5], [1, 5, 9]])

        Check(Inv(mat0), 1 / mat0)
        Check(Inv(mat1), np.linalg.inv(mat1))
        Check(Inv(mat2), np.linalg.inv(mat2))
        Check(Inv(mat3), np.linalg.inv(mat3))
        Check(Inv(mat4), np.linalg.inv(mat4))
