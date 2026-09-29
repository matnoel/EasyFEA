# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest
import numpy as np

from EasyFEA import ElemType, MatrixType
from EasyFEA.FEM._gauss import Gauss

REFERENCE_MEASURES = {
    "SEG": 2.0,
    "TRI": 1 / 2,
    "QUAD": 4.0,
    "TETRA": 1 / 6,
    "HEXA": 8.0,
    "PRISM": 1.0,
}


class TestGauss:

    @pytest.mark.parametrize("elemType", [e for e in ElemType if e != ElemType.POINT])
    @pytest.mark.parametrize("matrixType", [MatrixType.rigi, MatrixType.mass])
    def test_weights_sum_to_reference_measure(
        self, elemType: ElemType, matrixType: MatrixType
    ):
        gauss = Gauss(elemType, matrixType)

        measure = REFERENCE_MEASURES[elemType.topology]

        assert np.abs(gauss.weights.sum() - measure) / measure < 1e-14
