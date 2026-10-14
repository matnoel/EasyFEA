# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import numpy as np

from EasyFEA.FEM import ElemType, MatrixType, GroupElemFactory


def test_POINT_has_no_shape_functions():
    groupElem = GroupElemFactory.Create(
        ElemType.POINT, np.array([[0], [1]]), np.array([[0.0, 0, 0], [1, 0, 0]])
    )

    assert groupElem.dim == 0
    assert groupElem._N() is None
    assert groupElem.Get_N_pg(MatrixType.rigi) is None
    assert groupElem.Get_dN_pg(MatrixType.rigi) is None
