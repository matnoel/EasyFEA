# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, Union

import numpy as np

from .._linalg import FeArray
from .._utils import MatrixType
from ...Utilities import _types

if TYPE_CHECKING:
    from .._group_elem import _GroupElem


def V(
    groupElem: "_GroupElem",
    f: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    dof_n: int = 1,
    matrixType: MatrixType = MatrixType.mass,
) -> np.ndarray:
    """``∫_Ω f · v dΩ`` — returns ``(Ne, nPe·dof_n)``.

    ``dof_n=1`` for scalar fields (thermal, phase-field);
    ``dof_n=dim`` for vector fields (elastic dynamics).

    ``f`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :meth:`FeArray.broadcast` (stride view, no copy).
    """
    vec_e_pg = groupElem.Get_SourcePart_e_pg(matrixType, dof_n)
    Ne, nPg = vec_e_pg.shape[:2]
    f = FeArray.broadcast(f, Ne, nPg)
    return (f * vec_e_pg).integrate()


def InternalForce(
    groupElem: "_GroupElem",
    sigma_e_pg: FeArray.FeArrayALike,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω σ : ε(v) dΩ`` — internal force of a known stress field.

    Returns ``(Ne, nPe·dim)``.

    ``sigma_e_pg`` is Kelvin-Mandel with shape ``(Ne, nPg, nstrain)``. This is the residual side
    of a nonlinear problem: unlike :func:`Bilinear.LinearizedElasticity`, the stress is given
    rather than derived from ``C : ε(u)``.
    """
    leftDispPart_e_pg = groupElem.Get_leftDispPart_e_pg(matrixType)
    return (leftDispPart_e_pg @ FeArray.asfearray(sigma_e_pg)).integrate()
