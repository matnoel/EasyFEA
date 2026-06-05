# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, Union

from .._linalg import FeArray
from .._utils import MatrixType
from ...Utilities import _types

if TYPE_CHECKING:
    from .._group_elem import _GroupElem


def GradUGradV(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.rigi,
) -> FeArray.FeArrayALike:
    """``∫_Ω coef · ∇u · ∇v dΩ`` — returns ``(Ne, nPe, nPe)``.

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :pymeth:`FeArray.broadcast` (stride view, no copy).
    """
    mat_e_pg = groupElem.Get_DiffusePart_e_pg(matrixType)
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    Ne, nPg = dN_e_pg.shape[:2]
    coef = FeArray.broadcast(coef, Ne, nPg)
    return (coef * mat_e_pg @ dN_e_pg).sum(axis=1)


def UV(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    dof_n: int = 1,
    matrixType: MatrixType = MatrixType.mass,
) -> FeArray.FeArrayALike:
    """``∫_Ω coef · u · v dΩ`` — returns ``(Ne, nPe·dof_n, nPe·dof_n)``.

    ``dof_n=1`` for scalar fields (thermal, phase-field);
    ``dof_n=dim`` for vector fields (elastic dynamics).

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :pymeth:`FeArray.broadcast` (stride view, no copy).
    """
    mat_e_pg = groupElem.Get_ReactionPart_e_pg(matrixType, dof_n)
    Ne, nPg = mat_e_pg.shape[:2]
    coef = FeArray.broadcast(coef, Ne, nPg)
    return (coef * mat_e_pg).sum(axis=1)
