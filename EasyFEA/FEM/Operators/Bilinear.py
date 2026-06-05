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


def GradUGradV(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω coef · ∇u · ∇v dΩ`` — returns ``(Ne, nPe, nPe)``.

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :pymeth:`FeArray.broadcast` (stride view, no copy).
    """
    mat_e_pg = groupElem.Get_DiffusePart_e_pg(matrixType)
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    Ne, nPg = dN_e_pg.shape[:2]
    coef = FeArray.broadcast(coef, Ne, nPg)
    return (coef * mat_e_pg @ dN_e_pg).integrate()


def UV(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    dof_n: int = 1,
    matrixType: MatrixType = MatrixType.mass,
) -> np.ndarray:
    """``∫_Ω coef · u · v dΩ`` — returns ``(Ne, nPe·dof_n, nPe·dof_n)``.

    ``dof_n=1`` for scalar fields (thermal, phase-field);
    ``dof_n=dim`` for vector fields (elastic dynamics).

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :pymeth:`FeArray.broadcast` (stride view, no copy).
    """
    mat_e_pg = groupElem.Get_ReactionPart_e_pg(matrixType, dof_n)
    Ne, nPg = mat_e_pg.shape[:2]
    coef = FeArray.broadcast(coef, Ne, nPg)
    return (coef * mat_e_pg).integrate()


def LinearizedElasticity(
    groupElem: "_GroupElem",
    C: FeArray.FeArrayALike,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω ε(u) : C : ε(v) dΩ`` — small-strain elastic stiffness.

    Returns ``(Ne, nPe·dim, nPe·dim)``.

    ``C`` is the Hooke tensor in Kelvin–Mandel notation — accepts
    ``(nstrain, nstrain)`` (homogeneous) or ``(Ne, nPg, nstrain, nstrain)``
    (heterogeneous); broadcast via :pymeth:`FeArray.broadcast`.
    """
    leftDispPart_e_pg = groupElem.Get_leftDispPart_e_pg(matrixType)
    B_e_pg = groupElem.Get_B_e_pg(matrixType)
    Ne, nPg = B_e_pg.shape[:2]
    C = FeArray.broadcast(C, Ne, nPg)
    return (leftDispPart_e_pg @ C @ B_e_pg).integrate()


def GradU_A_GradV(
    groupElem: "_GroupElem",
    A: FeArray.FeArrayALike,
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω coef · ∇u · A · ∇v dΩ`` — anisotropic diffusion. Returns ``(Ne, nPe, nPe)``.

    ``A`` is the diffusion tensor: ``(dim, dim)`` (homogeneous) or ``(Ne, nPg, dim, dim)`` (heterogeneous).
    ``coef`` is a scalar weight: scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``.
    Both broadcast via :pymeth:`FeArray.broadcast` (stride view, no copy).
    For the isotropic form ``∫ coef · ∇u · ∇v dΩ``, use :func:`GradUGradV`.
    """
    diffusePart_e_pg = groupElem.Get_DiffusePart_e_pg(matrixType)
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    Ne, nPg = dN_e_pg.shape[:2]
    A = FeArray.broadcast(A, Ne, nPg)
    coef = FeArray.broadcast(coef, Ne, nPg)
    return (coef * diffusePart_e_pg @ A @ dN_e_pg).integrate()
