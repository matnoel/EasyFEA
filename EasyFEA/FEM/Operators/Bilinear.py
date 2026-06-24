# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import TYPE_CHECKING, Union

import numpy as np

from .._linalg import FeArray, TensorProd
from .._utils import MatrixType
from ...Utilities import _types

from ..Elems._beam import _EulerBernoulli, _Timoshenko  # noqa: F401

if TYPE_CHECKING:
    from .._group_elem import _GroupElem
    from ...Models.Beam._beam import BeamStructure


def einsum(*args):
    return np.asarray(np.einsum(*args, optimize=True))


def GradUGradV(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω coef · ∇u · ∇v dΩ`` — returns ``(Ne, nPe, nPe)``.

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :meth:`FeArray.broadcast` (stride view, no copy).
    """
    mat_e_pg = groupElem.Get_DiffusePart_e_pg(matrixType)
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    Ne, nPg = dN_e_pg.shape[:2]
    coef = FeArray.broadcast(coef, Ne, nPg)
    return einsum("epij,epjk->eik", coef * mat_e_pg, dN_e_pg)


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
    broadcast via :meth:`FeArray.broadcast` (stride view, no copy).
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

    ``C`` is the Hooke tensor in Kelvin–Mandel notation with trailing two
    dims ``(nstrain, nstrain)``; the leading dims may be empty (homogeneous),
    ``(Ne,)`` (per-element), or ``(Ne, nPg)`` (per-element per-gauss-point).
    """
    leftDispPart_e_pg = groupElem.Get_leftDispPart_e_pg(matrixType)
    B_e_pg = groupElem.Get_B_e_pg(matrixType)
    Ne, nPg = B_e_pg.shape[:2]
    C = FeArray.broadcast(C, Ne, nPg, tensor_ndim=2)
    return einsum("epij,epjk->eik", leftDispPart_e_pg @ C, B_e_pg)


def MassAlongNormal(
    groupElem: "_GroupElem",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.mass,
) -> np.ndarray:
    r"""``∫_Γ coef · (u · n̂)(v · n̂) dΓ`` — mass projected onto the surface normal.

    Returns ``(Ne, nPe·3, nPe·3)`` in ``(xi, yi, zi, ..., xn, yn, zn)`` order.

    The per-Gauss-point integrand is ``Nᵀ · (n̂ ⊗ n̂) · N`` where ``n̂`` is
    the reference-configuration unit normal supplied by
    :meth:`_GroupElem.Get_normals_e_pg` and ``N`` is the block-diagonal
    shape-function matrix from :meth:`_GroupElem.Get_N_pg_rep`. Typically
    used to penalise / enforce the normal component on a surface (Robin-style
    ``k · u·n̂ · v·n̂`` boundary conditions).

    Restricted to a 2D surface group in a 3D mesh (``groupElem.dim == 2``,
    ``groupElem.inDim == 3``).

    ``coef`` may be scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``;
    broadcast via :meth:`FeArray.broadcast` (stride view, no copy).
    """
    assert groupElem.dim in [1, 2]

    dim = 3
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)  # (Ne, nPg)
    N_pg = FeArray.asfearray(
        groupElem.Get_N_pg_rep(matrixType, dim)[np.newaxis]
    )  # (1, nPg, dim, dim·nPe)
    n_e_pg = groupElem.Get_normals_e_pg(matrixType)  # (Ne, nPg, dim) unit normal
    nn_e_pg = TensorProd(n_e_pg, n_e_pg)  # (Ne, nPg, dim, dim)

    Ne, nPg = wJ_e_pg.shape
    coef = FeArray.broadcast(coef, Ne, nPg)
    return einsum("ep,opji,epjk,opkl->eil", coef * wJ_e_pg, N_pg, nn_e_pg, N_pg)


def BeamBending(
    groupElem: "_GroupElem",
    beamStructure: "BeamStructure",
) -> np.ndarray:
    """``∫_e Bᵀ · D_bending · B dx`` — axial + bending (+ torsion) stiffness.

    Returns ``(Ne, nPe·dof_n, nPe·dof_n)`` with ``dof_n = beamStructure.dof_n``.

    Integrated at the full Gauss scheme ``MatrixType.beam``. For Timoshenko
    elements in 2D / 3D the transverse-shear rows of ``D`` are zeroed so this
    term carries only the axial / bending / torsion energy; combine with
    :func:`BeamShear` to get the full Timoshenko stiffness, or call
    :func:`BeamStiffness` directly.
    """
    matrixType = MatrixType.beam
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = groupElem.Get_beam_B_e_pg(beamStructure)
    D_e_pg = beamStructure.Calc_D_e_pg(groupElem, matrixType)

    dim = beamStructure.dim
    if dim != 1 and isinstance(groupElem, _Timoshenko):
        shear_rows = {2: (2,), 3: (4, 5)}[dim]
        D_e_pg = D_e_pg.copy()
        for r in shear_rows:
            D_e_pg[:, :, r, r] = 0.0

    return einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, D_e_pg, B_e_pg)


def BeamShear(
    groupElem: "_GroupElem",
    beamStructure: "BeamStructure",
) -> np.ndarray:
    """``∫_e Bᵀ · D_shear · B dx`` — transverse-shear stiffness, SRI.

    Returns ``(Ne, nPe·dof_n, nPe·dof_n)`` with ``dof_n = beamStructure.dof_n``.

    Active only for ``_Timoshenko`` groupElems in 2D / 3D — returns a zero
    matrix otherwise. Integrated at the reduced Gauss scheme
    ``MatrixType.beam_shear`` (one fewer point than ``MatrixType.beam``) with
    the non-shear rows of ``D`` zeroed. This is the selective-reduced-
    integration cure for Timoshenko shear locking.
    """
    dof_n = beamStructure.dof_n
    Ne = groupElem.Ne
    nPe = groupElem.nPe
    dim = beamStructure.dim

    if dim == 1 or not isinstance(groupElem, _Timoshenko):
        return np.zeros((Ne, nPe * dof_n, nPe * dof_n))

    matrixType = MatrixType.beam_shear
    shear_rows = {2: (2,), 3: (4, 5)}[dim]
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = groupElem.Get_beam_B_e_pg(beamStructure, matrixType)
    D_e_pg = beamStructure.Calc_D_e_pg(groupElem, matrixType).copy()
    for r in range(D_e_pg.shape[-1]):
        if r not in shear_rows:
            D_e_pg[:, :, r, r] = 0.0
    return einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, D_e_pg, B_e_pg)


def BeamStiffness(
    groupElem: "_GroupElem",
    beamStructure: "BeamStructure",
) -> np.ndarray:
    """Full beam stiffness ``K_e = BeamBending + BeamShear``.

    Returns ``(Ne, nPe·dof_n, nPe·dof_n)`` with ``dof_n = beamStructure.dof_n``.

    Euler-Bernoulli (or 1-D Timoshenko): reduces to ``∫ Bᵀ D B dx`` at
    ``MatrixType.beam``. Timoshenko in 2D / 3D: splits into bending (full
    Gauss) + shear (reduced Gauss) via :func:`BeamBending` and
    :func:`BeamShear` to defeat shear locking.
    """
    dim = beamStructure.dim
    if dim != 1 and isinstance(groupElem, _Timoshenko):
        return BeamBending(groupElem, beamStructure) + BeamShear(
            groupElem, beamStructure
        )

    matrixType = MatrixType.beam
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    B_e_pg = groupElem.Get_beam_B_e_pg(beamStructure)
    D_e_pg = beamStructure.Calc_D_e_pg(groupElem, matrixType)
    return einsum("ep,epji,epjk,epkl->eil", wJ_e_pg, B_e_pg, D_e_pg, B_e_pg)


def BeamMass(
    groupElem: "_GroupElem",
    beamStructure: "BeamStructure",
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
) -> np.ndarray:
    """``∫_e coef · Nᵀ · M · N dx`` — beam consistent-mass matrix.

    Returns ``(Ne, nPe·dof_n, nPe·dof_n)`` with ``dof_n = beamStructure.dof_n``.

    Integrated at ``MatrixType.beam``. ``coef`` is typically the density
    ``rho`` of the simulation; may be scalar, ``(Ne,)``, ``(nPg,)``, or
    ``(Ne, nPg)`` — broadcast via :meth:`FeArray.broadcast`.
    """
    matrixType = MatrixType.beam
    wJ_e_pg = groupElem.Get_weightedJacobian_e_pg(matrixType)
    N_e_pg = groupElem.Get_beam_N_e_pg(beamStructure)
    M_e_pg = beamStructure.Calc_M_e_pg(groupElem)
    Ne, nPg = wJ_e_pg.shape
    coef = FeArray.broadcast(coef, Ne, nPg)
    return einsum("ep,epji,epjk,epkl->eil", coef * wJ_e_pg, N_e_pg, M_e_pg, N_e_pg)


def GradU_A_GradV(
    groupElem: "_GroupElem",
    A: FeArray.FeArrayALike,
    coef: Union[_types.Number, FeArray.FeArrayALike] = 1.0,
    matrixType: MatrixType = MatrixType.rigi,
) -> np.ndarray:
    """``∫_Ω coef · ∇u · A · ∇v dΩ`` — anisotropic diffusion. Returns ``(Ne, nPe, nPe)``.

    ``A`` is the diffusion tensor with trailing two dims ``(dim, dim)``; the
    leading dims may be empty, ``(Ne,)``, or ``(Ne, nPg)``. ``coef`` is a
    scalar weight: scalar, ``(Ne,)``, ``(nPg,)``, or ``(Ne, nPg)``. Both
    broadcast via :meth:`FeArray.broadcast`. For the isotropic form
    ``∫ coef · ∇u · ∇v dΩ``, use :func:`GradUGradV`.
    """
    diffusePart_e_pg = groupElem.Get_DiffusePart_e_pg(matrixType)
    dN_e_pg = groupElem.Get_dN_e_pg(matrixType)
    Ne, nPg = dN_e_pg.shape[:2]
    A = FeArray.broadcast(A, Ne, nPg, tensor_ndim=2)
    coef = FeArray.broadcast(coef, Ne, nPg)
    # return (coef * diffusePart_e_pg @ A @ dN_e_pg).integrate()
    return einsum("epij,epjk->eik", coef * diffusePart_e_pg, A @ dN_e_pg)
