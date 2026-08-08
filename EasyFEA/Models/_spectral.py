# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Return mapping for a quadratic yield surface, in the material eigenspace.

For :math:`\phi^2 = \Sig : \Prm : \Sig` the flow is :math:`\dEps^p = \dGam\,\Prm\Sig/\phi`, so
with :math:`\theta = \dGam/\phi` the update is *linear* in the stress:

.. math::
    (\Irm + \theta\,\Crm\Prm)\,\Sig = \Sig_{tr}

Diagonalising :math:`\Crm^{1/2}\Prm\Crm^{1/2} = \Qrm\Lam\Qrm^T` — **once per material**, not per
Gauss point — makes that inverse diagonal, and the consistency condition becomes an explicit
scalar function of :math:`\theta`:

.. math::
    \phi(\theta)^2 = \sum_i \frac{\lambda_i y_i^2}{(1 + \theta\lambda_i)^2}
    \qquad y = \Qrm^T \Crm^{-1/2} \Sig_{tr}

monotone decreasing, so one safeguarded Newton solves it. Isotropic elasticity with von Mises
gives a single repeated :math:`\lambda`, the equation becomes linear, and this reduces exactly to
the classical radial return -- there is no separate isotropic path to keep in step.

Simo & Taylor (1986) use the same structure for plane-stress J2; the anisotropic form is the
"generalized radial return in material eigenspace".
"""

from typing import NamedTuple

import numpy as np

from ..FEM._linalg import FeArray, TensorProd
from ..Utilities import _types


class Return(NamedTuple):
    """What the scalar solve learned, including everything :func:`Tangent` needs."""

    sig: FeArray
    dGamma: FeArray
    phi: FeArray
    theta: FeArray
    y: FeArray
    d: FeArray
    slope: FeArray
    drdtheta: FeArray
    active: FeArray


class Eigenspace(NamedTuple):
    r"""The per-material decomposition the return runs in.

    - ``T`` / ``Ti`` — :math:`\Crm^{1/2}\Qrm` and its inverse, one ``(6, 6)`` map each way.
    - ``lam`` — the eigenvalues :math:`\lambda_i`.
    - ``Cinv`` — recovers the plastic strain as :math:`\Eps - \Crm^{-1}\Sig`.
    """

    T: _types.FloatArray
    Ti: _types.FloatArray
    lam: _types.FloatArray
    Cinv: _types.FloatArray


def Build(C: _types.FloatArray, P: _types.FloatArray) -> Eigenspace:
    """Diagonalises ``C^1/2 P C^1/2``. Called once, when the behaviour is built."""
    lamC, Qc = np.linalg.eigh(C)
    assert lamC.min() > 0, "the elastic stiffness must be positive definite"
    Ch = Qc @ np.diag(np.sqrt(lamC)) @ Qc.T
    Chi = Qc @ np.diag(1.0 / np.sqrt(lamC)) @ Qc.T

    lam, Q = np.linalg.eigh(Ch @ P @ Ch)
    lam = np.maximum(lam, 0.0)  # P is positive semi-definite; clean up round-off
    return Eigenspace(Ch @ Q, Q.T @ Chi, lam, Chi @ Chi)


def _Field(mat: _types.FloatArray, ref: FeArray) -> FeArray.FeArrayALike:
    """The constant ``(6, 6)`` map held at every Gauss point of ``ref``."""
    return FeArray.broadcast(mat, *ref.shape[:2], tensor_ndim=2)


def _Phi(y_e_pg: FeArray, lam: _types.FloatArray, theta_e_pg: FeArray):
    r"""``(phi, dphi/dtheta)`` from the diagonal form, both at once.

    ``theta`` is a scalar field and ``lam`` a constant 6-vector, so the product is the
    ``(Ne, nPg, 6)`` field of :math:`1 + \theta\lambda_i` without any reshaping.
    """
    d_e_pg = 1.0 / (1.0 + theta_e_pg * lam)
    w_e_pg = (lam * y_e_pg**2) * d_e_pg**2
    phi_e_pg = np.sqrt(np.maximum(w_e_pg.sum(axis=-1), 0.0))
    safe_e_pg = np.where(phi_e_pg > 0, phi_e_pg, 1.0)
    dphi_e_pg = -(w_e_pg * lam * d_e_pg).sum(axis=-1) / safe_e_pg
    return phi_e_pg, dphi_e_pg


def Solve(
    eigen: Eigenspace,
    sigTr_e_pg: FeArray,
    pOld_e_pg: FeArray,
    hardening,
    sigma_y: float,
    rate=None,
    dt: float = 0.0,
    tol: float = 1e-12,
    maxIter: int = 50,
) -> Return:
    r"""Newton on the one scalar :math:`\theta`, over every Gauss point at once."""
    lam = eigen.lam
    y_e_pg = _Field(eigen.Ti, sigTr_e_pg) @ sigTr_e_pg

    theta_e_pg = FeArray.zeros(*sigTr_e_pg.shape[:2])
    phi_e_pg, _ = _Phi(y_e_pg, lam, theta_e_pg)
    # a point yields when the trial state is already outside the surface
    active_e_pg = phi_e_pg - sigma_y - hardening.R(pOld_e_pg) > 0.0

    slope_e_pg = FeArray.zeros(*theta_e_pg.shape)
    for _ in range(maxIter):
        phi_e_pg, dphi_e_pg = _Phi(y_e_pg, lam, theta_e_pg)
        dG_e_pg = theta_e_pg * phi_e_pg

        r_e_pg = phi_e_pg - sigma_y - hardening.R(pOld_e_pg + dG_e_pg)
        slope_e_pg = hardening.dR(pOld_e_pg + dG_e_pg)
        if rate is not None:
            r_e_pg = r_e_pg - rate.inverse(dG_e_pg / dt)
            slope_e_pg = slope_e_pg + rate.dinverse(dG_e_pg / dt) / dt
        # dGamma = theta phi, so hardening and the rate law both feel theta through it
        ddG_e_pg = phi_e_pg + theta_e_pg * dphi_e_pg
        drdtheta_e_pg = dphi_e_pg - slope_e_pg * ddG_e_pg

        if np.max(np.where(active_e_pg, np.abs(r_e_pg), 0.0)) < tol * sigma_y:
            break
        step_e_pg = np.where(active_e_pg, r_e_pg / drdtheta_e_pg, 0.0)
        theta_e_pg = np.maximum(theta_e_pg - step_e_pg, 0.0)

    phi_e_pg, dphi_e_pg = _Phi(y_e_pg, lam, theta_e_pg)
    dG_e_pg = theta_e_pg * phi_e_pg
    ddG_e_pg = phi_e_pg + theta_e_pg * dphi_e_pg
    drdtheta_e_pg = dphi_e_pg - slope_e_pg * ddG_e_pg

    d_e_pg = 1.0 / (1.0 + theta_e_pg * lam)
    sig_e_pg = _Field(eigen.T, y_e_pg) @ (y_e_pg * d_e_pg)

    return Return(
        sig_e_pg, dG_e_pg, phi_e_pg, theta_e_pg, y_e_pg, d_e_pg,
        slope_e_pg, drdtheta_e_pg, active_e_pg,
    )


def Tangent(eigen: Eigenspace, res: Return, C_e_pg: FeArray) -> FeArray:
    r"""``dsigma/deps`` by exact linearisation of the return, as Simo & Taylor do it.

    :math:`\Sig = \Trm\Drm\Trm^{-1}\Sig_{tr}` with :math:`\Drm` diagonal, so differentiating
    gives the elastic part plus one rank-one term through :math:`\theta`:

    .. math::
        \dpartial{\Sig}{\Eps} = \left[\Trm\Drm\Trm^{-1}
            + \dpartial{\Sig}{\theta}\otimes\dpartial{\theta}{\Sig_{tr}}\right]\Crm
    """
    lam, d_e_pg, y_e_pg = eigen.lam, res.d, res.y

    # T diag(d) Ti C, with the constant maps held at every point rather than rebuilt
    elastic_e_pg = (_Field(eigen.T, y_e_pg) * d_e_pg[..., None, :]) @ (
        _Field(eigen.Ti, y_e_pg) @ C_e_pg
    )

    # dsigma/dtheta, and dtheta/dsigma_tr through the consistency condition
    w_e_pg = (lam * y_e_pg) * d_e_pg**2
    dsig_e_pg = _Field(eigen.T, y_e_pg) @ -w_e_pg
    safe_e_pg = np.where(res.phi > 0, res.phi, 1.0)
    n_e_pg = (_Field(eigen.Ti.T, y_e_pg) @ w_e_pg) / safe_e_pg
    dtheta_e_pg = -(1.0 - res.theta * res.slope) / res.drdtheta

    C_alg = elastic_e_pg + TensorProd(dtheta_e_pg * dsig_e_pg, C_e_pg @ n_e_pg)
    # a point that never yielded keeps the elastic stiffness, whatever the algebra says
    return np.where(res.active[..., None, None], C_alg, C_e_pg)
