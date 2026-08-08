# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Yield surfaces.

A surface says *when* the material flows and *which way*. It never says how much it has
hardened: the hardening force :math:`R` is handed to it by the free energy
(see :mod:`.IsotropicHardening`), so a surface and a hardening law compose freely.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np

from . import _kelvin
from ..FEM._linalg import FeArray, Norm, TensorProd
from ..Utilities import _types


class YieldSurface(NamedTuple):
    r"""Where the material yields, and which way it flows once it does.

    - ``f(sig, R) -> (Ne, nPg)`` — negative is elastic, positive is flowing. ``R`` is the
      isotropic hardening force, so ``f`` is written ``phi(sig) - sigma_y - R``.
    - ``N(sig, R) -> (Ne, nPg, 6)`` — the flow direction :math:`\dpartial{f}{\Sig}`.
    - ``scale`` — a representative stress, so the solver tolerance is dimensionless.
    - ``dNdSig(sig) -> (Ne, nPg, 6, 6)`` — :math:`\dpartial{N}{\Sig}`, which the local Jacobian
      needs. Written by hand here, as the derivatives of the hyperelastic potentials are.
    - ``P`` — the ``(6, 6)`` quadratic form when :math:`\phi^2 = \Sig : \Prm : \Sig`, else
      ``None``. Only that shape lets the local solve collapse to one scalar unknown, so it is
      what :class:`.Behaviour` dispatches on.

    To add a surface, write ``phi`` and differentiate it twice. Yield functions are linear in
    the stress invariants, so the chain rule is a line or two.
    """

    f: Callable[[FeArray.FeArrayALike, FeArray.FeArrayALike], FeArray.FeArrayALike]
    N: Callable[[FeArray.FeArrayALike, FeArray.FeArrayALike], FeArray.FeArrayALike]
    scale: float
    dNdSig: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    P: Optional[_types.FloatArray] = None


def Svm(sig_e_pg: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    r"""von Mises equivalent stress :math:`\sqrt{3/2}\,\|\dev\Sig\|`."""
    return np.sqrt(1.5) * Norm(_kelvin.Deviator(sig_e_pg), axis=-1)


def _Normal_J2(sig_e_pg: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    r""":math:`\dpartial{\sigma_{eq}}{\Sig} = \tfrac32 s/\sigma_{eq}`, deviatoric, ``N:N = 3/2``.

    Clamped at the apex, where the derivative of a norm does not exist.
    """
    s_e_pg = _kelvin.Deviator(sig_e_pg)
    svm_e_pg = np.sqrt(1.5) * Norm(s_e_pg, axis=-1)
    safe = np.where(svm_e_pg > 0, svm_e_pg, 1.0)
    return 1.5 * s_e_pg / safe


def _dNormal_J2(sig_e_pg: FeArray.FeArrayALike) -> FeArray.FeArrayALike:
    r""":math:`\dpartial{N}{\Sig} = \frac{3}{2\sigma_{eq}}\Irm_{dev} - \frac{1}{\sigma_{eq}} N\otimes N`.

    Drucker-Prager shares it exactly: its extra volumetric term is constant in ``sig``.
    """
    s_e_pg = _kelvin.Deviator(sig_e_pg)
    svm_e_pg = np.sqrt(1.5) * Norm(s_e_pg, axis=-1)
    safe = np.where(svm_e_pg > 0, svm_e_pg, 1.0)
    N_e_pg = 1.5 * s_e_pg / safe

    Ne, nPg = np.shape(svm_e_pg)
    dev = FeArray.broadcast(_kelvin.IDEV, Ne, nPg, tensor_ndim=2)
    NN_e_pg = TensorProd(N_e_pg, N_e_pg)
    return (1.5 * dev - NN_e_pg) / safe


def VonMises(sigma_y: float) -> YieldSurface:
    r"""J2 surface :math:`f = \sigma_{eq} - \sigma_y - R`.

    Parameters
    ----------
    sigma_y : float
        initial yield stress
    """
    assert sigma_y > 0, "sigma_y must be > 0"

    def f(sig_e_pg, R_e_pg):
        return Svm(sig_e_pg) - sigma_y - R_e_pg

    def N(sig_e_pg, R_e_pg):
        return _Normal_J2(sig_e_pg)

    return YieldSurface(f, N, sigma_y, _dNormal_J2, 1.5 * _kelvin.IDEV)


def Hill(
    sigma_y: float,
    F: float = 0.5,
    G: float = 0.5,
    H: float = 0.5,
    L: float = 1.5,
    M: float = 1.5,
    N: float = 1.5,
) -> YieldSurface:
    r"""Anisotropic surface :math:`f = \sqrt{\Sig : \Prm : \Sig} - \sigma_y - R`.

    Hill 1948, for rolled sheet and other textured metals. The defaults are the isotropic
    values, where it reduces exactly to von Mises.

    Its normal is not an eigendirection of ``C``, so the return path rotates it — which is why
    a frozen flow normal cannot integrate this surface, and why it needs the normal recomputed
    inside the local solve.

    Parameters
    ----------
    sigma_y : float
        reference yield stress
    F, G, H : float, optional
        normal anisotropy coefficients, by default 0.5 (isotropic)
    L, M, N : float, optional
        shear anisotropy coefficients for yz, xz, xy, by default 1.5 (isotropic)
    """
    assert sigma_y > 0, "sigma_y must be > 0"
    # Kelvin form: the sqrt(2) on the shear entries turns Hill's 2L*syz^2 into L*syz_kelvin^2
    P = np.array(
        [
            [G + H, -H, -G, 0.0, 0.0, 0.0],
            [-H, F + H, -F, 0.0, 0.0, 0.0],
            [-G, -F, F + G, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, L, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, M, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, N],
        ]
    )

    def __P_e_pg(Ne, nPg) -> FeArray:
        return FeArray.broadcast(P, Ne, nPg, tensor_ndim=2)

    def _parts(sig_e_pg):
        """``(P:sig, phi)`` with phi clamped away from the apex, where it is not differentiable."""
        Ne, nPg = np.shape(sig_e_pg)[:2]
        Ps_e_pg = __P_e_pg(Ne, nPg) @ sig_e_pg
        q_e_pg = np.maximum(sig_e_pg @ Ps_e_pg, 0.0)
        return Ps_e_pg, np.sqrt(q_e_pg)

    def f(sig_e_pg, R_e_pg):
        _, phi_e_pg = _parts(sig_e_pg)
        return phi_e_pg - sigma_y - R_e_pg

    def Normal(sig_e_pg, R_e_pg):
        Ps_e_pg, phi_e_pg = _parts(sig_e_pg)
        safe = np.where(phi_e_pg > 0, phi_e_pg, 1.0)
        return Ps_e_pg / safe

    def dNdSig(sig_e_pg):
        # dN/dsig = P/phi - N (x) N / phi
        Ne, nPg = np.shape(sig_e_pg)[:2]
        Ps_e_pg, phi_e_pg = _parts(sig_e_pg)
        safe = np.where(phi_e_pg > 0, phi_e_pg, 1.0)
        N_e_pg = Ps_e_pg / safe
        NN_e_pg = TensorProd(N_e_pg, N_e_pg)
        return (__P_e_pg(Ne, nPg) - NN_e_pg) / safe

    return YieldSurface(f, Normal, sigma_y, dNdSig, P)


def DruckerPrager(sigma_y: float, eta: float) -> YieldSurface:
    r"""Pressure-dependent surface :math:`f = \sigma_{eq} + \eta\,\tr\Sig - \sigma_y - R`.

    Associative, so the flow is dilatant.

    Parameters
    ----------
    sigma_y : float
        yield stress at zero hydrostatic stress
    eta : float
        pressure sensitivity
    """
    assert sigma_y > 0, "sigma_y must be > 0"

    def f(sig_e_pg, R_e_pg):
        return Svm(sig_e_pg) + eta * _kelvin.Trace(sig_e_pg) - sigma_y - R_e_pg

    def N(sig_e_pg, R_e_pg):
        return _Normal_J2(sig_e_pg) + eta * _kelvin.ONE

    return YieldSurface(f, N, sigma_y, _dNormal_J2)
