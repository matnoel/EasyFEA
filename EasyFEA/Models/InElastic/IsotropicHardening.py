# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Isotropic hardening, as stored energy.

Hardening is part of the free energy, not part of the yield surface:

.. math::
    \psi = \psi_{elas}(\Eps^e) + \psi_h(p)
    \qquad
    R = \dpartial{\psi_h}{p}

so the surface sees only :math:`R` and any hardening law composes with any surface. Writing
``H`` into the surface instead would need one factory per (surface, hardening) pair.
"""

from typing import Callable, NamedTuple

import numpy as np

from ..FEM._linalg import FeArray


class IsotropicHardening(NamedTuple):
    r"""Stored energy of the accumulated plastic strain :math:`p`.

    - ``psi(p) -> (Ne, nPg)`` — the stored energy.
    - ``R(p) -> (Ne, nPg)`` — :math:`\dpartial{\psi_h}{p}`, the force the surface sees.
    - ``dR(p) -> (Ne, nPg)`` — its slope, which the local Jacobian needs.

    Kinematic hardening carries a tensor back-stress rather than a scalar, so it will widen this
    tuple rather than reuse it.
    """

    psi: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    R: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    dR: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]


def Linear(H: float) -> IsotropicHardening:
    r""":math:`\psi_h = \tfrac12 H p^2`, so :math:`R = H p`.

    Parameters
    ----------
    H : float
        hardening modulus (H = 0 is perfect plasticity)
    """
    assert H >= 0, "H must be >= 0"
    return IsotropicHardening(
        lambda p: 0.5 * H * p**2,
        lambda p: H * p,
        lambda p: H * (p * 0 + 1.0),
    )


def Voce(Q: float, b: float) -> IsotropicHardening:
    r"""Saturating hardening :math:`R = Q(1 - e^{-b p})`.

    The one real metal fits use: ``R`` tends to ``Q`` once ``p >> 1/b``.

    Parameters
    ----------
    Q : float
        saturation stress
    b : float
        saturation rate
    """
    assert Q >= 0 and b > 0, "Q must be >= 0 and b > 0"
    return IsotropicHardening(
        lambda p: Q * (p + np.exp(-b * p) / b - 1 / b),
        lambda p: Q * (1 - np.exp(-b * p)),
        lambda p: Q * b * np.exp(-b * p),
    )


def Swift(K: float, n: float, eps0: float = 1e-4) -> IsotropicHardening:
    r"""Power-law hardening :math:`R = K(\varepsilon_0 + p)^n - K\varepsilon_0^n`.

    Offset so that ``R(0) = 0``, since the initial yield stress belongs to the surface.

    Parameters
    ----------
    K : float
        strength coefficient
    n : float
        hardening exponent, 0 < n < 1
    eps0 : float, optional
        pre-strain keeping the slope finite at the origin, by default 1e-4
    """
    assert K > 0 and 0 < n < 1 and eps0 > 0, "need K > 0, 0 < n < 1, eps0 > 0"
    return IsotropicHardening(
        lambda p: K * ((eps0 + p) ** (n + 1) - eps0 ** (n + 1)) / (n + 1)
        - K * eps0**n * p,
        lambda p: K * ((eps0 + p) ** n - eps0**n),
        lambda p: K * n * (eps0 + p) ** (n - 1),
    )
