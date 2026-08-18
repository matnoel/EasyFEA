# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Viscoplastic flow rates.

Plasticity lands exactly on the surface, so :math:`f \leq 0` always holds. Viscoplasticity is
allowed to stay outside it, and that overstress is precisely what drives the flow, at a finite
speed:

.. math::
    \text{plasticity} \quad & f = 0 \\
    \text{viscoplasticity} \quad & f = \phi^{-1}(\dot\gamma) \geq 0

Creep and relaxation are the same law seen under a held stress and a held strain.
"""

from typing import Callable, NamedTuple

import numpy as np

from ...FEM._linalg import FeArray

_TINY = 1e-300
"""Floor on the rate, so the inverse and its slope stay finite at zero flow."""


class RateLaw(NamedTuple):
    r"""How fast the material flows for a given overstress.

    - ``rate(f) -> (Ne, nPg)`` — the plastic rate an overstress ``f`` drives. Only used for the
      first Newton guess, which matters because ``dinverse`` is unbounded at zero flow.
    - ``inverse(gdot) -> (Ne, nPg)`` — the overstress needed to sustain ``gdot``.
    - ``dinverse(gdot) -> (Ne, nPg)`` — its slope, which the local Jacobian needs.

    The residual is written in **stress** units, ``f - inverse(dGamma/dt)``, not in strain units.
    Both have the same roots, but near the rate-independent limit the strain form evaluates the
    rate law on an ``f`` built from two large cancelling terms, and a 1e-16 error there is
    multiplied by a fluidity that can be 1e12.
    """

    rate: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    inverse: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]
    dinverse: Callable[[FeArray.FeArrayALike], FeArray.FeArrayALike]


def Norton(A: float, n: float = 1.0, sigma_0: float = 1.0) -> RateLaw:
    r"""Norton creep :math:`\dot\gamma = A\,(f/\sigma_0)^n`.

    Parameters
    ----------
    A : float
        fluidity; large ``A`` approaches rate-independent plasticity
    n : float, optional
        stress exponent, by default 1.0
    sigma_0 : float, optional
        reference stress, by default 1.0
    """
    assert A > 0 and n > 0 and sigma_0 > 0, "need A > 0, n > 0, sigma_0 > 0"

    def rate(f):
        return A * (np.maximum(f, 0.0) / sigma_0) ** n

    def inverse(g):
        return sigma_0 * (np.maximum(g, _TINY) / A) ** (1 / n)

    def dinverse(g):
        return sigma_0 / (n * A) * (np.maximum(g, _TINY) / A) ** (1 / n - 1)

    return RateLaw(rate, inverse, dinverse)


def Perzyna(eta: float, n: float = 1.0, sigma_0: float = 1.0) -> RateLaw:
    r"""Perzyna viscoplasticity :math:`\dot\gamma = \frac{1}{\eta}(f/\sigma_0)^n`.

    Algebraically Norton with ``A = 1/eta``; kept separate because the viscosity is the
    parameter that gets fitted.

    Parameters
    ----------
    eta : float
        viscosity; small ``eta`` approaches rate-independent plasticity
    n : float, optional
        stress exponent, by default 1.0
    sigma_0 : float, optional
        reference stress, by default 1.0
    """
    assert eta > 0, "eta must be > 0"
    return Norton(1.0 / eta, n, sigma_0)
