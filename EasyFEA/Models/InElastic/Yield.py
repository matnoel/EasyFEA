# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Yield surfaces ``f(sig, R)`` at one point, negative where elastic; the hardening force ``R`` is handed in, so any surface takes any hardening."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ._behavior import Trace, Von_Mises_stress

if TYPE_CHECKING:
    from jax import Array


class Surface(Protocol):
    """A yield function ``f(sig, R)`` on a (6,) Kelvin stress, with the stress that scales it."""

    sigma_y: float

    def __call__(self, sig: "Array", R: "Array") -> "Array": ...


@dataclass(frozen=True)
class VonMises:
    r""":math:`f = \sigma_{eq} - \sigma_y - R`."""

    sigma_y: float

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        return Von_Mises_stress(sig) - self.sigma_y - R


@dataclass(frozen=True)
class Hill:
    r""":math:`f = \sqrt{\Sig : \Prm : \Sig} - \sigma_y - R`, Hill 1948; the defaults are von Mises."""

    sigma_y: float
    F: float = 0.5
    G: float = 0.5
    H: float = 0.5
    L: float = 1.5
    M: float = 1.5
    N: float = 1.5

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    @property
    def P(self) -> np.ndarray:
        """(6, 6) Kelvin form; the sqrt(2) on the shear entries turns Hill's 2L syz^2 into L syz_kelvin^2."""
        F, G, H = self.F, self.G, self.H
        P = np.diag([0.0, 0.0, 0.0, self.L, self.M, self.N])
        P[:3, :3] = [[G + H, -H, -G], [-H, F + H, -F], [-G, -F, F + G]]
        return P

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        import jax.numpy as jnp

        # finite at sig = 0, so that it stays differentiable
        return jnp.sqrt(sig @ self.P @ sig + 1e-300) - self.sigma_y - R


@dataclass(frozen=True)
class DruckerPrager:
    r""":math:`f = \sigma_{eq} + \eta\,\tr\Sig - \sigma_y - R`; associated, so the flow is dilatant."""

    sigma_y: float
    eta: float

    def __post_init__(self):
        assert self.sigma_y > 0, "sigma_y must be > 0"

    def __call__(self, sig: "Array", R: "Array") -> "Array":
        return Von_Mises_stress(sig) + self.eta * Trace(sig) - self.sigma_y - R
