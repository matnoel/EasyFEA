# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Isotropic hardening ``R(p)`` at one point, with ``R(0) = 0``: the initial yield stress belongs to the surface."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from jax import Array

Hardening = Callable[["Array"], "Array"]
"""An isotropic hardening ``R(p)``."""


@dataclass(frozen=True)
class Perfect:
    """:math:`R = 0`."""

    def __call__(self, p: "Array") -> "Array":
        return 0.0 * p


@dataclass(frozen=True)
class Linear:
    """:math:`R = H p`."""

    H: float

    def __post_init__(self):
        assert self.H >= 0, "H must be >= 0"

    def __call__(self, p: "Array") -> "Array":
        return self.H * p


@dataclass(frozen=True)
class Voce:
    """:math:`R = Q (1 - e^{-b p})`, saturating at ``Q``."""

    Q: float
    b: float

    def __post_init__(self):
        assert self.Q >= 0 and self.b > 0, "Q must be >= 0 and b > 0"

    def __call__(self, p: "Array") -> "Array":
        import jax.numpy as jnp

        return self.Q * (1 - jnp.exp(-self.b * p))


@dataclass(frozen=True)
class Swift:
    r""":math:`R = K(\varepsilon_0 + p)^n - K\varepsilon_0^n`; ``eps0`` keeps the slope finite at the origin."""

    K: float
    n: float
    eps0: float = 1e-4

    def __post_init__(self):
        assert (
            self.K > 0 and 0 < self.n < 1 and self.eps0 > 0
        ), "need K > 0, 0 < n < 1, eps0 > 0"

    def __call__(self, p: "Array") -> "Array":
        return self.K * ((self.eps0 + p) ** self.n - self.eps0**self.n)
