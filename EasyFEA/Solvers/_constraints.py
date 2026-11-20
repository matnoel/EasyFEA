# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Constraints on the unknowns of `A x = b`: known dofs and Lagrange conditions."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from ..Utilities import _types


class ResolType(str, Enum):
    """Resolution type."""

    r1 = "r1"
    r""":math:`\mathrm{x}_\irm = {\Arm_{\irm\irm}}^{-1} \cdot (b_\irm - \Arm_{\irm\crm} * \mathrm{x}_\crm)`,
    where :math:`\irm` and :math:`\crm` are unknown and known degrees of freedom."""
    r2 = "r2"
    """Lagrange multipliers"""
    r3 = "r3"
    """Penality"""

    def __str__(self) -> str:
        return self.name


class Lagrange(NamedTuple):
    """A condition `coefs · x[dofs] = value`."""

    dofs: _types.IntArray
    coefs: _types.FloatArray
    value: float


@dataclass(frozen=True)
class Constraints:
    """The dofs `known` (a mask), their `target` values, the `lagrange` conditions and the `resolution`, None = r2 with Lagrange conditions, r1 without."""

    known: _types.BoolArray
    target: _types.FloatArray
    lagrange: tuple[Lagrange, ...] = ()
    resolution: ResolType | None = None

    def __post_init__(self) -> None:
        assert self.target.size == np.count_nonzero(self.known), (
            "one target per known dof"
        )
        if self.resolution is None:
            resolution = ResolType.r2 if self.lagrange else ResolType.r1
            object.__setattr__(self, "resolution", resolution)
        elif self.lagrange and self.resolution != ResolType.r2:
            raise NotImplementedError(
                f"{self.resolution} does not support Lagrange conditions."
            )
