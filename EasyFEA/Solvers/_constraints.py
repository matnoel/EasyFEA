# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Constraints on the unknowns of `A x = b`: known dofs and Lagrange constraints."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


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


class LagrangeConstraint(NamedTuple):
    """A constraint `coefs · x[dofs] = value`."""

    dofs: _types.IntArray
    coefs: _types.FloatArray
    value: float


@dataclass(frozen=True)
class Constraints:
    """The `knownDofs` (indices) at their `knownValues`, the Lagrange constraints and the `resolution`, None = r2 with Lagrange constraints, r1 without."""

    knownDofs: _types.IntArray
    knownValues: _types.FloatArray
    list_lagrangeConstraint: tuple[LagrangeConstraint, ...] = ()
    resolution: ResolType | None = None

    def __post_init__(self) -> None:
        assert self.knownValues.size == self.knownDofs.size, "one value per known dof"
        if self.resolution is None:
            resolution = ResolType.r2 if self.list_lagrangeConstraint else ResolType.r1
            object.__setattr__(self, "resolution", resolution)
        elif self.list_lagrangeConstraint and self.resolution != ResolType.r2:
            raise NotImplementedError(
                f"{self.resolution} does not support Lagrange constraints."
            )
