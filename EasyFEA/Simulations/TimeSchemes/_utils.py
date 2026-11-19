# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

r"""Time schemes of :math:`\Krm \, \urm_t + \Crm \, \vrm_t + \Mrm \, \arm_t = \Frm`: how the step's unknown sets the evaluation point and the next state."""

from abc import ABC, abstractmethod
from typing import Any

from ...Utilities import _params

_Field = Any
"""A dof vector, or a scalar: a scheme is affine and evaluated on both."""


class _TimeScheme(ABC):
    r"""A time scheme: the evaluation point :math:`(\urm_t, \vrm_t, \arm_t)` and the next state, both affine in the step's unknown `x`; stateless beyond its parameters.

    A new scheme subclasses this (or :class:`~EasyFEA.TimeSchemes.Hyperbolic._Hyperbolic`) and implements :meth:`Evaluate` and :meth:`Evaluate_np1`; A's coefficients follow from :meth:`Evaluate`."""

    unknown: str = "u"
    """The role the step solves for, one of :attr:`roles`."""

    roles: tuple[str, ...] = ("u",)
    """The roles the scheme integrates, of u, v and a."""

    @abstractmethod
    def Evaluate(
        self, x: _Field, u_n: _Field, v_n: _Field, a_n: _Field
    ) -> tuple[_Field, _Field | None, _Field | None]:
        r""":math:`(\urm_t, \vrm_t, \arm_t)` at the step's unknown `x` from the state at n, None for a role not integrated."""

    @abstractmethod
    def Evaluate_np1(
        self, x: _Field, u_n: _Field, v_n: _Field, a_n: _Field
    ) -> tuple[_Field, _Field | None, _Field | None]:
        r""":math:`(\urm, \vrm, \arm)^{n+1}` at the step's unknown `x` from the state at n, None for a role not integrated."""

    @property
    def coefs(self) -> tuple[float, float, float]:
        r"""``(coefK, coefC, coefM)`` of ``A = coefK·K + coefC·C + coefM·M``: :meth:`Evaluate` at ``x = 1`` from rest."""
        return tuple(  # type: ignore [return-value]
            0.0 if field is None else float(field)
            for field in self.Evaluate(1.0, 0.0, 0.0, 0.0)
        )

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)


class Elliptic(_TimeScheme):
    r""":math:`\Krm \, \urm = \Frm`, no time."""

    def Evaluate(self, x, u_n, v_n, a_n):
        return x, None, None

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        return x, None, None


class Parabolic(_TimeScheme):
    r""":math:`\Krm \, \urm^{n+1} + \Crm \, \vrm^{n+1} = \Frm^{n+1}` with :math:`\urm^{n+1} = \urm^n + \dt \, \vrm^{n+\alpha}`: alpha 1 backward Euler, 1/2 Crank-Nicolson."""

    roles = ("u", "v")

    dt: float = _params.StrictlyPositiveScalarParameter()
    """time increment"""

    alpha: float = _params.IntervalocScalarParameter(0.0, 1.0)
    """weight of the end of the step in the speed"""

    def __init__(self, dt: float, alpha: float = 1 / 2):
        self.dt = dt
        self.alpha = alpha

    def Evaluate(self, x, u_n, v_n, a_n):
        return self.Evaluate_np1(x, u_n, v_n, a_n)

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        # Hughes 1987 Chapter 8
        dt, alpha = self.dt, self.alpha
        ut_np1 = u_n + (1 - alpha) * dt * v_n
        v_np1 = (x - ut_np1) / (alpha * dt)
        return x, v_np1, None
