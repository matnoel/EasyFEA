# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from .._utils import _Interval
from ._newmark import Newmark


class HHT(Newmark):
    r"""Hilber–Hughes–Taylor-:math:`\alpha` method: `Newmark`'s update, every field evaluated at :math:`(1-\alpha) \, \cdot^{n+1} + \alpha \, \cdot^n`, :math:`\alpha \in [0, 1[` damping the high frequencies (0 is Newmark, 1/2 with the default β, γ the midpoint rule)."""

    alpha: float = _Interval(0.0, 1.0)

    def __init__(
        self, dt: float, alpha: float = 1 / 2, beta: float = 1 / 4, gamma: float = 1 / 2
    ):
        super().__init__(dt, beta, gamma)
        self.alpha = alpha

    def Evaluate(self, x, u_n, v_n, a_n):
        alpha = self.alpha
        u_np1, v_np1, a_np1 = self.Evaluate_np1(x, u_n, v_n, a_n)
        u_t = (1 - alpha) * u_np1 + alpha * u_n
        v_t = (1 - alpha) * v_np1 + alpha * v_n
        a_t = (1 - alpha) * a_np1 + alpha * a_n
        return u_t, v_t, a_t
