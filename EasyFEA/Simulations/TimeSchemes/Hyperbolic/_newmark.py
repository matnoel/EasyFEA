# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from .._utils import _Interval
from ._hyperbolic import _Hyperbolic


class Newmark(_Hyperbolic):
    r"""Newmark method, by default the average acceleration (:math:`\beta=1/4`, :math:`\gamma=1/2`).

    Predictor :math:`\tilde{\urm} = \urm^n + \dt\vrm^n + \dt^2/2\,(1-2\beta)\arm^n`, update :math:`\arm^{n+1} = (\urm^{n+1} - \tilde{\urm})/(\beta\dt^2)`, :math:`\vrm^{n+1} = \vrm^n + \dt[(1-\gamma)\arm^n + \gamma\arm^{n+1}]`."""

    beta: float = _Interval(0.0, closedInf=False)
    gamma: float = _Interval(0.0)

    def __init__(self, dt: float, beta: float = 1 / 4, gamma: float = 1 / 2):
        super().__init__(dt)
        self.beta = beta
        self.gamma = gamma

    def Evaluate(self, x, u_n, v_n, a_n):
        return self.Evaluate_np1(x, u_n, v_n, a_n)

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        # Hughes 1987 Chapter 9
        dt, beta, gamma = self.dt, self.beta, self.gamma
        ut_np1 = u_n + dt * v_n + dt**2 / 2 * (1 - 2 * beta) * a_n
        vt_np1 = v_n + dt * (1 - gamma) * a_n
        a_np1 = (x - ut_np1) / (beta * dt**2)
        v_np1 = vt_np1 + gamma * dt * a_np1
        return x, v_np1, a_np1
