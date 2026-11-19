# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ._hyperbolic import _Hyperbolic


class Midpoint(_Hyperbolic):
    r"""Midpoint rule, every field at the middle of the step: :math:`\vrm^{n+1} = 2/\dt\,(\urm^{n+1}-\urm^n) - \vrm^n`, :math:`\arm^{n+1} = 2/\dt\,(\vrm^{n+1}-\vrm^n) - \arm^n`."""

    def Evaluate(self, x, u_n, v_n, a_n):
        u_np1, v_np1, a_np1 = self.Evaluate_np1(x, u_n, v_n, a_n)
        return (u_np1 + u_n) / 2, (v_np1 + v_n) / 2, (a_np1 + a_n) / 2

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        dt = self.dt
        v_np1 = 2 / dt * (x - u_n) - v_n
        a_np1 = 2 / dt * (v_np1 - v_n) - a_n
        return x, v_np1, a_np1
