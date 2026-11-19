# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ._hyperbolic import _Hyperbolic


class EulerImplicit(_Hyperbolic):
    r"""Backward Euler (first order, dissipative): :math:`\vrm^{n+1} = (\urm^{n+1}-\urm^n)/\dt`, :math:`\arm^{n+1} = (\vrm^{n+1} - \vrm^n)/\dt`."""

    def Evaluate(self, x, u_n, v_n, a_n):
        return self.Evaluate_np1(x, u_n, v_n, a_n)

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        dt = self.dt
        v_np1 = (x - u_n) / dt
        a_np1 = (v_np1 - v_n) / dt
        return x, v_np1, a_np1
