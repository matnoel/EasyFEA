# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ._hyperbolic import _Hyperbolic


class EulerExplicit(_Hyperbolic):
    r"""Forward Euler (first order, linear only, stable for :math:`\dt < h_e / c`): solves :math:`\Mrm\,\arm^n = \Frm^n - \Crm\vrm^n - \Krm\urm^n`, then :math:`\urm^{n+1} = \urm^n + \dt\vrm^n`, :math:`\vrm^{n+1} = \vrm^n + \dt\arm^n`."""

    unknown = "a"

    def Evaluate(self, x, u_n, v_n, a_n):
        return u_n, v_n, x

    def Evaluate_np1(self, x, u_n, v_n, a_n):
        dt = self.dt
        return u_n + dt * v_n, v_n + dt * x, x
