# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from .._utils import _TimeScheme, _Interval


class _Hyperbolic(_TimeScheme):
    r"""A scheme of :math:`\Krm \, \urm_t + \Crm \, \vrm_t + \Mrm \, \arm_t = \Frm`, integrating u, v and a over steps of `dt`."""

    roles = ("u", "v", "a")

    dt: float = _Interval(0.0, closedInf=False, message="Time increment must be > 0")
    """time increment"""

    def __init__(self, dt: float):
        self.dt = dt
