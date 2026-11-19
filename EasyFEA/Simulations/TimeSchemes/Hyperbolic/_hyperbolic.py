# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from ....Utilities import _params
from .._utils import _TimeScheme


class _Hyperbolic(_TimeScheme):
    r"""A scheme of :math:`\Krm \, \urm_t + \Crm \, \vrm_t + \Mrm \, \arm_t = \Frm`, integrating u, v and a over steps of `dt`."""

    roles = ("u", "v", "a")

    dt: float = _params.StrictlyPositiveScalarParameter()
    """time increment"""

    def __init__(self, dt: float):
        self.dt = dt
