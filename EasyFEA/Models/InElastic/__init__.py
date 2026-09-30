# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module implementing constitutive laws used in simulations."""

from ._behavior import (
    _Behavior,
    MaterialPoint,
    Newton,
    Trace,
    Deviator,
    Von_Mises_stress,
    ONE,
    ZERO_TENSOR,
    ZERO_SCALAR,
)
from . import IsotropicHardening
from . import Yield
from ._maxwell import Maxwell
from ._plasticity import Plasticity, Norton, Chaboche
