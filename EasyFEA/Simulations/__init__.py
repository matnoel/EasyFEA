# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""This module contains all available simulation classes."""

from ._simu import _Simu, Load_Simu
from ._utils import Save_pickle, Load_pickle

# ------------------------------------------------------------------------------
# simulations
# ------------------------------------------------------------------------------
from ._elastic import Elastic, Mesh_Optim_ZZ1
from ._hyperelastic import HyperElastic
from ._phasefield import PhaseField
from ._beam import Beam
from ._thermal import Thermal
from ._weakforms import WeakForms

# ------------------------------------------------------------------------------
# DIC
# ------------------------------------------------------------------------------
try:
    from ._dic import DIC, Load_DIC, Get_Circle
except ModuleNotFoundError:
    # you must install opencv-python
    pass
