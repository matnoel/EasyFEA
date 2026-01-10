# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""This module contains all available simulation classes."""

from .simulations._simu import _Simu, Load_Simu
from .simulations._utils import Save_pickle, Load_pickle

# ------------------------------------------------------------------------------
# Elastic
# ------------------------------------------------------------------------------

from .simulations._elastic import Elastic, Mesh_Optim_ZZ1

# ----------------------------------------------
# HyperElastic
# ----------------------------------------------

from .simulations._hyperelastic import HyperElastic

# ------------------------------------------------------------------------------
# PhaseField
# ------------------------------------------------------------------------------

from .simulations._phasefield import PhaseField

# ------------------------------------------------------------------------------
# Beam
# ------------------------------------------------------------------------------

from .simulations._beam import Beam

# ------------------------------------------------------------------------------
# Thermal
# ------------------------------------------------------------------------------

from .simulations._thermal import Thermal

# ------------------------------------------------------------------------------
# WeakForm
# ------------------------------------------------------------------------------

from .simulations._weak_forms import WeakForm

# ------------------------------------------------------------------------------
# DIC
# ------------------------------------------------------------------------------

try:
    from .simulations._dic import DIC, Load_DIC, Get_Circle
except ModuleNotFoundError:
    # you must install opencv-python
    pass
