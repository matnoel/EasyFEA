# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""This module contains all available simulation classes."""

from .simulations._simu import _Simu, Load_Simu
from .simulations._utils import Save_pickle, Load_pickle

# ------------------------------------------------------------------------------
# Elastic
# ------------------------------------------------------------------------------

from .simulations._elastic import ElasticSimu, Mesh_Optim_ZZ1

# ----------------------------------------------
# HyperElastic
# ----------------------------------------------

from .simulations._hyperelastic import HyperElasticSimu

# ------------------------------------------------------------------------------
# PhaseField
# ------------------------------------------------------------------------------

from .simulations._phasefield import PhaseFieldSimu

# ------------------------------------------------------------------------------
# Beam
# ------------------------------------------------------------------------------

from .simulations._beam import BeamSimu

# ------------------------------------------------------------------------------
# Thermal
# ------------------------------------------------------------------------------

from .simulations._thermal import ThermalSimu

# ------------------------------------------------------------------------------
# WeakForm
# ------------------------------------------------------------------------------

from .simulations._weak_forms import WeakFormSimu

# ------------------------------------------------------------------------------
# DIC
# ------------------------------------------------------------------------------

try:
    from .simulations._dic import DIC, Load_DIC, Get_Circle
except ModuleNotFoundError:
    # you must install opencv-python
    pass
