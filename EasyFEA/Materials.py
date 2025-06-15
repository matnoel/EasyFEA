# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing constitutive laws for materials, including elastic, hyperelastic, damage, thermal, and beam materials."""

from .materials import Reshape_variable

# ----------------------------------------------
# Elastic
# ----------------------------------------------

from .materials._linear_elastic_laws import (
    _Elas,
    ElasIsot,
    ElasIsotTrans,
    ElasAnisot,
)

# ----------------------------------------------
# HyperElastic
# ----------------------------------------------

from .materials._hyperelastic_laws import NeoHookean, MooneyRivlin, SaintVenantKirchhoff
from .materials._hyperelastic import HyperElastic

# ----------------------------------------------
# PhaseField
# ----------------------------------------------

from .materials._phasefield import PhaseField

# ----------------------------------------------
# Beam
# ----------------------------------------------

from .materials._beam import _Beam, BeamStructure, BeamElasIsot

# ----------------------------------------------
# Thermal
# ----------------------------------------------

from .materials._thermal import Thermal
