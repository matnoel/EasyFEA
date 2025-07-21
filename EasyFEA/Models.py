# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing constitutive laws for materials, including elastic, hyperelastic, damage, thermal, and beam materials."""

from .models import Reshape_variable

# ----------------------------------------------
# Elastic
# ----------------------------------------------

from .models._linear_elastic_laws import (
    _Elas,
    ElasIsot,
    ElasIsotTrans,
    ElasAnisot,
)

# ----------------------------------------------
# HyperElastic
# ----------------------------------------------

from .models._hyperelastic_laws import NeoHookean, MooneyRivlin, SaintVenantKirchhoff
from .models._hyperelastic import HyperElastic

# ----------------------------------------------
# PhaseField
# ----------------------------------------------

from .models._phasefield import PhaseField

# ----------------------------------------------
# Beam
# ----------------------------------------------

from .models._beam import _Beam, BeamStructure, BeamElasIsot

# ----------------------------------------------
# Thermal
# ----------------------------------------------

from .models._thermal import Thermal

# ----------------------------------------------
# WeakForms
# ----------------------------------------------

from .models._weak_forms import WeakForms
