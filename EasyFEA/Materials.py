# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

"""Module containing constitutive laws for materials, including elastic, damage, thermal, and beam materials."""

from .materials import Reshape_variable

# ----------------------------------------------
# Elastic
# ----------------------------------------------

from .materials._elastic import _Elas, Elas_Isot, Elas_IsotTrans, Elas_Anisot

# ----------------------------------------------
# PhaseField
# ----------------------------------------------

from .materials._phasefield import PhaseField

# ----------------------------------------------
# Beam
# ----------------------------------------------

from .materials._beam import _Beam, BeamStructure, Beam_Elas_Isot

# ----------------------------------------------
# Thermal
# ----------------------------------------------

from .materials._thermal import Thermal