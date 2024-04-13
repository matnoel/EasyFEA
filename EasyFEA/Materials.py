"""Module containing material behavior. Such as elastic, damage, thermal and beam materials."""

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

from .materials._beam import _Beam, Beam_Structure, Beam_Elas_Isot

# ----------------------------------------------
# Thermal
# ----------------------------------------------

from .materials._thermal import Thermal