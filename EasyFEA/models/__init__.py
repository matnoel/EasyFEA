# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from . import Beam
from . import Elastic
from . import HyperElastic
from ._phasefield import PhaseField
from ._thermal import Thermal
from ._weakforms import WeakForms

from ._utils import (
    ModelType,
    _IModel,
    Reshape_variable,
    Heterogeneous_Array,
    KelvinMandel_Matrix,
    Project_vector_to_matrix,
    Project_matrix_to_vector,
    Project_Kelvin,
    Result_in_Strain_or_Stress_field,
    Get_Pmat,
    Apply_Pmat,
)
