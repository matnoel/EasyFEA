# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3 or later, see LICENSE.txt and CREDITS.md for more information.

from ._utils import (ModelType, _IModel,
                     Reshape_variable, Heterogeneous_Array,
                     Tensor_Product,
                     KelvinMandel_Matrix, Project_Kelvin,
                     Result_in_Strain_or_Stress_field,
                     Get_Pmat, Apply_Pmat)