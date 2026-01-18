# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

# utilities
import numpy as np

from ._utils import _IModel, ModelType
from ..Utilities import _params

# ----------------------------------------------
# Thermal
# ----------------------------------------------


class Thermal(_IModel):
    """Thermal class."""

    @property
    def modelType(self) -> ModelType:
        return ModelType.thermal

    dim: int = _params.ParameterInValues([1])

    # ThermalModel Anisot with different diffusion coefficients for each direction! k becomes a matrix
    k = _params.PositiveParameter()
    """thermal conductivity [W m^-1]"""

    c = _params.PositiveParameter()
    """mass heat capacity [J K^-1 kg^-1]"""

    thickness: float = _params.PositiveScalarParameter()

    def __str__(self) -> str:
        text = f"\n{type(self).__name__} :"
        text += f"\nthermal conductivity (k)  : {self.k}"
        text += f"\nthermal mass capacity (c) : {self.c}"
        return text

    def __init__(self, k: float, c=0.0, thickness: float = 1.0):
        """Creates a thermal model.

        Parameters
        ----------
        k : float
            thermal conductivity [W m^-1]
        c : float, optional
            mass heat capacity [J K^-1 kg^-1], by default 0.0
        thickness : float, optional
            thickness of part, by default 1.0
        """
        self.dim = 1
        self.k = k
        self.c = c
        self.thickness = thickness

    @property
    def isHeterogeneous(self) -> bool:
        return isinstance(self.k, np.ndarray) or isinstance(self.c, np.ndarray)
