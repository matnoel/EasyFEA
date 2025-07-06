# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from typing import Union

# utilities
import numpy as np

from ._utils import _IModel, ModelType
from ..utilities import _params, _types

# ----------------------------------------------
# Thermal
# ----------------------------------------------


class Thermal(_IModel):
    """Thermal class."""

    __modelType = ModelType.thermal

    @property
    def modelType(self) -> ModelType:
        return Thermal.__modelType

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def thickness(self) -> float:
        return self.__thickness

    def __str__(self) -> str:
        text = f"\n{type(self).__name__} :"
        text += f"\nthermal conductivity (k)  : {self.__k}"
        text += f"\nthermal mass capacity (c) : {self.__c}"
        return text

    def __init__(self, dim: int, k: float, c=0.0, thickness: float = 1.0):
        """Creates a thermal model.

        Parameters
        ----------
        dim : int
            model dimension
        k : float
            thermal conductivity [W m^-1]
        c : float, optional
            mass heat capacity [J K^-1 kg^-1], by default 0.0
        thickness : float, optional
            thickness of part, by default 1.0
        """
        assert dim in [1, 2, 3]
        self.__dim = dim

        self.k = k

        # ThermalModel Anisot with different diffusion coefficients for each direction! k becomes a matrix

        self.c = c

        _params.CheckIsPositive(thickness)
        self.__thickness: float = thickness

        self.Need_Update()

        self.useNumba = False

    @property
    def k(self) -> Union[float, _types.FloatArray]:
        """thermal conductivity [W m^-1]"""
        return self.__k

    @k.setter
    def k(self, value: Union[float, _types.FloatArray]) -> None:
        _params.CheckIsPositive(value)
        self.Need_Update()
        self.__k = value

    @property
    def c(self) -> Union[float, _types.FloatArray]:
        """mass heat capacity [J K^-1 kg^-1]"""
        return self.__c

    @c.setter
    def c(self, value: Union[float, _types.FloatArray]) -> None:
        _params.CheckIsInIntervaloo(value, 0, np.inf)
        self.Need_Update()
        self.__c = value

    @property
    def isHeterogeneous(self) -> bool:
        return isinstance(self.k, np.ndarray) or isinstance(self.c, np.ndarray)
