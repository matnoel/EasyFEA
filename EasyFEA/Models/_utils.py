# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
from functools import cached_property

# utilities
from ..Utilities._observers import Observable
from ..Utilities import _types
from ..Utilities._params import Updatable, _Parameter
import numpy as np

# pyright: reportPossiblyUnboundVariable=false

# ----------------------------------------------
# Types
# ----------------------------------------------


class _IModel(Observable, Updatable, ABC):
    """Model interface."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """model dimension"""

    @property
    @abstractmethod
    def thickness(self) -> float:
        """thickness used in the model"""

    def Need_Update(self, value=True):
        super().Need_Update(value)
        if value:
            for cls in type(self).__mro__:
                for name, attr in vars(cls).items():
                    if isinstance(attr, cached_property):
                        self.__dict__.pop(name, None)
            self._Notify("The model has been modified.")

    @property
    def isHeterogeneous(self) -> bool:
        """indicates whether the model has heterogeneous parameters"""
        return False

    def _Get_parameters(self) -> dict[str, object]:
        """Parameters declared as ``_params`` descriptors, in declaration order, base classes first."""
        parameters: dict[str, object] = {}
        for cls in reversed(type(self).__mro__):
            for name, attr in vars(cls).items():
                if (
                    isinstance(attr, _Parameter)
                    and not name.startswith("_")
                    and name in self.__dict__
                ):
                    parameters[name] = getattr(self, name)
        if self.dim == 3:
            parameters.pop("thickness", None)
            parameters.pop("planeStress", None)
        return parameters

    def __str__(self) -> str:
        text = f"{type(self).__name__}:"
        for name, value in self._Get_parameters().items():
            text += f"\n{name} = {_Format_parameter(value)}"
        return text


# ----------------------------------------------
# Functions
# ----------------------------------------------


def _Format_parameter(value) -> str:
    """A scalar or a small array in full, a field as its shape and range."""
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, (int, float, np.number)):
        return f"{value:.4g}"
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if value.ndim == 0:
            return f"{value.item():.4g}"
        if value.size <= 9:
            return np.array_str(value, precision=4)
        return f"{value.shape} in [{value.min():.4g}, {value.max():.4g}]"
    return str(value)


def Heterogeneous_Array(array) -> _types.FloatArray:
    """(…, I, J) float array from an (I, J) nested list or object array of scalars and arrays broadcasting together."""
    rows = [[np.asarray(value, dtype=float) for value in row] for row in array]
    shape = np.broadcast_shapes(*(value.shape for row in rows for value in row))
    return np.stack(
        [
            np.stack([np.broadcast_to(value, shape) for value in row], -1)
            for row in rows
        ],
        -2,
    )
