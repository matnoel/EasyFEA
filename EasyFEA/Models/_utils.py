# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from abc import ABC, abstractmethod
from typing import Callable

# utilities
from ..Utilities._observers import Observable
from ..Utilities import _types
from ..Utilities._params import Updatable, _Parameter
from ..FEM._linalg import FeArray
from ..FEM._kelvin_mandel import (
    ORDER,
    IDX,
    R2,
    Weights,
    From_Voigt,
    Vector_to_Matrix,
    Matrix_to_Vector,
    Tensor_to_Kelvin,
    Get_Pmat as Get_Pmat,
    Apply_Pmat as Apply_Pmat,
)
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


__erroDim = "Pay attention to the dimensions of the material constants.\nIf the material constants are in arrays, these arrays must have the same dimension."


def Heterogeneous_Array(array: _types.FloatArray):
    """Builds a heterogeneous array."""

    dimI, dimJ = array.shape

    shapes = [
        np.shape(array[i, j])
        for i in range(dimI)
        for j in range(dimJ)
        if len(np.shape(array[i, j])) > 0
    ]
    if len(shapes) > 0:
        idx = np.argmax([len(shape) for shape in shapes])
        shape = shapes[idx]
    else:
        shape = ()

    shapeNew = list(shape)
    shapeNew.extend(array.shape)

    newArray = np.zeros(shapeNew)

    def SetMat(i, j):
        values = array[i, j]
        if isinstance(values, (int, float)):
            values = np.ones(shape) * values
        if len(shape) == 0:
            newArray[i, j] = values
        elif len(shape) == 1:
            newArray[:, i, j] = values
        elif len(shape) == 2:
            newArray[:, :, i, j] = values
        else:
            raise Exception(
                "The material constants must be of maximum dimension (Ne, nPg)"
            )

    [SetMat(i, j) for i in range(dimI) for j in range(dimJ)]

    return newArray


def KelvinMandel_Matrix(dim: int, M: _types.FloatArray) -> _types.FloatArray:
    """Voigt stiffness → Kelvin–Mandel; alias of :func:`From_Voigt`."""
    return From_Voigt(M)


Project_vector_to_matrix = Vector_to_Matrix
Project_matrix_to_vector = Matrix_to_Vector
Project_Kelvin = Tensor_to_Kelvin


def __Result_in_Strain_or_Stress_field(
    field_e_pg: FeArray.FeArrayALike, result: str, coef=R2
) -> _types.FloatArray:
    """Extracts a specific result from a 2D or 3D strain or stress field.

    Parameters
    ----------
    field_e_pg : _types.FloatArray
        Strain or stress field in each element and gauss points.
    result : str
        Desired result/value to extract:\n
            2D: [xx, yy, xy, vm, Strain, Stress] \n
            3D: [xx, yy, zz, yz, xz, xy, vm, Strain, Stress] \n
    coef : float, optional
        Coefficient used to scale cross components in the field (e.g., xy/coef in dim=2, or yz/coef, xz/coef, xy/coef if dim=3).

    Returns
    -------
    _types.FloatArray
        The extracted field corresponding to the specified result.
    """

    assert isinstance(field_e_pg, FeArray), "must be a FeArray"
    assert field_e_pg._ndim == 1, "must be a vector"

    Ne, nPg = field_e_pg.shape[:2]
    # rescaled below: a caller may hand over the stress it keeps
    field_e_pg = field_e_pg.copy()

    if field_e_pg.shape == (Ne, nPg, 3):
        dim = 2
    elif field_e_pg.shape == (Ne, nPg, 6):
        dim = 3
    else:
        raise Exception("field_e_pg must be of shape (Ne, nPg, 3) or (Ne, nPg, 6)")

    names = [ORDER[i] for i in IDX[dim]]
    field_e_pg[..., Weights(dim) != 1] *= 1 / coef
    values = {name: np.asarray(field_e_pg[:, :, i]) for i, name in enumerate(names)}

    name = next((name for name in names if name in result), None)
    if name is not None:
        result_e_pg = values[name]
    elif "vm" in result:
        xx, yy, zz, yz, xz, xy = [values.get(name, 0.0) for name in ORDER]
        result_e_pg = np.sqrt(
            0.5
            * (
                (xx - yy) ** 2
                + (yy - zz) ** 2
                + (zz - xx) ** 2
                + 6 * (xy**2 + yz**2 + xz**2)
            )
        )
    elif result in ("Strain", "Stress", "Green-Lagrange", "Piola-Kirchhoff"):
        result_e_pg = field_e_pg
    else:
        raise Exception(
            f"result must be in [{', '.join(names)}, vm, Strain, Stress, Green-Lagrange, Piola-Kirchhoff]"
        )

    return np.asarray(result_e_pg)  # type: ignore


def _Field_per_groupElem(
    field_e_pg: Callable[..., FeArray.FeArrayALike], list_groupElem: list
) -> FeArray.FeArrayALike | dict:
    """``field_e_pg(groupElem)`` as an ``FeArray`` on one group, ``{groupElem: FeArray}`` on several."""
    if len(list_groupElem) == 1:
        return field_e_pg(list_groupElem[0])
    return {groupElem: field_e_pg(groupElem) for groupElem in list_groupElem}


def Result_strain_or_stress_field_e(
    field: FeArray.FeArrayALike | dict,
    result: str,
    coef=R2,
) -> _types.FloatArray:
    """Per-element (Ne,) component ``result`` of a strain/stress ``field`` — one ``FeArray`` or ``{groupElem: FeArray}`` concatenated in its order — Gauss points averaged."""
    fields = field.values() if isinstance(field, dict) else [field]
    return np.concatenate(
        [
            np.asarray(
                __Result_in_Strain_or_Stress_field(field_e_pg, result, coef).mean(1)
            )
            for field_e_pg in fields
        ]
    )
