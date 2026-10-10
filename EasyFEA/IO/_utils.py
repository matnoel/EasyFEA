# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Helpers shared by the IO and Viz front-ends."""

from __future__ import annotations
from functools import singledispatch
import numpy as np

from ..Utilities import _types
from ..FEM import Mesh, _GroupElem
from ..Simulations._simu import _Simu


@singledispatch
def _Init_obj(
    obj: _Simu | Mesh | _GroupElem, deformFactor: float = 0.0
) -> tuple[_Simu | None, Mesh, _types.FloatArray, int]:
    """Returns (simu, mesh, coord, inDim) from an ojbect that could be either a _Simu, a Mesh or a _GroupElem object.

    Parameters
    ----------
    obj : _Simu | Mesh | _GroupElem
        An object that contain the mesh
    deformFactor : float, optional
        the factor used to deform the mesh, by default 0.0

    Returns
    -------
    tuple[_Simu|None, Mesh, ndarray, int]
        (simu, mesh, coord, inDim)
    """
    raise NotImplementedError(
        "obj must be a simulation, a mesh or a group of elements."
    )


@_Init_obj.register
def _(obj: _Simu, deformFactor: float = 0.0):
    simu = obj
    mesh = simu.mesh
    u = simu.Results_displacement_matrix()
    coord: _types.FloatArray = mesh.coord + u * np.abs(deformFactor)
    return simu, mesh, coord, simu.inDim


@_Init_obj.register
def _(obj: Mesh, deformFactor: float = 0.0):
    simu = None
    mesh = obj
    coord = mesh.coord
    inDim = mesh.inDim
    return simu, mesh, coord, inDim


@_Init_obj.register
def _(obj: _GroupElem, deformFactor: float = 0.0):
    simu = None
    mesh = Mesh({obj.elemType: obj})
    coord = mesh.coord
    inDim = mesh.inDim
    return simu, mesh, coord, inDim


def _Get_values(
    simu: _Simu | None,
    mesh: Mesh,
    result: str | _types.AnyArray,
    nodeValues=True,
) -> _types.AnyArray:
    """Retrieves values and ensures compatibility with the mesh.

    Parameters
    ----------
    simu : _Simu | None
        Simulation (can be set to None).
    mesh : Mesh
        Mesh used to display the result.
    result : str | _types.AnyArray
        Result you want to display.
        Must be included in simu.Get_Results() or be a numpy array of size (Nn, Ne).
    nodeValues : bool, optional
        Displays result on nodes; otherwise, displays it on elements. Default is True.

    Returns
    -------
    _types.AnyArray
        values
    """

    Ne = mesh.Ne
    Nn = mesh.Nn

    if isinstance(result, str):
        if simu is None:
            raise Exception(
                "obj is a mesh, so the result must be an array of dimension Nn or Ne"
            )
        values = simu.Result(result, nodeValues)  # Retrieve result from option
        if not isinstance(values, np.ndarray):
            return None  # type: ignore [return-value]

    elif isinstance(result, np.ndarray):
        values = result
        size = result.shape[0]
        if size not in [Ne, Nn]:
            raise Exception("Must be an array of dimension Nn or Ne")
        else:
            if size == mesh.Ne and nodeValues:
                # calculate nodal values for element values
                values = mesh.Get_Node_Values(result)
            elif size == mesh.Nn and not nodeValues:
                # several element groups can share the main dimension (e.g. QUAD4 + TRI3); the per-group mean over nPe collapses the ragged element axis so the result concatenates in Get_list_groupElem(dim) order (matching mesh.Ne)
                values = np.concatenate(
                    [
                        np.mean(groupElem.Locates_sol_e(result), 1)
                        for groupElem in mesh.Get_list_groupElem(mesh.dim)
                    ]
                )
    elif result is None:
        return None
    else:
        raise Exception("result must be a string or an array")

    return values  # type: ignore [return-value]
